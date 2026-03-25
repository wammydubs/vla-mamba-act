"""
Tests for MambaSupervisor.

These run without a real environment or VLA — everything is mocked tensors.
mamba-ssm must be installed (pip install mamba-ssm).
"""

import pytest  # type: ignore[import-untyped]
import torch

from models import SupervisorDecision, VLAOutput
from models.mamba_supervisor import MambaSupervisor, MambaSupervisorConfig


def _make_config(**kwargs) -> MambaSupervisorConfig:
    defaults = dict(
        vla_embedding_dim=64,
        visual_feature_dim=64,
        robot_state_dim=6,
        input_proj_dim=32,
        mamba_d_model=32,
        mamba_n_layers=2,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_d_conv=4,
        deviation_metric="cosine",
        soft_threshold=0.15,
        hard_threshold=0.40,
        cooldown_timesteps=20,
    )
    defaults.update(kwargs)
    return MambaSupervisorConfig(**defaults)  # type: ignore[arg-type]


def _make_vla_output(dim: int = 64, device="cpu") -> VLAOutput:
    task_emb = torch.randn(dim)
    vis_ctx = torch.randn(dim)
    return VLAOutput(
        task_embedding=task_emb,
        visual_context=vis_ctx,
        raw_action_tokens=None,
    )


@pytest.fixture
def supervisor():
    cfg = _make_config()
    sup = MambaSupervisor(cfg)
    sup.eval()
    return sup


class TestHiddenState:
    def test_initial_hidden_state_has_correct_keys(self, supervisor):
        h = supervisor.get_initial_hidden_state(batch_size=1)
        for i in range(supervisor.config.mamba_n_layers):
            assert f"layer_{i}_ssm" in h
            assert f"layer_{i}_conv" in h

    def test_initial_hidden_state_is_zeros(self, supervisor):
        h = supervisor.get_initial_hidden_state(batch_size=2)
        for v in h.values():
            assert v.sum() == 0.0

    def test_reset_returns_dict(self, supervisor):
        h = supervisor.reset(batch_size=1)
        assert isinstance(h, dict)
        assert len(h) == supervisor.config.mamba_n_layers * 2


class TestSingleStep:
    def test_step_returns_supervisor_decision(self, supervisor):
        h = supervisor.reset()
        vla_out = _make_vla_output(dim=64)
        visual = torch.randn(1, 64)
        robot = torch.randn(1, 6)

        decision = supervisor.step(visual, robot, vla_out, h, timestep=1, last_vla_query_timestep=0)
        assert isinstance(decision, SupervisorDecision)

    def test_action_is_valid_label(self, supervisor):
        h = supervisor.reset()
        vla_out = _make_vla_output()
        decision = supervisor.step(
            torch.randn(1, 64), torch.randn(1, 6), vla_out, h, timestep=5, last_vla_query_timestep=0
        )
        assert decision.action in {"continue", "soft_intervene", "hard_intervene"}

    def test_confidence_in_unit_interval(self, supervisor):
        h = supervisor.reset()
        vla_out = _make_vla_output()
        decision = supervisor.step(
            torch.randn(1, 64), torch.randn(1, 6), vla_out, h, timestep=5, last_vla_query_timestep=0
        )
        assert 0.0 <= decision.confidence <= 1.0

    def test_deviation_score_nonnegative(self, supervisor):
        h = supervisor.reset()
        vla_out = _make_vla_output()
        decision = supervisor.step(
            torch.randn(1, 64), torch.randn(1, 6), vla_out, h, timestep=1, last_vla_query_timestep=0
        )
        assert decision.deviation_score >= 0.0

    def test_trigger_requery_consistent_with_action(self, supervisor):
        h = supervisor.reset()
        vla_out = _make_vla_output()
        decision = supervisor.step(
            torch.randn(1, 64), torch.randn(1, 6), vla_out, h, timestep=50, last_vla_query_timestep=0
        )
        if decision.action == "hard_intervene":
            assert decision.trigger_requery is True
        else:
            assert decision.trigger_requery is False


class TestCooldown:
    def test_cooldown_prevents_hard_intervene(self):
        cfg = _make_config(
            hard_threshold=0.0,   # force hard_intervene from thresholds
            soft_threshold=0.0,
            cooldown_timesteps=50,
        )
        sup = MambaSupervisor(cfg)
        sup.eval()
        h = sup.reset()
        vla_out = _make_vla_output()

        # Within cooldown window (timestep=5, last_query=0, cooldown=50)
        decision = sup.step(
            torch.randn(1, 64), torch.randn(1, 6), vla_out, h,
            timestep=5,
            last_vla_query_timestep=0,
        )
        # hard_intervene should be downgraded to soft_intervene
        assert decision.action != "hard_intervene", (
            "Supervisor must not hard_intervene within cooldown window"
        )

    def test_outside_cooldown_allows_hard_intervene(self):
        cfg = _make_config(
            hard_threshold=0.0,   # any deviation triggers hard
            soft_threshold=0.0,
            cooldown_timesteps=5,
        )
        sup = MambaSupervisor(cfg)
        sup.eval()
        h = sup.reset()

        # Use orthogonal vectors to guarantee maximum cosine deviation (=1.0)
        visual = torch.zeros(1, 64)
        visual[0, 0] = 1.0
        vla_ctx = torch.zeros(64)
        vla_ctx[1] = 1.0
        vla_out = VLAOutput(
            task_embedding=torch.randn(64),
            visual_context=vla_ctx,
            raw_action_tokens=None,
        )

        # Outside cooldown (timestep=100, last_query=0, cooldown=5)
        decision = sup.step(
            visual, torch.randn(1, 6), vla_out, h,
            timestep=100,
            last_vla_query_timestep=0,
        )
        assert decision.action == "hard_intervene"


class TestDeviationMetric:
    def test_cosine_deviation_near_zero_for_identical_features(self):
        cfg = _make_config(deviation_metric="cosine")
        sup = MambaSupervisor(cfg)
        sup.eval()

        reference = torch.randn(64)
        reference = reference / reference.norm()
        visual = reference.unsqueeze(0)  # (1, 64)
        vla_out = VLAOutput(task_embedding=torch.randn(64), visual_context=reference, raw_action_tokens=None)

        h = sup.reset()
        decision = sup.step(visual, torch.randn(1, 6), vla_out, h, timestep=1, last_vla_query_timestep=-999)
        assert decision.deviation_score < 0.05, f"Expected near-zero deviation, got {decision.deviation_score}"

    def test_cosine_deviation_near_one_for_orthogonal_features(self):
        cfg = _make_config(deviation_metric="cosine", hard_threshold=2.0)  # disable auto-intervention
        sup = MambaSupervisor(cfg)
        sup.eval()

        visual = torch.zeros(1, 64)
        visual[0, 0] = 1.0
        ref = torch.zeros(64)
        ref[1] = 1.0  # orthogonal to visual
        vla_out = VLAOutput(task_embedding=torch.randn(64), visual_context=ref, raw_action_tokens=None)

        h = sup.reset()
        decision = sup.step(visual, torch.randn(1, 6), vla_out, h, timestep=1, last_vla_query_timestep=-999)
        assert decision.deviation_score > 0.9, f"Expected near-1 deviation, got {decision.deviation_score}"


class TestTrainingForward:
    def test_forward_decision_logits_shape(self):
        cfg = _make_config()
        sup = MambaSupervisor(cfg)
        B, T = 2, 10
        # Prepare a random sequence
        v = torch.randn(B, T, 64)
        s = torch.randn(B, T, 6)
        t = torch.randn(B, T, 64)
        c = torch.randn(B, T, 64)
        seq = sup.prepare_sequence(v, s, t, c)

        _, logits = sup.forward(seq)
        assert logits.shape == (B, T, 3)

    def test_forward_deviation_head_when_learned_mlp(self):
        cfg = _make_config(deviation_metric="learned_mlp")
        sup = MambaSupervisor(cfg)
        B, T = 2, 10
        v = torch.randn(B, T, 64)
        s = torch.randn(B, T, 6)
        t = torch.randn(B, T, 64)
        c = torch.randn(B, T, 64)
        seq = sup.prepare_sequence(v, s, t, c)

        dev_scores, logits = sup.forward(seq)
        assert dev_scores is not None
        assert dev_scores.shape == (B, T, 1)
        assert (dev_scores >= 0).all() and (dev_scores <= 1).all()

    def test_forward_deviation_head_is_none_for_cosine(self):
        cfg = _make_config(deviation_metric="cosine")
        sup = MambaSupervisor(cfg)
        seq = torch.randn(1, 5, 4 * 32)  # 4 * input_proj_dim
        dev_scores, _ = sup.forward(seq)
        assert dev_scores is None
