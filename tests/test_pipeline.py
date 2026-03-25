"""
Integration tests for VLAMambaACTPipeline.

Uses MockVLAInterface + a trivial gymnasium env to test pipeline logic
without any model weights or GPU.
"""

import pytest  # type: ignore[import-untyped]
import gymnasium as gym  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import torch

from models import ExecutionState, ObservationBundle
from models.vla_interface import MockVLAInterface
from models.mamba_supervisor import MambaSupervisor, MambaSupervisorConfig
from scripts.pipeline import VLAMambaACTPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Minimal mock ACT decoder (avoids loading any LeRobot weights)
# ---------------------------------------------------------------------------

class _MockACTDecoder:
    """Minimal stand-in for ACTDecoder for pipeline tests."""

    class config:
        robot_state_dim = 6
        image_shape = (3, 64, 64)
        camera_keys = ["observation.images.top"]
        n_action_steps = 1
        chunk_size = 10

    def __init__(self):
        self._features = torch.zeros(1, 64)
        self._call_count = 0

    def predict_chunk(self, obs):
        self._call_count += 1
        self._features = torch.randn(1, 64)
        # Return a (1, chunk_size, action_dim) tensor
        return torch.zeros(1, self.config.chunk_size, self.config.robot_state_dim)

    def get_action_at(self, chunk, position):
        return chunk[:, position, :]

    def get_encoder_features(self):
        return self._features

    def reset(self):
        self._call_count = 0
        self._features = None


def _make_supervisor(visual_dim=64, state_dim=6, vla_dim=64, **kwargs):
    cfg = MambaSupervisorConfig(
        vla_embedding_dim=vla_dim,
        visual_feature_dim=visual_dim,
        robot_state_dim=state_dim,
        input_proj_dim=32,
        mamba_d_model=32,
        mamba_n_layers=2,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_d_conv=4,
        **kwargs,
    )
    sup = MambaSupervisor(cfg)
    sup.eval()
    return sup


def _make_pipeline(supervisor_enabled=True, **vla_kwargs):
    vla = MockVLAInterface(embedding_dim=64, visual_context_dim=64, **vla_kwargs)
    act = _MockACTDecoder()
    supervisor = _make_supervisor()
    cfg = PipelineConfig(
        supervisor_enabled=supervisor_enabled,
        vla_query_interval=10,
        n_action_steps=1,
        device="cpu",
    )
    pipeline = VLAMambaACTPipeline(vla, act, supervisor, cfg)  # type: ignore[arg-type]
    pipeline.device = torch.device("cpu")
    return pipeline, vla, act, supervisor


def _make_obs(timestep=0):
    return ObservationBundle(
        images=[torch.zeros(1, 3, 64, 64)],
        robot_state=torch.zeros(1, 6),
        task_language="test task",
        timestep=timestep,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineReset:
    def test_reset_returns_execution_state(self):
        pipeline, *_ = _make_pipeline()
        state = pipeline.reset("test task")
        assert isinstance(state, ExecutionState)
        assert state.current_vla_output is None
        assert state.current_action_chunk is None
        assert state.chunk_position == 0

    def test_reset_initializes_mamba_hidden_state(self):
        pipeline, *_ = _make_pipeline()
        state = pipeline.reset("test task")
        assert state.mamba_ssm_state is not None
        assert isinstance(state.mamba_ssm_state, dict)


class TestSupervisedStep:
    def test_step_returns_action_tensor(self):
        pipeline, _, act, _ = _make_pipeline()
        state = pipeline.reset("test")
        obs = _make_obs(timestep=0)
        action, state, decision = pipeline.step(obs, state)
        assert isinstance(action, torch.Tensor)
        assert action.shape == (act.config.robot_state_dim,)

    def test_first_step_triggers_vla_query(self):
        pipeline, vla, _, _ = _make_pipeline()
        state = pipeline.reset("test")
        obs = _make_obs(timestep=0)
        pipeline.step(obs, state)
        assert vla._query_count == 1

    def test_chunk_position_advances(self):
        pipeline, _, _, _ = _make_pipeline()
        state = pipeline.reset("test")
        for t in range(3):
            obs = _make_obs(timestep=t)
            _, state, _ = pipeline.step(obs, state)
        # With n_action_steps=1, each step re-predicts — position resets each time
        assert state.chunk_position == 1


class TestBaselineMode:
    def test_baseline_vla_queries_at_interval(self):
        pipeline, vla, _, _ = _make_pipeline(supervisor_enabled=False)
        cfg = pipeline.config
        state = pipeline.reset("test")

        total_steps = 35
        for t in range(total_steps):
            obs = _make_obs(timestep=t)
            _, state, decision = pipeline.step(obs, state)
            assert decision is None  # supervisor not active

        # Initial query at t=0 + interval queries
        expected_queries = 1 + (total_steps - 1) // cfg.vla_query_interval
        assert vla._query_count == expected_queries

    def test_baseline_decision_is_none(self):
        pipeline, _, _, _ = _make_pipeline(supervisor_enabled=False)
        state = pipeline.reset("test")
        _, _, decision = pipeline.step(_make_obs(), state)
        assert decision is None


class TestCooldownIntegration:
    def test_supervisor_cooldown_limits_vla_queries(self):
        # Use drift at every step to maximize re-query pressure
        pipeline, vla, _, _ = _make_pipeline(drift_at_step=0, drift_magnitude=3.0)
        state = pipeline.reset("test")

        total_steps = 100
        for t in range(total_steps):
            obs = _make_obs(timestep=t)
            _, state, _ = pipeline.step(obs, state)

        cooldown = pipeline.supervisor.config.cooldown_timesteps
        # Queries should be bounded by cooldown (plus initial)
        max_queries = 1 + total_steps // cooldown
        assert vla._query_count <= max_queries, (
            f"Too many VLA queries: {vla._query_count} > {max_queries}"
        )
