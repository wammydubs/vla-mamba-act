"""Tests for the VLA interface (mock-only, no GPU required)."""

import time

import pytest  # type: ignore[import-untyped]
import torch

from models import ObservationBundle, VLAOutput
from models.vla_interface import MockVLAInterface, VLAInterface


def _make_obs(timestep: int = 0) -> ObservationBundle:
    return ObservationBundle(
        images=[torch.zeros(1, 3, 64, 64)],
        robot_state=torch.zeros(1, 14),
        task_language="Pick up the cube.",
        timestep=timestep,
    )


class TestMockVLAInterface:
    def test_returns_correct_embedding_dim(self):
        vla = MockVLAInterface(embedding_dim=256, visual_context_dim=128)
        assert vla.get_embedding_dim() == 256
        assert vla.get_visual_context_dim() == 128

    def test_query_returns_vla_output(self):
        vla = MockVLAInterface(embedding_dim=512)
        out = vla.query(_make_obs())
        assert isinstance(out, VLAOutput)
        assert out.task_embedding.shape == (512,)
        assert out.visual_context.shape == (512,)
        assert out.raw_action_tokens is None

    def test_query_is_deterministic_before_drift(self):
        vla = MockVLAInterface(embedding_dim=64, seed=0)
        out1 = vla.query(_make_obs(timestep=0))
        out2 = vla.query(_make_obs(timestep=0))
        assert torch.allclose(out1.task_embedding, out2.task_embedding)

    def test_drift_injection_changes_embedding(self):
        vla = MockVLAInterface(embedding_dim=64, drift_at_step=10, seed=0)
        pre_drift = vla.query(_make_obs(timestep=5))
        post_drift = vla.query(_make_obs(timestep=10))
        # Embeddings should differ after drift
        assert not torch.allclose(pre_drift.task_embedding, post_drift.task_embedding)

    def test_drift_cosine_similarity_drops(self):
        vla = MockVLAInterface(embedding_dim=256, drift_at_step=10, seed=42)
        pre = vla.query(_make_obs(timestep=0)).visual_context
        post = vla.query(_make_obs(timestep=10)).visual_context
        import torch.nn.functional as F
        sim = F.cosine_similarity(pre.unsqueeze(0), post.unsqueeze(0))
        # Drift should produce a meaningfully different direction
        assert sim.item() < 0.95

    def test_reset_clears_query_count(self):
        vla = MockVLAInterface()
        vla.query(_make_obs())
        vla.query(_make_obs())
        vla.reset()
        out = vla.query(_make_obs())
        assert out.metadata["query_count"] == 1

    def test_name_property(self):
        assert MockVLAInterface().name == "mock"

    def test_query_timestamp_is_recent(self):
        vla = MockVLAInterface()
        before = time.time()
        out = vla.query(_make_obs())
        after = time.time()
        assert before <= out.query_timestamp <= after

    def test_query_timestep_matches_obs(self):
        vla = MockVLAInterface()
        out = vla.query(_make_obs(timestep=42))
        assert out.query_timestep == 42


class TestVLAInterfaceABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            VLAInterface()  # type: ignore[abstract]

    def test_concrete_must_implement_all_methods(self):
        class Incomplete(VLAInterface):
            # Missing: get_embedding_dim, get_visual_context_dim, reset, name
            def query(self, obs):  # type: ignore[override]
                return None  # type: ignore[return-value]

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
