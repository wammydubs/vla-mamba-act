"""
ACT Decoder: wraps LeRobot's ACTPolicy with a lifecycle-aware interface.

Key design: exposes the full action chunk and current position so the Mamba
supervisor can interrupt mid-chunk — ACTPolicy.select_action() manages an
internal queue and we bypass it entirely with predict_action_chunk().
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# Make lerobot importable from the local submodule
_LEROBOT_SRC = Path(__file__).resolve().parent.parent / "lerobot" / "src"
if str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from models import ObservationBundle


@dataclass
class ACTDecoderConfig:
    """Configuration for the ACT Decoder wrapper."""

    # HuggingFace Hub model ID or local path to pretrained ACT weights
    pretrained_path: str = "lerobot/act_aloha_sim_transfer_cube_human"
    # Number of actions in one predicted chunk
    chunk_size: int = 100
    # How many actions to execute before re-predicting (must be 1 with temporal ensembling)
    n_action_steps: int = 1
    # Temporal ensembling coefficient (None = disabled)
    temporal_ensemble_coeff: Optional[float] = 0.01
    # Whether to use the VAE encoder
    use_vae: bool = True
    # Inference device
    device: str = "cuda"
    # Proprioceptive state dimension (e.g. 14 for 7-DOF arm + gripper)
    robot_state_dim: int = 14
    # Action dimension
    action_dim: int = 14
    # Image shape (C, H, W)
    image_shape: tuple = (3, 480, 640)
    # Camera key names in the observation
    camera_keys: list = field(default_factory=lambda: ["observation.images.top"])


class ACTDecoder:
    """
    Lifecycle-aware wrapper around LeRobot's ACTPolicy.

    The central design choice: predict_chunk() returns the full (chunk_size, action_dim)
    prediction and does NOT consume it. The caller tracks chunk_position in ExecutionState
    and calls get_action_at() each step. This lets the Mamba supervisor interrupt at any
    point in chunk execution.

    With temporal ensembling enabled, ACT runs every step — so get_encoder_features() 
    returns fresh features every call.
    """

    def __init__(self, config: ACTDecoderConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Build ACTConfig from our wrapper config
        input_features = {}
        for key in config.camera_keys:
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=config.image_shape)
        input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE, shape=(config.robot_state_dim,)
        )
        output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(config.action_dim,))
        }

        act_config = ACTConfig(
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            temporal_ensemble_coeff=config.temporal_ensemble_coeff,
            use_vae=config.use_vae,
            input_features=input_features,
            output_features=output_features,
        )

        self.policy = ACTPolicy(act_config)
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal ensembler reference (used in get_action_at with ensembling)
        self._temporal_ensembler: Optional[ACTTemporalEnsembler] = (
            self.policy.temporal_ensembler if config.temporal_ensemble_coeff is not None else None
        )

        # Hook storage for encoder feature extraction
        self._encoder_features: Optional[Tensor] = None
        self._register_encoder_hook()

    def _register_encoder_hook(self):
        """Register a forward hook on the ACT transformer encoder to capture features."""

        def _hook(module, input, output):
            # output shape: (seq_len, B, dim_model) — pool over seq_len for a flat vector
            # ACT encoder output is (seq_len, batch, dim_model)
            if isinstance(output, tuple):
                features = output[0]
            else:
                features = output
            # Mean pool over sequence dim -> (B, dim_model)
            self._encoder_features = features.mean(dim=0).detach()

        self.policy.model.encoder.register_forward_hook(_hook)

    def build_batch(self, obs: ObservationBundle) -> dict[str, Tensor]:
        """
        Convert an ObservationBundle to the LeRobot batch dict format.

        Maps obs.images -> individual camera keys (e.g. "observation.images.top")
        Maps obs.robot_state -> OBS_STATE key
        """
        batch: dict[str, Tensor] = {}

        for i, key in enumerate(self.config.camera_keys):
            if i < len(obs.images):
                batch[key] = obs.images[i].to(self.device)
            else:
                raise ValueError(
                    f"ObservationBundle has {len(obs.images)} images but "
                    f"config expects {len(self.config.camera_keys)} cameras."
                )

        batch[OBS_STATE] = obs.robot_state.to(self.device)
        return batch

    @torch.no_grad()
    def predict_chunk(self, obs: ObservationBundle) -> Tensor:
        """
        Run a full ACT forward pass and return the raw action chunk.

        Returns: (B, chunk_size, action_dim)

        Does NOT consume the chunk — caller tracks position in ExecutionState.chunk_position.
        Also triggers the encoder feature hook, making get_encoder_features() valid after this call.
        """
        batch = self.build_batch(obs)
        self.policy.eval()
        return self.policy.predict_action_chunk(batch)

    def get_action_at(self, chunk: Tensor, position: int) -> Tensor:
        """
        Return the action at `position` in the chunk.

        Args:
            chunk: (B, chunk_size, action_dim) full predicted chunk
            position: current index into the chunk

        Returns: (B, action_dim)

        When temporal ensembling is disabled, this is a simple index.
        When temporal ensembling is enabled, position tracking is managed externally
        and this returns chunk[:, position, :] — the ensembling happens inside
        predict_chunk() -> select_action() in that mode. We expose both paths.
        """
        if position >= chunk.shape[1]:
            raise IndexError(
                f"position {position} is out of bounds for chunk of size {chunk.shape[1]}"
            )
        return chunk[:, position, :]

    def get_encoder_features(self) -> Optional[Tensor]:
        """
        Return the most recently captured ACT encoder output features.

        Shape: (B, dim_model) — mean-pooled over the sequence dimension.
        Valid only after at least one call to predict_chunk().

        Returns None if predict_chunk() has not been called yet.
        """
        return self._encoder_features

    def reset(self):
        """Reset internal ACT state. Call at the start of each episode."""
        self.policy.reset()
        self._encoder_features = None

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "ACTDecoder":
        """Convenience constructor that loads pretrained weights from Hub or disk."""
        config = ACTDecoderConfig(pretrained_path=pretrained_path, **kwargs)
        instance = cls(config)
        # Load pretrained weights via LeRobot's from_pretrained
        instance.policy = ACTPolicy.from_pretrained(pretrained_path)
        instance.policy.to(instance.device)
        instance.policy.eval()
        instance._register_encoder_hook()
        return instance