"""
VLA Interface: abstract base class + concrete implementations.

Designed to be VLA-agnostic — the Mamba supervisor only sees VLAOutput,
never the internals of any specific VLA model.

Implementations:
  - VLAInterface     (ABC)         — defines the contract
  - MockVLAInterface (test/dev)    — deterministic mock with optional drift injection
  - OpenVLAInterface (production)  — openvla/openvla-7b via HuggingFace Transformers
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812

from models import ObservationBundle, VLAOutput


class VLAInterface(abc.ABC):
    """
    Abstract interface for any Vision-Language-Action model.

    VLAs are queried infrequently (every N motor steps or on supervisor trigger).
    Each query takes an ObservationBundle and returns a VLAOutput containing the
    semantic intent embedding the Mamba supervisor monitors.
    """

    @abc.abstractmethod
    def query(self, obs: ObservationBundle) -> VLAOutput:
        """
        Run a full VLA forward pass.

        This is the expensive operation (100ms–2s depending on hardware/model size).
        Returns VLAOutput with populated task_embedding and visual_context.
        """
        ...

    @abc.abstractmethod
    def get_embedding_dim(self) -> int:
        """Returns the dimensionality of VLAOutput.task_embedding."""
        ...

    @abc.abstractmethod
    def get_visual_context_dim(self) -> int:
        """Returns the dimensionality of VLAOutput.visual_context."""
        ...

    @abc.abstractmethod
    def reset(self):
        """Clear any cached internal state. Call at the start of each episode."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'openvla-7b' or 'mock'."""
        ...


class MockVLAInterface(VLAInterface):
    """
    Deterministic mock for development and testing — no GPU or model weights required.

    By default emits a fixed random embedding seeded by `seed`.
    Set `drift_at_step` to inject a simulated semantic shift at that timestep,
    which is the primary mechanism for testing the supervisor's deviation detection.

    Example:
        vla = MockVLAInterface(embedding_dim=512, drift_at_step=50, drift_magnitude=2.0)
        # Steps 0-49: consistent embedding
        # Steps 50+:  embedding shifts by drift_magnitude in a random direction
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        visual_context_dim: int = 512,
        drift_at_step: Optional[int] = None,
        drift_magnitude: float = 2.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        self._embedding_dim = embedding_dim
        self._visual_context_dim = visual_context_dim
        self.drift_at_step = drift_at_step
        self.drift_magnitude = drift_magnitude
        self.device = torch.device(device)

        # Reproducible base embeddings
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._base_task_embedding = F.normalize(
            torch.randn(embedding_dim, generator=rng), dim=0
        ).to(self.device)
        self._base_visual_context = F.normalize(
            torch.randn(visual_context_dim, generator=rng), dim=0
        ).to(self.device)

        # Drift target: orthogonal-ish direction for a measurable cosine gap
        rng_drift = torch.Generator()
        rng_drift.manual_seed(seed + 1)
        self._drift_task_embedding = F.normalize(
            torch.randn(embedding_dim, generator=rng_drift), dim=0
        ).to(self.device)
        self._drift_visual_context = F.normalize(
            torch.randn(visual_context_dim, generator=rng_drift), dim=0
        ).to(self.device)

        self._query_count = 0

    @property
    def name(self) -> str:
        return "mock"

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def get_visual_context_dim(self) -> int:
        return self._visual_context_dim

    def reset(self):
        self._query_count = 0

    def query(self, obs: ObservationBundle) -> VLAOutput:
        self._query_count += 1
        drifted = self.drift_at_step is not None and obs.timestep >= self.drift_at_step

        task_embedding = self._drift_task_embedding if drifted else self._base_task_embedding
        visual_context = self._drift_visual_context if drifted else self._base_visual_context

        return VLAOutput(
            task_embedding=task_embedding.clone(),
            visual_context=visual_context.clone(),
            raw_action_tokens=None,
            query_timestamp=time.time(),
            query_timestep=obs.timestep,
            metadata={"drifted": drifted, "query_count": self._query_count},
        )


class OpenVLAInterface(VLAInterface):
    """
    OpenVLA integration via HuggingFace Transformers.

    Loads openvla/openvla-7b (or any compatible path) and extracts:
      - task_embedding: last-layer hidden states pooled over action token positions
        Shape: (4096,) for the 7B LLaMA backbone
      - visual_context: vision encoder projection tokens, mean-pooled
        Shape: (4096,) projected

    Requires: pip install transformers>=4.40 and model weights (~15GB).
    """

    # LLaMA-2 7B hidden dimension (same for OpenVLA's language backbone)
    _EMBEDDING_DIM = 4096
    _VISUAL_CONTEXT_DIM = 4096

    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "transformers is required for OpenVLAInterface. "
                "Install with: pip install transformers>=4.40"
            ) from e

        self._model_path = model_path
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Hook storage for visual token extraction
        self._visual_tokens: Optional[torch.Tensor] = None
        self._register_visual_hook()

    def _register_visual_hook(self):
        """Capture projected vision encoder tokens from the language model's input embeddings."""

        def _hook(module, input, output):
            # output: (B, seq_len, hidden_dim) — first visual_token_count tokens are vision
            self._visual_tokens = output.detach()

        # Hook on the vision-language projection layer (varies by architecture)
        # OpenVLA uses Prismatic: self.model.language_model.model.embed_tokens
        try:
            self.model.language_model.model.embed_tokens.register_forward_hook(_hook)
        except AttributeError:
            # Fallback: hook on whatever the first embedding layer is
            pass

    @property
    def name(self) -> str:
        return f"openvla:{self._model_path.split('/')[-1]}"

    def get_embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def get_visual_context_dim(self) -> int:
        return self._VISUAL_CONTEXT_DIM

    def reset(self):
        self._visual_tokens = None

    @torch.no_grad()
    def query(self, obs: ObservationBundle) -> VLAOutput:
        """
        Run a full OpenVLA forward pass and extract embeddings.

        Steps:
        1. Format primary image + task_language through processor
        2. Run model.generate() with output_hidden_states=True
        3. Pool last hidden layer at action token positions -> task_embedding
        4. Pool visual tokens from hook -> visual_context
        """
        from PIL import Image as PILImage

        # Use the first camera image; convert to PIL for the processor
        img_tensor = obs.images[0]  # (B, C, H, W) or (C, H, W)
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]  # take batch[0]
        # (C, H, W) uint8 or float -> PIL
        if img_tensor.dtype != torch.uint8:
            img_tensor = (img_tensor.clamp(0, 1) * 255).byte()
        pil_image = PILImage.fromarray(
            img_tensor.permute(1, 2, 0).cpu().numpy()
        )

        # Processor encodes image + instruction
        inputs = self.processor(
            images=pil_image,
            text=obs.task_language,
            return_tensors="pt",
        ).to(self.device, dtype=self.torch_dtype)

        # Generate action tokens with hidden states
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=7,  # OpenVLA outputs 7 action tokens by default
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Extract task embedding from last-layer hidden states at action token positions
        # hidden_states: tuple of (num_generated_steps,) each (num_layers+1, B, seq_len, hidden_dim)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            last_step_hidden = outputs.hidden_states[-1]  # last generated token
            last_layer = last_step_hidden[-1]              # last transformer layer
            task_embedding = last_layer[0].mean(dim=0)     # mean over seq -> (hidden_dim,)
        else:
            task_embedding = torch.zeros(self._EMBEDDING_DIM, device=self.device)

        # Extract visual context from hook-captured tokens
        if self._visual_tokens is not None:
            # Mean-pool over all token positions
            visual_context = self._visual_tokens[0].mean(dim=0)  # (hidden_dim,)
        else:
            visual_context = torch.zeros(self._VISUAL_CONTEXT_DIM, device=self.device)

        # Raw action token IDs
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]

        return VLAOutput(
            task_embedding=task_embedding.float(),
            visual_context=visual_context.float(),
            raw_action_tokens=generated_ids.float().unsqueeze(-1),
            query_timestamp=time.time(),
            query_timestep=obs.timestep,
            metadata={"model": self.name},
        )
