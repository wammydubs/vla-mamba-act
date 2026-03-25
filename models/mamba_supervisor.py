"""
Mamba Supervisor: the core research contribution.

A Mamba SSM-based module that runs at motor frequency (O(1) per step) to
continuously monitor whether the current world state has deviated from the
VLA's expected world model, triggering re-queries when needed.

Architecture:
    (visual_features, robot_state, vla_task_embedding, vla_visual_context)
    -> 4 separate linear projections -> concat -> fusion projection
    -> Mamba SSM layers (recurrent, O(1) per step)
    -> deviation score (cosine / L2 / learned MLP)
    -> decision classification (continue / soft_intervene / hard_intervene)

Requires: pip install mamba-ssm  (CUDA + triton needed for compilation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from mamba_ssm import Mamba as MambaLayer  # type: ignore[import-untyped]

from models import SupervisorDecision, VLAOutput


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MambaSupervisorConfig:
    """All hyperparameters for the Mamba supervisor."""

    # ---- Input dimensions ----
    # VLA task embedding dim (must match VLAInterface.get_embedding_dim())
    vla_embedding_dim: int = 512
    # Visual feature dim (from ACT encoder hook, mean-pooled)
    visual_feature_dim: int = 512
    # Robot proprioceptive state dim
    robot_state_dim: int = 14

    # ---- Internal architecture ----
    # All inputs projected to this dim before Mamba
    input_proj_dim: int = 256
    # Mamba SSM hidden dimension (d_model in mamba-ssm)
    mamba_d_model: int = 256
    # Number of stacked Mamba layers
    mamba_n_layers: int = 4
    # SSM state dimension (d_state in mamba-ssm)
    mamba_d_state: int = 16
    # Inner dimension expansion factor
    mamba_expand: int = 2
    # Convolutional kernel width
    mamba_d_conv: int = 4

    # ---- Deviation detection ----
    # "cosine": 1 - cosine_sim(visual_features, vla_visual_context)  [no training needed]
    # "l2":     normalized L2 distance
    # "learned_mlp": scalar output of a learned MLP head on the SSM hidden state
    deviation_metric: Literal["cosine", "l2", "learned_mlp"] = "cosine"
    soft_threshold: float = 0.15   # score >= this triggers soft intervention
    hard_threshold: float = 0.40   # score >= this triggers hard intervention (VLA re-query)

    # ---- Intervention policy ----
    # Minimum timesteps between VLA re-queries (prevents query spam)
    cooldown_timesteps: int = 20
    tiered_intervention: bool = True

    # ---- Training ----
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# MambaSupervisor
# ---------------------------------------------------------------------------


class MambaSupervisor(nn.Module):
    """
    Mamba SSM-based supervisor for continuous semantic monitoring.

    Two code paths:
      step()    — O(1) recurrent inference, called at motor frequency
      forward() — full-sequence parallel processing for supervised training

    The recurrent state is the key to temporal reasoning: a single deviant frame
    may be noise, but a persistent trend over many timesteps is a real problem.
    Mamba's linear-time recurrence captures this for free.
    """

    def __init__(self, config: MambaSupervisorConfig):
        super().__init__()
        self.config = config

        # ---- Input projection layers ----
        self.visual_proj = nn.Linear(config.visual_feature_dim, config.input_proj_dim)
        self.state_proj = nn.Linear(config.robot_state_dim, config.input_proj_dim)
        self.task_embed_proj = nn.Linear(config.vla_embedding_dim, config.input_proj_dim)
        self.visual_ctx_proj = nn.Linear(config.visual_feature_dim, config.input_proj_dim)

        # Concat of 4 streams -> d_model
        self.fusion_proj = nn.Linear(4 * config.input_proj_dim, config.mamba_d_model)

        # ---- Mamba SSM backbone ----
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=config.mamba_d_model,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            )
            for _ in range(config.mamba_n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.mamba_d_model)
            for _ in range(config.mamba_n_layers)
        ])

        # ---- Output heads ----
        if config.deviation_metric == "learned_mlp":
            self.deviation_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(config.mamba_d_model, 64),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.deviation_head = None  # analytic metric, no learned params

        # 3-class: 0=continue, 1=soft_intervene, 2=hard_intervene
        self.decision_head = nn.Sequential(
            nn.Linear(config.mamba_d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 3),
        )

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def get_initial_hidden_state(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Return zero-initialized recurrent states for the start of an episode.

        Mamba1 needs:
          - ssm_state:  (B, d_inner, d_state)  where d_inner = d_model * expand
          - conv_state: (B, d_inner, d_conv - 1)
        """
        device = next(self.parameters()).device
        d_inner = self.config.mamba_d_model * self.config.mamba_expand
        hidden: dict[str, torch.Tensor] = {}
        for i in range(self.config.mamba_n_layers):
            hidden[f"layer_{i}_ssm"] = torch.zeros(
                batch_size, d_inner, self.config.mamba_d_state, device=device
            )
            hidden[f"layer_{i}_conv"] = torch.zeros(
                batch_size, d_inner, self.config.mamba_d_conv - 1, device=device
            )
        return hidden

    def reset(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Return fresh hidden state. Call at episode start."""
        return self.get_initial_hidden_state(batch_size)

    # ------------------------------------------------------------------
    # Input fusion
    # ------------------------------------------------------------------

    def _fuse_inputs(
        self,
        visual_features: torch.Tensor,
        robot_state: torch.Tensor,
        vla_output: VLAOutput,
    ) -> torch.Tensor:
        """Project all input streams and fuse into a single d_model vector.

        Returns: (B, mamba_d_model)
        """
        device = visual_features.device
        B = visual_features.shape[0]

        v = F.relu(self.visual_proj(visual_features))
        s = F.relu(self.state_proj(robot_state))

        task_emb = vla_output.task_embedding.to(device)
        vis_ctx = vla_output.visual_context.to(device)
        if task_emb.dim() == 1:
            task_emb = task_emb.unsqueeze(0).expand(B, -1)
        if vis_ctx.dim() == 1:
            vis_ctx = vis_ctx.unsqueeze(0).expand(B, -1)

        t = F.relu(self.task_embed_proj(task_emb))
        c = F.relu(self.visual_ctx_proj(vis_ctx))

        return F.relu(self.fusion_proj(torch.cat([v, s, t, c], dim=-1)))

    # ------------------------------------------------------------------
    # Deviation scoring
    # ------------------------------------------------------------------

    def _compute_deviation(
        self,
        hidden: torch.Tensor,
        visual_features: torch.Tensor,
        vla_output: VLAOutput,
    ) -> float:
        """Scalar deviation score in [0, ∞).

        cosine:      1 - cosine_sim(current visual, VLA reference visual)
        l2:          normalized L2 distance
        learned_mlp: MLP output on SSM hidden state
        """
        metric = self.config.deviation_metric
        device = visual_features.device

        if metric == "cosine":
            ref = vla_output.visual_context.to(device)
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            sim = F.cosine_similarity(visual_features.float(), ref.float(), dim=-1)
            return float((1.0 - sim).mean().clamp(0.0, 2.0))

        elif metric == "l2":
            ref = vla_output.visual_context.to(device)
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            diff = visual_features.float() - ref.float()
            return float(diff.norm(dim=-1).mean() / (visual_features.shape[-1] ** 0.5))

        else:  # "learned_mlp"
            assert self.deviation_head is not None
            return float(self.deviation_head(hidden).mean())

    # ------------------------------------------------------------------
    # Single-step recurrent inference — the motor-frequency path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        visual_features: torch.Tensor,
        robot_state: torch.Tensor,
        vla_output: VLAOutput,
        hidden_state: dict[str, torch.Tensor],
        timestep: int,
        last_vla_query_timestep: int,
    ) -> SupervisorDecision:
        """
        O(1) single-step inference. Called every motor timestep.

        Updates hidden_state in-place and returns a SupervisorDecision.

        Args:
            visual_features:         (B, visual_feature_dim) — current obs features
            robot_state:             (B, robot_state_dim)
            vla_output:              most recent VLAOutput (unchanged between re-queries)
            hidden_state:            carry-over from previous step (mutated in-place)
            timestep:                current episode timestep
            last_vla_query_timestep: episode timestep of the last VLA query
        """
        # 1. Project and fuse all inputs -> (B, d_model)
        x = self._fuse_inputs(visual_features, robot_state, vla_output)

        # 2. Single-step recurrent pass through each Mamba layer
        #    Mamba1.step(hidden_states, conv_state, ssm_state) -> (out, ssm_state)
        for i, (mamba_layer, norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            residual = x
            x_norm = norm(x)
            out, new_ssm = mamba_layer.step(  # type: ignore[union-attr]
                x_norm,
                hidden_state[f"layer_{i}_conv"],
                hidden_state[f"layer_{i}_ssm"],
            )
            hidden_state[f"layer_{i}_ssm"] = new_ssm
            x = out + residual

        # 3. Deviation score
        deviation_score = self._compute_deviation(x, visual_features, vla_output)

        # 4. Decision classification
        decision_logits = self.decision_head(x)   # (B, 3)
        probs = torch.softmax(decision_logits, dim=-1)
        decision_class = int(probs.argmax(dim=-1)[0])

        # 5. Blend threshold logic with the learned head
        #    Thresholds act as a floor — if the model says "continue" but deviation
        #    is above a threshold, we upgrade. This ensures the supervisor works at
        #    zero-shot before any training data is collected.
        if decision_class == 0:
            if deviation_score >= self.config.hard_threshold:
                decision_class = 2
            elif deviation_score >= self.config.soft_threshold:
                decision_class = 1

        # 6. Cooldown: hard_intervene -> soft_intervene within cooldown window
        within_cooldown = (timestep - last_vla_query_timestep) < self.config.cooldown_timesteps
        if within_cooldown and decision_class == 2:
            decision_class = 1

        # 7. Build result
        _LABELS = ["continue", "soft_intervene", "hard_intervene"]
        _SPEED = [1.0, 0.5, 0.0]
        action = cast(Literal["continue", "soft_intervene", "hard_intervene"], _LABELS[decision_class])

        return SupervisorDecision(
            action=action,
            deviation_score=deviation_score,
            confidence=float(probs.max()),
            execution_speed_factor=_SPEED[decision_class],
            trigger_requery=(action == "hard_intervene"),
            hidden_state=hidden_state,
        )

    # ------------------------------------------------------------------
    # Full-sequence training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Batch training forward pass over a full episode sequence.

        Args:
            sequence: (B, T, 4 * input_proj_dim) — use prepare_sequence() to build this

        Returns:
            deviation_scores: (B, T, 1) if deviation_metric="learned_mlp", else None
            decision_logits:  (B, T, 3)
        """
        x = F.relu(self.fusion_proj(sequence))  # (B, T, d_model)

        for mamba_layer, norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = norm(x)
            x = mamba_layer(x) + residual

        dev_scores = self.deviation_head(x) if self.deviation_head is not None else None
        decision_logits = self.decision_head(x)
        return dev_scores, decision_logits

    def prepare_sequence(
        self,
        visual_features: torch.Tensor,
        robot_states: torch.Tensor,
        task_embeddings: torch.Tensor,
        visual_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare fused input tensor for the training forward() pass.

        Args:
            visual_features:  (B, T, visual_feature_dim)
            robot_states:     (B, T, robot_state_dim)
            task_embeddings:  (B, T, vla_embedding_dim)
            visual_contexts:  (B, T, visual_feature_dim)

        Returns: (B, T, 4 * input_proj_dim)
        """
        v = F.relu(self.visual_proj(visual_features))
        s = F.relu(self.state_proj(robot_states))
        t = F.relu(self.task_embed_proj(task_embeddings))
        c = F.relu(self.visual_ctx_proj(visual_contexts))
        return torch.cat([v, s, t, c], dim=-1)
