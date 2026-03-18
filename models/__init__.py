"""
Defines the data classes for the entire pipeline. Every module imports from here, and none import from
each other. This means that you can swap out the VLA (or even the decoder or supervisor) as long as
they adhere to these typs. This ensures that this stack can be tested in a VLA-agnostic fashion. 
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class ObservationBundle:
    """Everything perceived at a timestep, passed to all pipeline components. Device management is handled per-module at construction time"""

    """List of (B, C, H, W) image tensors, one per camera.Following Pytorch convention, B is batch size, 
    C are color channels, H is height (# of pixel rows), and W is width (# of pixel columns). 
    """
    images: list[torch.Tensor]
    # Proprioceptive state, the robot's understanding of itself from internal sensors: (B, state_dim)
    robot_state: torch.Tensor #7 DOF robot arm -> 7 joint angles and velocities vector is (B, 14)
    # Task description as a raw string to be decoded by VLA
    task_language: str
    # Timestep within the current episode
    timestep: int


@dataclass
class VLAOutput:
    #Output of a VLA query.

    # Semantic intent embedding: (embedding_dim,)
    # Encodes "what the VLA thinks it should do"
    task_embedding: torch.Tensor
    # Reference visual state from the VLA encoder: (visual_context_dim,)
    # Encodes "what the VLA saw when it made this decision"
    visual_context: torch.Tensor
    # Raw action tokens before ACT decoding (optional): (B, num_tokens, token_dim)
    raw_action_tokens: torch.Tensor | None
    # Wall-clock timestamp of this query (for cooldown tracking)
    query_timestamp: float = field(default_factory=time.time)
    # Episode timestep at which this query was made
    query_timestep: int = 0
    # Implementation-specific extras
    metadata: dict = field(default_factory=dict)


@dataclass
class SupervisorDecision:
    """What the Mamba supervisor emits at every motor timestep."""

    # Primary decision: what to do
    action: Literal["continue", "soft_intervene", "hard_intervene"]
    # Raw scalar deviation score — for logging and threshold tuning
    deviation_score: float
    # Supervisor confidence in its own assessment: [0, 1]
    confidence: float
    # For soft intervention: execution speed multiplier in (0, 1]
    # 1.0 = normal speed, 0.5 = slow to 50%
    execution_speed_factor: float
    # Whether a VLA re-query should be triggered right now
    trigger_requery: bool
    # Mamba recurrent hidden state carry-over for next timestep
    hidden_state: dict[str, torch.Tensor]


@dataclass
class ExecutionState:
    """Shared mutable state of the pipeline across timesteps."""

    # Most recent VLA output (updated on each re-query)
    current_vla_output: VLAOutput | None = None
    # Current action chunk being executed: (chunk_size, action_dim)
    current_action_chunk: torch.Tensor | None = None
    # Index into the current chunk (0 = fresh chunk just received)
    chunk_position: int = 0
    # Episode timestep of the last VLA re-query (for cooldown enforcement)
    last_vla_query_timestep: int = -999
    # Whether the supervisor is currently active
    supervisor_active: bool = True
    # Mamba SSM carry-over hidden states (None before first episode step)
    mamba_ssm_state: dict[str, torch.Tensor] | None = None
    # Running log of deviation scores for analysis
    deviation_history: list[float] = field(default_factory=list)
