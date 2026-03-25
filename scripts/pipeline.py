"""
VLA-Mamba-ACT Pipeline: the main execution orchestrator.

Ties together:
  - VLAInterface   — slow, expensive semantic planning
  - ACTDecoder     — action chunk prediction and execution
  - MambaSupervisor — fast, continuous semantic monitoring

The supervisor is the key addition: it runs every motor step (O(1)) and can
trigger early VLA re-queries before the action chunk naturally expires.

Usage:
    from models.vla_interface import MockVLAInterface
    from models.act_decoder import ACTDecoder, ACTDecoderConfig
    from models.mamba_supervisor import MambaSupervisor, MambaSupervisorConfig
    from scripts.pipeline import VLAMambaACTPipeline, PipelineConfig

    vla = MockVLAInterface()
    act = ACTDecoder(ACTDecoderConfig(...))
    supervisor = MambaSupervisor(MambaSupervisorConfig(...))

    pipeline = VLAMambaACTPipeline(vla, act, supervisor, PipelineConfig())
    metrics = pipeline.run_episode(env, task_instruction="Pick up the cube.")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

try:
    import wandb  # type: ignore[import-untyped]
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from models import ExecutionState, ObservationBundle, SupervisorDecision
from models.act_decoder import ACTDecoder
from models.mamba_supervisor import MambaSupervisor
from models.vla_interface import VLAInterface


@dataclass
class PipelineConfig:
    """Configuration for the VLAMambaACTPipeline."""

    # When False: reverts to a plain VLA+ACT baseline (no supervisor)
    supervisor_enabled: bool = True
    # Baseline mode only: how often to query the VLA (every N motor steps)
    vla_query_interval: int = 100
    # Number of actions to execute from a chunk before re-predicting
    # Must match ACTDecoderConfig.n_action_steps (typically 1 with temporal ensembling)
    n_action_steps: int = 1
    # W&B logging
    wandb_project: str = "vla-mamba-act"
    wandb_run_name: Optional[str] = None
    log_deviation: bool = True
    # Log raw visual features per step (expensive, off by default)
    log_visual_features: bool = False
    device: str = "cuda"


class VLAMambaACTPipeline:
    """
    Orchestrates the VLA -> Mamba Supervisor -> ACT Decoder pipeline.

    Motor-frequency execution loop (called once per environment step):

    1. Get observation from environment
    2. Extract visual features (free: ACT encoder hook)
    3. Run Mamba supervisor step (O(1) SSM recurrence)
    4. If supervisor triggers re-query (and cooldown is clear):
         a. Query VLA (expensive)
         b. Get new action chunk from ACT
         c. Reset chunk position to 0
    5. If chunk is exhausted naturally: get new chunk (baseline behavior)
    6. Get current action from chunk at chunk_position
    7. Advance chunk_position
    8. Execute action, log metrics

    Baseline mode (supervisor_enabled=False):
        VLA is queried every vla_query_interval steps unconditionally.
        Supervisor is bypassed entirely.
        Use this as the comparison baseline for evaluation.
    """

    def __init__(
        self,
        vla: VLAInterface,
        act: ACTDecoder,
        supervisor: MambaSupervisor,
        config: PipelineConfig,
    ):
        self.vla = vla
        self.act = act
        self.supervisor = supervisor
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self._wandb_run = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, task_instruction: str, batch_size: int = 1) -> ExecutionState:
        """Reset all components and return a fresh ExecutionState.

        Call at the start of every episode.
        """
        self.vla.reset()
        self.act.reset()
        mamba_hidden = self.supervisor.reset(batch_size)

        return ExecutionState(
            current_vla_output=None,
            current_action_chunk=None,
            chunk_position=0,
            last_vla_query_timestep=-self.config.vla_query_interval,  # allows immediate query
            supervisor_active=self.config.supervisor_enabled,
            mamba_ssm_state=mamba_hidden,
            deviation_history=[],
        )

    # ------------------------------------------------------------------
    # Single motor step
    # ------------------------------------------------------------------

    def step(
        self,
        obs: ObservationBundle,
        state: ExecutionState,
    ) -> tuple[torch.Tensor, ExecutionState, Optional[SupervisorDecision]]:
        """
        Single motor-frequency step.

        Returns:
            action:   (action_dim,) tensor ready to send to the robot
            state:    updated ExecutionState
            decision: SupervisorDecision if supervisor is active, else None
        """
        decision: Optional[SupervisorDecision] = None
        should_requery = False

        if state.supervisor_active:
            # --- Supervised path ---
            # Need an initial VLA output to monitor against
            if state.current_vla_output is None:
                state.current_vla_output = self.vla.query(obs)
                state.last_vla_query_timestep = obs.timestep

            # Get visual features (free: uses cached ACT encoder hook output)
            # On first step, features may be None — predict a chunk to populate
            visual_features = self.act.get_encoder_features()
            if visual_features is None:
                state.current_action_chunk = self.act.predict_chunk(obs)
                state.chunk_position = 0
                visual_features = self.act.get_encoder_features()

            # Fallback: zero features if hook still not populated
            if visual_features is None:
                visual_features = torch.zeros(
                    1, self.act.config.image_shape[0],  # rough fallback dim
                    device=self.device,
                )

            robot_state = obs.robot_state.to(self.device)
            if robot_state.dim() == 1:
                robot_state = robot_state.unsqueeze(0)

            # Run Mamba supervisor step (O(1))
            assert state.mamba_ssm_state is not None
            decision = self.supervisor.step(
                visual_features=visual_features,
                robot_state=robot_state,
                vla_output=state.current_vla_output,
                hidden_state=state.mamba_ssm_state,
                timestep=obs.timestep,
                last_vla_query_timestep=state.last_vla_query_timestep,
            )
            state.mamba_ssm_state = decision.hidden_state
            state.deviation_history.append(decision.deviation_score)

            should_requery = decision.trigger_requery

        else:
            # --- Baseline path: fixed-interval VLA queries ---
            steps_since_query = obs.timestep - state.last_vla_query_timestep
            should_requery = (steps_since_query >= self.config.vla_query_interval)

            if state.current_vla_output is None:
                should_requery = True

        # --- VLA re-query (supervisor-triggered or baseline interval) ---
        if should_requery:
            state.current_vla_output = self.vla.query(obs)
            state.last_vla_query_timestep = obs.timestep
            state.current_action_chunk = self.act.predict_chunk(obs)
            state.chunk_position = 0

        # --- Natural chunk boundary: predict new chunk ---
        elif (
            state.current_action_chunk is None
            or state.chunk_position >= self.config.n_action_steps
        ):
            state.current_action_chunk = self.act.predict_chunk(obs)
            state.chunk_position = 0

        # --- Get action from current chunk position ---
        action = self.act.get_action_at(state.current_action_chunk, state.chunk_position)
        state.chunk_position += 1

        # Squeeze batch dim for single-env execution
        action = action.squeeze(0)

        return action, state, decision

    # ------------------------------------------------------------------
    # Full episode loop
    # ------------------------------------------------------------------

    def run_episode(
        self,
        env,
        task_instruction: str,
        max_steps: int = 500,
        log_to_wandb: bool = False,
    ) -> dict:
        """
        Run a complete episode.

        Args:
            env:              gymnasium-compatible environment
            task_instruction: natural language task description
            max_steps:        safety cap on episode length
            log_to_wandb:     whether to log per-step metrics to W&B

        Returns: episode metrics dict with keys:
            - total_steps
            - total_reward
            - vla_query_count
            - success (if env provides it)
            - deviation_history
            - episode_time_s
        """
        if log_to_wandb and _WANDB_AVAILABLE and self._wandb_run is None:
            self._wandb_run = wandb.init(  # type: ignore[possibly-undefined]
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "supervisor_enabled": self.config.supervisor_enabled,
                    "vla": self.vla.name,
                    "task": task_instruction,
                },
            )

        state = self.reset(task_instruction)
        raw_obs, _ = env.reset()
        obs = self._wrap_obs(raw_obs, task_instruction, timestep=0)

        total_reward = 0.0
        vla_query_count = 0
        success = False
        t_start = time.time()
        t = 0

        for t in range(max_steps):
            obs.timestep = t
            prev_query_ts = state.last_vla_query_timestep

            action, state, decision = self.step(obs, state)

            if state.last_vla_query_timestep != prev_query_ts:
                vla_query_count += 1

            raw_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            total_reward += float(reward)

            obs = self._wrap_obs(raw_obs, task_instruction, timestep=t + 1)

            if log_to_wandb and _WANDB_AVAILABLE and self._wandb_run is not None:
                log_dict = {
                    "step": t,
                    "reward": reward,
                    "vla_query_count": vla_query_count,
                }
                if decision is not None and self.config.log_deviation:
                    log_dict["deviation_score"] = decision.deviation_score
                    log_dict["supervisor_action"] = decision.action
                    log_dict["supervisor_confidence"] = decision.confidence
                wandb.log(log_dict)  # type: ignore[possibly-undefined]

            if terminated or truncated:
                success = info.get("success", False)
                break

        return {
            "total_steps": t + 1,
            "total_reward": total_reward,
            "vla_query_count": vla_query_count,
            "success": success,
            "deviation_history": state.deviation_history,
            "episode_time_s": time.time() - t_start,
        }

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------

    def _wrap_obs(self, raw_obs, task_instruction: str, timestep: int) -> ObservationBundle:
        """
        Convert a raw gymnasium observation dict to an ObservationBundle.

        Expects raw_obs to have:
          - "pixels" or "image": np.ndarray (H, W, C) or dict of arrays
          - "state": np.ndarray (state_dim,)

        Adapt this method to match your specific environment's obs format.
        """
        import numpy as np

        images = []
        # Handle dict observations with image keys
        if isinstance(raw_obs, dict):
            for key, val in raw_obs.items():
                if "image" in key or "pixel" in key or "rgb" in key.lower():
                    if isinstance(val, np.ndarray):
                        img = torch.from_numpy(val).float()
                        if img.dim() == 3 and img.shape[-1] == 3:
                            img = img.permute(2, 0, 1) / 255.0  # (H, W, C) -> (C, H, W)
                        images.append(img.unsqueeze(0).to(self.device))  # (1, C, H, W)

            state_key = next(
                (k for k in raw_obs if "state" in k or "qpos" in k or "joint" in k), None
            )
            if state_key is not None:
                robot_state = torch.from_numpy(
                    np.array(raw_obs[state_key], dtype=np.float32)
                ).unsqueeze(0).to(self.device)
            else:
                robot_state = torch.zeros(1, self.act.config.robot_state_dim, device=self.device)

        else:
            # Flat array observation (state-only envs)
            if isinstance(raw_obs, np.ndarray):
                robot_state = torch.from_numpy(raw_obs.astype(np.float32)).unsqueeze(0).to(self.device)
            else:
                robot_state = torch.zeros(1, self.act.config.robot_state_dim, device=self.device)

        if not images:
            # No images found: create a dummy (for state-only envs)
            h, w = self.act.config.image_shape[1], self.act.config.image_shape[2]
            images = [torch.zeros(1, 3, h, w, device=self.device)]

        return ObservationBundle(
            images=images,
            robot_state=robot_state,
            task_language=task_instruction,
            timestep=timestep,
        )
