# vla-mamba-act
A Mamba-based supervisor for Vision-Language-Action architectures with ACT as the action decoder. 

## Overview
This is an independent research project that investigates the usage of a Mamba-based supervisor to guide Vision-Language-Action (VLA) models, with ACT (Action Chunking with Transformers) as the action decoder. The goal is to leverage Mamba's efficient state-space modeling for supervision of a physical system that uses ACT for dexterous manipulation tasks. The Mamba-supervisor is designed to be VLA-agnostic, hopefully allowing for its continued use as new models are inevitably produced. 

## Tools
* NVIDIA Isaac Lab: all simulation, testing, and evaluation is conducted in Isaac Lab, 
leveraging GPU-accelerated physics for realistic manipulation environments and compatibility 
with robot learning frameworks.
* Hugging Face: model weights, datasets, and pipeline components are sourced and managed 
through Hugging Face Hub, including LeRobot for ACT and target VLA checkpoints.

## The Core Problem
ACT is an action-decoder that produces action chunks, or seqeuences of actions. The strength of ACT lies in the ability to bridge the gap between slow VLA inferences and quick motor control steps. A VLA inference takes far too long for every inference to be connected to every motor control command. Due to the VLA's inability to run at motor frequency, the action sequence is often running open-loop, which means it is not taking in any external information during the action seqeunce. The system is essentially blind before recalculating at the end of the action chunk. While ACT allows for smoother movement in comparison to jittery movement caused by only moving when the VLA is able to compute an inference, the responsiveness of the system is reduced. 

ACT also supports temporal ensembling, where a new action chunk is computed at every timestep and 
overlapping predictions are blended with exponential weighting. This largely addresses the 
motor-level blind execution problem — the robot is continuously recomputing and smoothing its 
planned movements.

However, temporal ensembling does not address the outdated VLA inference. Each new chunk ACT computes is conditioned on the most recent VLA task embedding, and that embedding can only update as fast as the VLA can run. Between VLA queries, ACT may be continuously recomputing smoot action chunks that are all built on an outdated understanding of the world. If the semantic situation changes — an object is displaced, a human enters the workspace, or a subtask completes unexpectedly — ACT has no mechanism to detect this. It will continue executing confidently on a stale intent signal.

During this semantic supervision gap, several failure modes can arise:
* Unexpected collisions if an object shifts or a person enters the workspace mid-execution
* Continued execution toward a goal that has already been achieved or is no longer valid
* Compounding kinematic error if early actions diverge from expected outcomes without triggering replanning

## The Mamba Supervisor
Mamba is a state space model with linear scaling in sequence length and constant compute per timestep during recurrent inference. It can run continuously at or near motor control frequency while the VLA cannot. This allows for the development of a supervisor that monitors the system during an action chunk execution, and can thus truncate an action sequence and recall the VLA if needed. 

## Motivation
VLA models are powerful but expensive. Deploying them on edge physical AI devices — robot arms, mobile manipulators, legged robots with onboard compute — introduces a fundamental tension between capability and safety. Large transformer-based VLAs are not designed to run at motor control frequency, and running them end-to-end without supervision creates unpredictable behavior on physical hardware where failures have real consequences. This architecture is motivated by the need to harness the power of frontier VLA models without the deployment cost or safety unpredictability of running them as the sole control signal.

## Limitations and Open Problems
The Mamba supervisor monitors semantic validity continuously and can trigger early VLA 
re-queries when it detects deviation between observed world state and the active task 
embedding. In highly dynamic environments, this creates a potential failure mode: if the 
supervisor is too sensitive, it may over-call the VLA, negating the computational efficiency 
that motivates the architecture in the first place.

This is an open design question. Proposed mitigations include:

* Learned deviation thresholds — rather than a fixed threshold for triggering re-query, 
  train the supervisor to learn when deviation is actually meaningful versus 
  expected variance during normal execution
* Cooldown windows — enforce a minimum interval between VLA re-queries to bound 
  worst-case inference load
* Confidence-gated re-query — only trigger a re-query when Mamba's uncertainty about 
  the current state exceeds a threshold, not just when the state changes
* Tiered intervention — distinguish between soft intervention (modulate ACT conditioning) 
  and hard intervention (full VLA re-query), reserving the expensive re-query for only the 
  most severe deviations

Characterizing supervisor sensitivity across environments of varying dynamism is a key 
evaluation target for this project.

## References and Related Works
- [ACT — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Mamba — Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [OpenVLA — An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Octo — An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [π0 — A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [RoboMamba — Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation](https://arxiv.org/abs/2406.04339)
- [DuoCore-FS — Asynchronous Fast-Slow Vision-Language-Action Policies](https://arxiv.org/abs/2512.20188)
- [AutoHorizon — Dynamic Execution Horizon Estimation for Action Chunking](https://arxiv.org/abs/2602.21445)
- [Mamba Policy — Towards Efficient 3D Diffusion Policy with Hybrid Selective State Models](https://arxiv.org/abs/2409.07163)
- [RT-2 — Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [Inner Monologue — Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608)
- [Hi Robot — Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](https://arxiv.org/abs/2502.11170)
- [Awesome VLA Papers — Living Paper List](https://github.com/Psi-Robot/Awesome-VLA-Papers)
- [LeRobot — HuggingFace Robotics Framework](https://github.com/huggingface/lerobot)
- [OpenVLA — Official Codebase](https://github.com/openvla/openvla)
- [Mamba — Official Implementation](https://github.com/state-spaces/mamba)

## Project Status
Early-stage independent research. Active development. 

### Architecture
- [ ] Define Mamba supervisor input/output interface
- [ ] Design deviation detection mechanism
- [ ] Design intervention logic (soft vs hard re-query)
- [ ] Implement Mamba supervisor module
- [ ] Implement VLA-agnostic embedding interface

### Integration
- [ ] OpenVLA integration — primary development target
- [ ] ACT integration and temporal ensembling compatibility
- [ ] End-to-end pipeline (OpenVLA → Mamba → ACT)
- [ ] Octo integration — generalization validation
- [ ] π0 integration — stretch goal

### Evaluation
- [ ] Define baseline (VLA + ACT without supervisor)
- [ ] Design evaluation environments of varying dynamism
- [ ] Characterize supervisor sensitivity vs re-query frequency tradeoff
- [ ] Ablation studies (supervisor on/off, threshold sensitivity)
- [ ] Simulation benchmark results
