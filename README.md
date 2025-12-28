# **Steve vs Gravity** 

_Experimental RL + Imitation project for "Steve" the Untitree robot—learning to run, jump, do backflips, and dodge obstacles!_

🚧 **Work in Progress** 🚧

![Steve punch](videos/steve_steve_punch_run_02_chk.gif) ![Steve Walk](videos/steve_steve_walk_run_01_chk.gif)

## Overview

Steve vs Gravity trains a **Unitree G1** humanoid robot to perform natural locomotion by imitating motion capture data. The project uses **PPO** with **residual joint position actions** (not absolute positions) to learn robust walking and dynamic movements.

Motion data from the **LAFAN1** dataset is retargeted to the Unitree G1 using the **GMR (General Motion Retargeting)** pipeline.

## Key Features

- DeepMimic-inspired imitation learning with residual actions
- Motion retargeting using GMR from LAFAN1 to Unitree G1
- PPO training with parallel environments in IsaacLab
- Phase-based control for cyclic motion tracking
- Reward shaping for joint tracking, root positioning, and end-effector alignment

## Quick Start

**Training:**
```
python run_steve.py --mode train
```

**Evaluation:**
```
python run_steve.py --mode eval --load
```

## Technical Details

- **Action Space:** Residual joint position deltas (±18°)
- **Observation:** Joint states, velocities, projected gravity, motion phase
- **Rewards:** Joint position/velocity tracking, root tracking, end-effector matching
- **Policy:** MLP [1024, 512, 256] actor-critic
- **Simulation:** 30 Hz control in IsaacLab

## Roadmap

- [ ] Switch to **RSL RL** framework
- [ ] Integrate **AMP (Adversarial Motion Priors)**
- [ ] Add **diffusion-based autoregressive locomotion**

## References

- **DeepMimic:** Motion imitation framework
- **GMR:** General Motion Retargeting pipeline
- **LAFAN1:** Motion capture dataset
- **IsaacLab:** NVIDIA robotics simulation

---

**Robot:** Unitree G1 Humanoid  
**Status:** Active Development 🚧
