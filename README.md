# Autonomous Robotic Arm Manipulation via Sparse-Reward Reinforcement Learning & Hindsight Experience Replay

> **Final Year Research Project** | Dept. of CS (AI & ML)  
> **Author:** Abhi Virani  
> **Live Demo & Telemetry Dashboard:** [https://abhiguru25.github.io/robotic-arm/](https://abhiguru25.github.io/robotic-arm/)

---

## 🎯 Executive Summary

This research conducts a rigorous empirical comparative evaluation of **Soft Actor-Critic (SAC)**, **Twin Delayed DDPG (TD3)**, **Deep Deterministic Policy Gradient (DDPG)**, and **Proximal Policy Optimization (PPO)** augmented with **Hindsight Experience Replay (HER)** on a 7-DOF Franka Emika Panda arm executing sparse binary reward manipulation tasks (`PandaPickAndPlace-v3`).

Our proposed **SAC + HER** model achieves a **78.0% success rate** over 1,000,000 steps, outperforming baseline model-free algorithms which fail under severe reward sparsity.

---

## 📊 Key Benchmark Results (1,000,000 Steps)

| Model | Policy Type | HER Strategy | Final Success Rate | Steps to 50% Success | Mean Reward |
|:---|:---|:---|:---:|:---:|:---:|
| **SAC + HER (Proposed)** | Stochastic, Off-Policy | `future` (k=4) | **78.0%** | **~480,000** | **-12.8** |
| **TD3 + HER** | Deterministic, Off-Policy | `future` (k=4) | **72.0%** | ~550,000 | -14.2 |
| **SAC Baseline** | Stochastic, Off-Policy | None | 15.0% | N/A | -45.1 |
| **TD3 Baseline** | Deterministic, Off-Policy | None | 12.0% | N/A | -46.8 |
| **DDPG Baseline** | Deterministic, Off-Policy | None | 8.0% | N/A | -48.2 |

---

## 💡 Key Contributions

1. **Custom First-Principles Implementations:** Native PyTorch implementations of TD3 and SAC optimized for goal-conditioned environments.
2. **Goal-Conditioned Hindsight Experience Replay:** Re-labeling unachieved trajectory goals using the `future` sampling strategy ($k=4$), transforming 100% of failed exploration steps into dense learning signals.
3. **4-Stage Operational Space Controller:** Operational controller ensuring smooth approach, vertical descent, binary clamping, and transport without block teleportation or joint singularity.
4. **Interactive Telemetry Web Dashboard:** Standalone 3D web interface featuring real-time state visualization and telemetry monitoring.

---

## 🚀 Quick Usage

### Installation
```bash
pip install -r requirements.txt
```

### Evaluation & Inference
To evaluate the trained SAC + HER model checkpoint:
```bash
python scripts/evaluate.py --algo sac --her --episodes 100
```

### Video Recording
To generate a 1080p telemetry HUD video:
```bash
python scripts/record_linkedin_video.py --algo sac --her
```

---

## 📜 Citation

```bibtex
@article{virani2026comparative,
  title   = {Comparative Study of DDPG, TD3, SAC, and PPO for Sparse-Reward Robotic Arm Manipulation using Hindsight Experience Replay},
  author  = {Virani, Abhi},
  journal = {Department of Computer Science (AI & ML)},
  year    = {2026}
}
```
