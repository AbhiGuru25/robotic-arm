# 🎓 Final Year Project Defense & Viva Presentation Deck

> **Project Title:** Language-Guided Goal-Conditioned Robotic Manipulation via Soft Actor-Critic (SAC), Hindsight Experience Replay (HER), and Sim-to-Real Domain Randomization  
> **Author:** Abhi Virani  
> **Department:** Department of Computer Science (AI & ML), Adani University  
> **Academic Session:** 2025–2026

---

## 📌 Slide 1: Introduction & Problem Statement

### The Problem in Autonomous Robotics:
- **Continuous 7-DOF Control:** Controlling a 7-degree-of-freedom Franka Emika Panda arm in 3D continuous space ($dx, dy, dz, da$).
- **The Sparse Reward Trap:** Under binary indicator rewards ($r = 0.0$ on completion, $r = -1.0$ otherwise), standard RL exploration algorithm gradient stays zero for millions of random steps.
- **Goal-Conditioned Generalization:** The policy $\pi(a | s, g)$ must generalize across random starting block locations and arbitrary target 3D goal vectors.

---

## 💡 Slide 2: Next-Gen System Architecture

```
[ Human Text Prompt ] ──> [ Language Goal Encoder ] ──> Goal Vector g
                                                             │
[ Gymnasium Physics ] ──> [ Sim-to-Real Randomization ] ──> Observation s
                                                             │
                                                             ▼
                                                    [ SAC + HER Policy π ]
                                                             │
                                                             ▼
                                                [ Intrinsic RND Curiosity ]
```

1. **Language Goal Encoder:** Translates natural language instructions into dense goal embeddings.
2. **Goal-Conditioned HER (Future Strategy $k=4$):** Relabels unachieved trajectory states, turning 100% of failed exploration steps into positive learning signals.
3. **Soft Actor-Critic (SAC) Optimization:** Optimizes expected return + policy entropy $\mathcal{H}(\pi(\cdot|s_t))$.
4. **Sim-to-Real Domain Randomization:** Randomizes table friction ($\mu \in [0.2, 0.8]$), block mass ($m$), and joint noise during training.

---

## 📊 Slide 3: Multi-Task Empirical Benchmark Results (1,000,000 Steps)

| Task | Environment | SAC + HER (Proposed) | TD3 + HER | SAC Baseline | Zero-Shot Transfer Index |
|:---|:---|:---:|:---:|:---:|:---:|
| 🎯 **Reach** | `PandaReach-v3` | **99.5%** | 99.0% | 62.0% | 98.2% |
| 📦 **Pick & Place** | `PandaPickAndPlace-v3` | **78.0%** | 72.0% | 15.0% | **96.5%** |
| 🛷 **Push** | `PandaPush-v3` | **84.5%** | 79.0% | 22.0% | 95.0% |
| 🏒 **Slide** | `PandaSlide-v3` | **68.0%** | 61.0% | 9.0% | 92.4% |

**Key Takeaways:**
- Without HER, model-free baselines collapse under sparse rewards (<15% success).
- SAC + HER outperforms TD3 + HER due to maximum entropy exploration.
- Sim-to-Real domain randomization maintains **>96% zero-shot transfer robustness**.

---

## 🎥 Slide 4: Interactive Telemetry & Video Demonstration

- **Live 3D Telemetry Web Dashboard:** [https://abhiguru25.github.io/robotic-arm/](https://abhiguru25.github.io/robotic-arm/)
- **Demonstrated Reasoning Scenarios:**
  1. **Dynamic Obstacle Avoidance:** Parabolic elevation arc trajectory over barrier wall.
  2. **Mid-Flight Goal Re-Conditioning:** Adapting trajectory when target shifts dynamically mid-air.
  3. **Zero-Shot Sim-to-Real Adaptation:** Increasing clamping force under low-friction surface ($\mu = 0.22$).

---

## 📜 Slide 5: Academic Contributions & Future Scope

### Primary Contributions:
1. First-principles PyTorch implementations of TD3, SAC, and Goal-Conditioned HER.
2. Natural Language Instruction interface mapping prompts to goal vectors.
3. Published IEEE LaTeX paper and live interactive web visualizer.

### Future Research Directions:
- Bimanual dual-arm manipulation.
- Vision-Language-Action (VLA) foundation model integration.
