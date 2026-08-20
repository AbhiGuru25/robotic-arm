# Comparative Study of DDPG, TD3, SAC, and PPO for Sparse-Reward Robotic Arm Manipulation using HER

> **Final Year Research Project** — Adani University, Dept. of CS (AI & ML)
> _"Comparative Study of DDPG, TD3, SAC, and PPO for Sparse-Reward Robotic Arm Manipulation
> using Hindsight Experience Replay"_

---

## Project Overview

This project benchmarks four deep reinforcement learning algorithms on robotic arm manipulation
tasks with sparse rewards, using the **Franka Panda arm** in the `panda-gym` simulator:

| Algorithm | Policy Type | Implementation |
|-----------|-------------|----------------|
| DDPG | Deterministic, off-policy | Stable-Baselines3 |
| TD3  | Deterministic, off-policy | **From scratch** |
| SAC  | Stochastic, off-policy   | **From scratch** |
| PPO  | Stochastic, on-policy    | Stable-Baselines3 |

**Research Questions:**
1. How do DDPG, TD3, SAC, and PPO compare in success rate and sample efficiency on sparse-reward manipulation tasks?
2. Does Hindsight Experience Replay (HER) meaningfully close the gap between on-policy and off-policy performance?
3. What is the trade-off in training stability and wall-clock time across these algorithms?

---

## Project Structure

```
robotic-arm-rl/
├── algorithms/
│   ├── scratch/        # TD3, SAC implemented from first principles
│   └── baselines/      # DDPG, PPO via Stable-Baselines3
├── envs/               # Environment wrappers
├── configs/            # YAML hyperparameter configs per algorithm/task
├── scripts/            # train.py, evaluate.py, record_video.py, run_all_experiments.py
├── utils/              # Replay buffer, HER buffer, logger, plotting
├── logs/               # TensorBoard logs (git-ignored)
├── checkpoints/        # Saved model weights (git-ignored)
├── results/            # Final plots, comparison tables, CSVs
├── notebooks/          # Exploratory analysis
├── docs/               # Setup guide, report drafts
├── paper/              # IEEE LaTeX research paper
└── requirements.txt
```

---

## Quick Start (Google Colab — Recommended)

```python
# Cell 1: Clone and install
!git clone <your-repo-url> robotic-arm-rl
%cd robotic-arm-rl
!pip install -r requirements.txt

# Cell 2: Verify environment
!python scripts/verify_setup.py

# Cell 3: Train TD3 on PandaReach (smoke test)
!python scripts/train.py --algo td3 --task reach --seed 0 --steps 50000

# Cell 4: Full training run with HER
!python scripts/train.py --algo td3 --task pickandplace --her --seed 0
```

---

## Local Setup (Windows / Linux / macOS)

```bash
# 1. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python scripts/verify_setup.py

# 4. Run a quick smoke test
python scripts/train.py --algo td3 --task reach --seed 0 --steps 1000
```

---

## Training Commands

### Single algorithm run
```bash
python scripts/train.py --algo td3  --task reach         --seed 0
python scripts/train.py --algo td3  --task pickandplace  --seed 0
python scripts/train.py --algo td3  --task pickandplace  --her --seed 0
python scripts/train.py --algo sac  --task pickandplace  --her --seed 0
python scripts/train.py --algo ddpg --task reach         --seed 0
python scripts/train.py --algo ppo  --task reach         --seed 0
```

### Batch run all experiments (all algo x task x HER x seed)
```bash
python scripts/run_all_experiments.py
```

### Evaluate a trained checkpoint
```bash
python scripts/evaluate.py --algo td3 --task pickandplace --her --seed 0 --episodes 100
```

### Generate comparison plots
```bash
python scripts/plot_results.py
```

---

## Algorithms Explained

### TD3 (Twin Delayed DDPG) — from scratch
Three improvements over DDPG: (1) twin critics to reduce Q-value overestimation,
(2) delayed policy updates for stability, (3) target policy smoothing for robustness.
See `algorithms/scratch/td3.py`.

### SAC (Soft Actor-Critic) — from scratch
Maximum entropy RL framework: optimises both reward and policy entropy simultaneously.
Uses reparameterisation trick for stochastic policy, automatic temperature tuning.
See `algorithms/scratch/sac.py`.

### HER (Hindsight Experience Replay)
Replay augmentation strategy: relabels failed episodes using states actually achieved
as hindsight goals, providing learning signal even from failed trajectories.
Uses the 'future' strategy. See `utils/her_buffer.py`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Success Rate (%) | Fraction of eval episodes that reach the goal |
| Sample Efficiency | Steps to reach 80% success rate |
| Training Stability | Variance across ≥3 random seeds |
| Wall-Clock Time | Actual training time per config on Colab |

---

## Reproducibility

All experiments use fixed random seeds and versioned YAML configs.
To reproduce any result:
```bash
python scripts/train.py --config configs/td3_her_pickandplace.yaml --seed 42
```

---

## Citation

```bibtex
@article{virani2026comparative,
  title   = {Comparative Study of DDPG, TD3, SAC, and PPO for Sparse-Reward
             Robotic Arm Manipulation using Hindsight Experience Replay},
  author  = {Virani, Abhi},
  journal = {Final Year Research Project, Adani University},
  year    = {2026}
}
```

---

## Key References
- Lillicrap et al. (2016) — DDPG, arXiv:1509.02971
- Fujimoto et al. (2018) — TD3, arXiv:1802.09477
- Haarnoja et al. (2018) — SAC, arXiv:1801.01290
- Schulman et al. (2017) — PPO, arXiv:1707.06347
- Andrychowicz et al. (2017) — HER, arXiv:1707.01495
- Gallouedec et al. (2021) — panda-gym, arXiv:2106.13687
- Raffin et al. (2021) — Stable-Baselines3, JMLR 22(268)
