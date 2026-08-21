"""
scripts/train.py
================
Unified training entry point for the robotic arm RL project.

Supports all four algorithms (TD3, SAC, DDPG, PPO) with optional HER,
across both panda-gym tasks (reach, pickandplace).

Usage
-----
    # TD3 on PandaReach (no HER)
    python scripts/train.py --algo td3 --task reach --seed 0

    # TD3 + HER on PandaPickAndPlace
    python scripts/train.py --algo td3 --task pickandplace --her --seed 0

    # SAC + HER, custom config
    python scripts/train.py --algo sac --task pickandplace --her --seed 1

    # DDPG baseline (SB3)
    python scripts/train.py --algo ddpg --task reach --seed 0

    # PPO baseline (SB3)
    python scripts/train.py --algo ppo --task reach --seed 0

    # Quick smoke test (1000 steps)
    python scripts/train.py --algo td3 --task reach --seed 0 --steps 1000

Config loading
--------------
If --config is not specified, the script automatically selects the
corresponding YAML file from the configs/ directory, e.g.:
    td3  + reach        + no HER  -> configs/td3_reach.yaml
    td3  + pickandplace + HER     -> configs/td3_her_pickandplace.yaml

All hyperparameters can be overridden with --steps, --lr, etc.
"""

import argparse
import os
import pathlib
import sys

# Add project root to path so relative imports work
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RL agent on robotic arm manipulation task."
    )
    p.add_argument(
        "--algo", type=str, required=True,
        choices=["td3", "sac", "ddpg", "ppo"],
        help="Algorithm to use."
    )
    p.add_argument(
        "--task", type=str, required=True,
        choices=["reach", "pickandplace", "push", "slide"],
        help="panda-gym task to train on."
    )
    p.add_argument(
        "--her", action="store_true", default=False,
        help="Use Hindsight Experience Replay (off-policy algos only)."
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility."
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="Total environment steps (overrides config value if set)."
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file.  Auto-detected if not specified."
    )
    p.add_argument(
        "--log_dir", type=str, default="logs",
        help="TensorBoard log directory."
    )
    p.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Checkpoint save directory."
    )
    p.add_argument(
        "--device", type=str, default=None,
        help='PyTorch device ("cpu" or "cuda"). Auto-detected if not set.'
    )
    p.add_argument(
        "--eval_freq", type=int, default=5_000,
        help="Evaluate every this many steps."
    )
    p.add_argument(
        "--eval_episodes", type=int, default=20,
        help="Number of evaluation episodes per eval."
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    """Load YAML config, with CLI overrides applied on top."""
    if args.config is not None:
        config_path = pathlib.Path(args.config)
    else:
        # Auto-detect config
        her_tag = "_her" if args.her else ""
        config_name = f"{args.algo}{her_tag}_{args.task}.yaml"
        config_path = ROOT / "configs" / config_name

    if not config_path.exists():
        print(f"[train] Warning: config file not found: {config_path}")
        print("[train] Using default hyperparameters.")
        return {}

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[train] Loaded config: {config_path}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device(args) -> str:
    if args.device is not None:
        return args.device
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Seed utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# TD3 training loop (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def train_td3(args, cfg: dict, device: str) -> None:
    """Full training loop for TD3 (from-scratch implementation)."""
    import gymnasium as gym
    import panda_gym  # noqa

    from algorithms.scratch.td3 import TD3
    from utils.replay_buffer import ReplayBuffer
    from utils.her_buffer import HERBuffer
    from utils.logger import Logger
    from envs.wrappers import make_env

    # ── Hyperparameters (config → CLI override) ───────────────────────
    total_steps    = args.steps or cfg.get("total_timesteps", 500_000)
    batch_size     = cfg.get("batch_size",    256)
    buffer_size    = cfg.get("buffer_size",   1_000_000)
    start_steps    = cfg.get("learning_starts", 1000)
    actor_lr       = cfg.get("actor_lr",      3e-4)
    critic_lr      = cfg.get("critic_lr",     3e-4)
    gamma          = cfg.get("gamma",         0.98)
    tau            = cfg.get("tau",            0.005)
    policy_delay   = cfg.get("policy_delay",  2)
    policy_noise   = cfg.get("policy_noise",  0.2)
    noise_clip     = cfg.get("noise_clip",    0.5)
    expl_noise     = cfg.get("exploration_noise", 0.1)
    max_steps      = cfg.get("max_episode_steps", 50)
    hidden_sizes   = tuple(cfg.get("hidden_sizes", [256, 256]))
    her_k          = cfg.get("her_k", 4)

    run_tag = f"td3{'_her' if args.her else ''}_{args.task}_seed{args.seed}"

    # ── Environments ──────────────────────────────────────────────────
    env      = make_env(args.task, flatten=True, max_steps=max_steps)
    eval_env = make_env(args.task, flatten=True, max_steps=max_steps)

    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # HER needs goal dimensions
    ag_dim = env.ag_dim
    dg_dim = env.dg_dim

    # ── Agent ─────────────────────────────────────────────────────────
    agent = TD3(
        obs_dim           = obs_dim,
        act_dim           = act_dim,
        max_action        = max_action,
        device            = device,
        actor_lr          = actor_lr,
        critic_lr         = critic_lr,
        gamma             = gamma,
        tau               = tau,
        policy_noise      = policy_noise,
        noise_clip        = noise_clip,
        policy_delay      = policy_delay,
        exploration_noise = expl_noise,
        hidden_sizes      = hidden_sizes,
    )
    print(f"[train] Agent: {agent}")

    # ── Replay buffer ─────────────────────────────────────────────────
    if args.her:
        from utils.her_buffer import HERBuffer
        buf = HERBuffer(
            obs_dim      = obs_dim,
            act_dim      = act_dim,
            ag_dim       = ag_dim,
            dg_dim       = dg_dim,
            capacity     = buffer_size,
            k            = her_k,
            max_eps_len  = max_steps,
            device       = device,
        )
    else:
        buf = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                           capacity=buffer_size, device=device)

    # ── Logger ────────────────────────────────────────────────────────
    log_path = pathlib.Path(args.log_dir) / run_tag
    logger   = Logger(log_dir=str(log_path), algo="td3",
                      task=args.task, seed=args.seed)

    # ── Checkpoint directory ──────────────────────────────────────────
    ckpt_dir = pathlib.Path(args.checkpoint_dir) / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_success = -1.0

    # ── Main training loop ────────────────────────────────────────────
    obs_dict, _ = env.env.reset(seed=args.seed)  # original dict obs for HER
    obs, _      = env.reset(seed=args.seed)      # flat obs for agent
    episode_step = 0

    print(f"[train] Starting TD3 training: {run_tag}, "
          f"total_steps={total_steps}, device={device}")

    from tqdm import tqdm
    for step in tqdm(range(1, total_steps + 1), desc=run_tag):

        # ── Select action ─────────────────────────────────────────────
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, add_noise=True)

        # ── Step environment ──────────────────────────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)
        done_bool = terminated and not truncated

        # ── Store transition ──────────────────────────────────────────
        if args.her:
            # HER buffer needs original dict obs for goal relabelling
            next_obs_dict, _, _, _, _ = env.env.step.__self__.step(action) if False else (None, None, None, None, None)
            # Simpler: for HER we pass flat obs and store goals separately
            buf.add_step(
                obs       = obs,
                action    = action,
                reward    = reward,
                next_obs  = next_obs,
                done      = done_bool,
                ag        = obs[env.obs_dim : env.obs_dim + env.ag_dim],
                dg        = obs[env.obs_dim + env.ag_dim :],
                next_ag   = next_obs[env.obs_dim : env.obs_dim + env.ag_dim],
            )
        else:
            buf.add(obs, action, reward, next_obs, done_bool)

        obs          = next_obs
        episode_step += 1

        if terminated or truncated:
            if args.her:
                buf.finish_episode()
            obs_raw, _ = env.reset()
            obs        = obs_raw
            episode_step = 0

        # ── Training update ───────────────────────────────────────────
        if step >= start_steps and len(buf) >= batch_size:
            train_info = agent.train(buf, batch_size)
            logger.log_dict(train_info, step)

        # ── Periodic evaluation ───────────────────────────────────────
        if step % args.eval_freq == 0:
            success_rate, mean_reward = evaluate_td3(
                agent, eval_env, n_episodes=args.eval_episodes
            )
            logger.log_scalar("eval/success_rate", success_rate, step)
            logger.log_scalar("eval/mean_reward",  mean_reward,  step)
            print(f"  Step {step:>8d} | success={success_rate:.2%} | "
                  f"reward={mean_reward:.3f}")

            # Save best checkpoint
            if success_rate > best_success:
                best_success = success_rate
                agent.save(str(ckpt_dir / "best.pt"))

            # Save periodic checkpoint
            if step % (args.eval_freq * 10) == 0:
                agent.save(str(ckpt_dir / f"step_{step}.pt"))

    logger.close()
    agent.save(str(ckpt_dir / "final.pt"))
    print(f"[train] Done. Best success rate: {best_success:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_td3(agent, eval_env, n_episodes: int = 20):
    """Run deterministic evaluation episodes, return success_rate + mean_reward."""
    successes = []
    rewards   = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        for _ in range(200):
            action = agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            if terminated or truncated:
                successes.append(float(info.get("is_success", 0.0)))
                break
        rewards.append(ep_reward)
    return float(np.mean(successes)), float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────────────────────
# SB3 training (DDPG, PPO — delegates to baseline wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def train_ddpg(args, cfg: dict) -> None:
    """Train DDPG using SB3 baseline wrapper."""
    from algorithms.baselines.ddpg_baseline import DDPGBaseline
    total_steps = args.steps or cfg.get("total_timesteps", 500_000)
    agent = DDPGBaseline(
        task           = args.task,
        seed           = args.seed,
        her            = args.her,
        log_dir        = args.log_dir,
        checkpoint_dir = args.checkpoint_dir,
    )
    agent.train(total_timesteps=total_steps, eval_freq=args.eval_freq)


def train_ppo(args, cfg: dict) -> None:
    """Train PPO using SB3 baseline wrapper."""
    from algorithms.baselines.ppo_baseline import PPOBaseline
    total_steps = args.steps or cfg.get("total_timesteps", 1_000_000)
    agent = PPOBaseline(
        task           = args.task,
        seed           = args.seed,
        log_dir        = args.log_dir,
        checkpoint_dir = args.checkpoint_dir,
    )
    agent.train(total_timesteps=total_steps, eval_freq=args.eval_freq)


def train_sac(args, cfg: dict, device: str) -> None:
    """Full training loop for SAC (from-scratch implementation)."""
    import gymnasium as gym
    import panda_gym  # noqa

    from algorithms.scratch.sac import SAC
    from utils.replay_buffer import ReplayBuffer
    from utils.her_buffer import HERBuffer
    from utils.logger import Logger
    from envs.wrappers import make_env

    total_steps    = args.steps or cfg.get("total_timesteps", 1_000_000)
    batch_size     = cfg.get("batch_size",    256)
    buffer_size    = cfg.get("buffer_size",   1_000_000)
    start_steps    = cfg.get("learning_starts", 1000)
    actor_lr       = cfg.get("actor_lr",      3e-4)
    critic_lr      = cfg.get("critic_lr",     3e-4)
    alpha_lr       = cfg.get("alpha_lr",      3e-4)
    gamma          = cfg.get("gamma",         0.98)
    tau            = cfg.get("tau",            0.005)
    alpha_init     = cfg.get("alpha_init",     0.2)
    auto_alpha     = cfg.get("auto_entropy_tuning", True)
    max_steps      = cfg.get("max_episode_steps", 50)
    hidden_sizes   = tuple(cfg.get("hidden_sizes", [256, 256]))
    her_k          = cfg.get("her_k", 4)

    run_tag = f"sac{'_her' if args.her else ''}_{args.task}_seed{args.seed}"

    env      = make_env(args.task, flatten=True, max_steps=max_steps)
    eval_env = make_env(args.task, flatten=True, max_steps=max_steps)

    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    ag_dim = env.ag_dim
    dg_dim = env.dg_dim

    agent = SAC(
        obs_dim             = obs_dim,
        act_dim             = act_dim,
        max_action          = max_action,
        device              = device,
        actor_lr            = actor_lr,
        critic_lr           = critic_lr,
        alpha_lr            = alpha_lr,
        gamma               = gamma,
        tau                 = tau,
        alpha_init          = alpha_init,
        auto_entropy_tuning = auto_alpha,
        hidden_sizes        = hidden_sizes,
    )
    print(f"[train] Agent: {agent}")

    if args.her:
        buf = HERBuffer(
            obs_dim      = obs_dim,
            act_dim      = act_dim,
            ag_dim       = ag_dim,
            dg_dim       = dg_dim,
            capacity     = buffer_size,
            k            = her_k,
            max_eps_len  = max_steps,
            device       = device,
        )
    else:
        buf = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                           capacity=buffer_size, device=device)

    log_path = pathlib.Path(args.log_dir) / run_tag
    logger   = Logger(log_dir=str(log_path), algo="sac",
                      task=args.task, seed=args.seed)

    ckpt_dir = pathlib.Path(args.checkpoint_dir) / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_success = -1.0

    obs, _ = env.reset(seed=args.seed)
    episode_step = 0

    print(f"[train] Starting SAC training: {run_tag}, "
          f"total_steps={total_steps}, device={device}")

    from tqdm import tqdm
    for step in tqdm(range(1, total_steps + 1), desc=run_tag):

        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done_bool = terminated and not truncated

        if args.her:
            buf.add_step(
                obs       = obs,
                action    = action,
                reward    = reward,
                next_obs  = next_obs,
                done      = done_bool,
                ag        = obs[env.obs_dim : env.obs_dim + env.ag_dim],
                dg        = obs[env.obs_dim + env.ag_dim :],
                next_ag   = next_obs[env.obs_dim : env.obs_dim + env.ag_dim],
            )
        else:
            buf.add(obs, action, reward, next_obs, done_bool)

        obs          = next_obs
        episode_step += 1

        if terminated or truncated:
            if args.her:
                buf.finish_episode()
            obs, _       = env.reset()
            episode_step = 0

        if step >= start_steps and len(buf) >= batch_size:
            train_info = agent.train(buf, batch_size)
            logger.log_dict(train_info, step)

        if step % args.eval_freq == 0:
            success_rate, mean_reward = evaluate_td3(
                agent, eval_env, n_episodes=args.eval_episodes
            )
            logger.log_scalar("eval/success_rate", success_rate, step)
            logger.log_scalar("eval/mean_reward",  mean_reward,  step)
            print(f"  Step {step:>8d} | success={success_rate:.2%} | "
                  f"reward={mean_reward:.3f}")

            if success_rate > best_success:
                best_success = success_rate
                agent.save(str(ckpt_dir / "best.pt"))

            if step % (args.eval_freq * 10) == 0:
                agent.save(str(ckpt_dir / f"step_{step}.pt"))

    logger.close()
    agent.save(str(ckpt_dir / "final.pt"))
    print(f"[train] Done. Best success rate: {best_success:.2%}")



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args)
    device = get_device(args)

    set_seeds(args.seed)

    print(f"[train] algo={args.algo}  task={args.task}  "
          f"her={args.her}  seed={args.seed}  device={device}")

    if args.algo == "td3":
        train_td3(args, cfg, device)
    elif args.algo == "ddpg":
        train_ddpg(args, cfg)
    elif args.algo == "ppo":
        train_ppo(args, cfg)
    elif args.algo == "sac":
        train_sac(args, cfg, device)


if __name__ == "__main__":
    main()
