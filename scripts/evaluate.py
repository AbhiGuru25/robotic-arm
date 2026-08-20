"""
scripts/evaluate.py
====================
Load a trained checkpoint and evaluate it for N episodes.

Usage
-----
    python scripts/evaluate.py --algo td3 --task reach --seed 0
    python scripts/evaluate.py --algo td3 --task pickandplace --her --seed 0 --episodes 100
    python scripts/evaluate.py --algo sac --task pickandplace --her --seed 1 --checkpoint checkpoints/sac_her_pickandplace_seed1/best.pt

Output
------
    Success rate, mean reward, std reward, and per-episode breakdown.
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
    p.add_argument("--algo",       type=str, required=True,
                   choices=["td3", "sac", "ddpg", "ppo"])
    p.add_argument("--task",       type=str, required=True,
                   choices=["reach", "pickandplace", "push", "slide"])
    p.add_argument("--her",        action="store_true", default=False)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--episodes",   type=int, default=100,
                   help="Number of evaluation episodes.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pt checkpoint file. Auto-detected if not set.")
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--render",     action="store_true", default=False,
                   help="Render environment during evaluation.")
    return p.parse_args()


def auto_checkpoint_path(algo, task, her, seed):
    her_tag = "_her" if her else ""
    run_tag = f"{algo}{her_tag}_{task}_seed{seed}"
    best    = ROOT / "checkpoints" / run_tag / "best.pt"
    final   = ROOT / "checkpoints" / run_tag / "final.pt"
    if best.exists():
        return str(best)
    if final.exists():
        return str(final)
    return None


def evaluate_scratch_agent(agent, env, n_episodes: int, deterministic: bool = True):
    """Evaluate a from-scratch TD3/SAC agent."""
    successes, rewards = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(200):
            if hasattr(agent, "select_action"):
                # TD3
                action = agent.select_action(obs, add_noise=not deterministic)
            else:
                action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                successes.append(float(info.get("is_success", 0.0)))
                rewards.append(ep_reward)
                break
    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "n_episodes":   n_episodes,
    }


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from envs.wrappers import make_env
    render_mode = "human" if args.render else None
    env = make_env(args.task, flatten=True, render_mode=render_mode)

    ckpt_path = args.checkpoint or auto_checkpoint_path(
        args.algo, args.task, args.her, args.seed
    )

    print(f"[eval] algo={args.algo}  task={args.task}  "
          f"her={args.her}  seed={args.seed}")
    print(f"[eval] checkpoint={ckpt_path}")
    print(f"[eval] episodes={args.episodes}  device={device}")

    if args.algo == "td3":
        from algorithms.scratch.td3 import TD3
        obs_dim    = env.observation_space.shape[0]
        act_dim    = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = TD3(obs_dim=obs_dim, act_dim=act_dim,
                    max_action=max_action, device=device)
        if ckpt_path:
            agent.load(ckpt_path)
        results = evaluate_scratch_agent(agent, env, args.episodes)

    elif args.algo == "sac":
        from algorithms.scratch.sac import SAC
        obs_dim    = env.observation_space.shape[0]
        act_dim    = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = SAC(obs_dim=obs_dim, act_dim=act_dim,
                    max_action=max_action, device=device)
        if ckpt_path:
            agent.load(ckpt_path)
        results = evaluate_scratch_agent(agent, env, args.episodes)

    elif args.algo in ("ddpg", "ppo"):
        # SB3 evaluation
        import stable_baselines3 as sb3
        from stable_baselines3.common.evaluation import evaluate_policy
        Cls = sb3.DDPG if args.algo == "ddpg" else sb3.PPO
        model = Cls.load(ckpt_path or "", env=env)
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=args.episodes,
                                        deterministic=True)
        results = {"mean_reward": mean_r, "std_reward": std_r,
                   "n_episodes": args.episodes}

    print()
    print("=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print("=" * 50)

    # Save results to CSV
    import csv
    out_path = ROOT / "results" / f"eval_{args.algo}{'_her' if args.her else ''}_{args.task}_seed{args.seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"[eval] Results saved: {out_path}")


if __name__ == "__main__":
    main()
