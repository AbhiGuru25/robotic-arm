"""
scripts/record_video.py
========================
Record an MP4 video of a trained agent performing the task.

Usage
-----
    # Record TD3 + HER on PandaPickAndPlace
    python scripts/record_video.py --algo td3 --task pickandplace --her --seed 0

    # Specify checkpoint manually
    python scripts/record_video.py --algo sac --task pickandplace --her \
        --checkpoint checkpoints/sac_her_pickandplace_seed0/best.pt \
        --episodes 5

Output
------
results/videos/<algo>_<task>_<seed>.mp4
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Record a video of a trained agent.")
    p.add_argument("--algo",       type=str, required=True,
                   choices=["td3", "sac", "ddpg", "ppo"])
    p.add_argument("--task",       type=str, required=True,
                   choices=["reach", "pickandplace", "push", "slide"])
    p.add_argument("--her",        action="store_true", default=False)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--episodes",   type=int, default=5,
                   help="Number of episodes to record.")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--device",     type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from envs.wrappers import make_env

    # Use rgb_array render mode for video capture
    env = make_env(args.task, render_mode="rgb_array", flatten=True)

    her_tag  = "_her" if args.her else ""
    run_tag  = f"{args.algo}{her_tag}_{args.task}_seed{args.seed}"
    vid_dir  = ROOT / "results" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    vid_path = vid_dir / f"{run_tag}.mp4"

    # ── Load agent ────────────────────────────────────────────────────
    ckpt = args.checkpoint or str(
        ROOT / "checkpoints" / run_tag / "best.pt"
    )
    if args.algo == "td3":
        from algorithms.scratch.td3 import TD3
        obs_dim    = env.observation_space.shape[0]
        act_dim    = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = TD3(obs_dim=obs_dim, act_dim=act_dim,
                    max_action=max_action, device=device)
        if pathlib.Path(ckpt).exists():
            agent.load(ckpt)
        def get_action(obs): return agent.select_action(obs, add_noise=False)

    elif args.algo == "sac":
        from algorithms.scratch.sac import SAC
        obs_dim    = env.observation_space.shape[0]
        act_dim    = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = SAC(obs_dim=obs_dim, act_dim=act_dim,
                    max_action=max_action, device=device)
        if pathlib.Path(ckpt).exists():
            agent.load(ckpt)
        def get_action(obs): return agent.select_action(obs, deterministic=True)

    elif args.algo in ("ddpg", "ppo"):
        import stable_baselines3 as sb3
        Cls   = sb3.DDPG if args.algo == "ddpg" else sb3.PPO
        model = Cls.load(ckpt, env=env)
        def get_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

    # ── Record episodes ────────────────────────────────────────────────
    try:
        import imageio
    except ImportError:
        print("[record] imageio not installed. Run: pip install imageio imageio-ffmpeg")
        sys.exit(1)

    frames = []
    successes = 0

    print(f"[record] Recording {args.episodes} episodes of {run_tag}...")
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_success = False
        for step in range(200):
            action = get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                ep_success = info.get("is_success", False)
                break
        successes += int(ep_success)
        print(f"  Episode {ep + 1}/{args.episodes}: {'SUCCESS' if ep_success else 'fail'}")

    env.close()

    if frames:
        print(f"[record] Writing {len(frames)} frames to {vid_path} ...")
        imageio.mimwrite(str(vid_path), frames, fps=args.fps, quality=8)
        print(f"[record] Video saved: {vid_path}")
    else:
        print("[record] No frames captured. Make sure render_mode='rgb_array' works.")

    print(f"[record] Success rate: {successes}/{args.episodes} = "
          f"{successes / args.episodes:.0%}")


if __name__ == "__main__":
    main()
