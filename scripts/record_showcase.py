"""
scripts/record_showcase.py
===========================
Generates a cinematic HD multi-scene video showcase of the Franka Panda Robotic Arm.
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description="Record a cinematic showcase video.")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--episodes_per_task", type=int, default=2)
    args = p.parse_args()

    import gymnasium as gym
    import panda_gym  # noqa: F401
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv

    try:
        import imageio
    except ImportError:
        print("[showcase] Error: imageio not installed. Run: pip install imageio imageio-ffmpeg")
        sys.exit(1)

    vid_dir = ROOT / "results" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    out_path = vid_dir / "robotic_arm_showcase_hd.mp4"

    tasks = [
        ("reach", "td3", "PandaReach-v3"),
        ("pickandplace", "sac", "PandaPickAndPlace-v3"),
    ]

    all_frames = []

    for task_name, algo, env_id in tasks:
        print(f"[showcase] Recording cinematic scenes for {task_name.upper()}...")

        def make_env():
            return gym.make(env_id, render_mode="rgb_array", width=800, height=600)

        env = DummyVecEnv([make_env])
        run_tag = f"{algo}_her_{task_name}_seed0"
        ckpt_dir = ROOT / "checkpoints" / run_tag

        candidates = list(ckpt_dir.glob("*.zip")) if ckpt_dir.exists() else []

        model = None
        if candidates:
            cand = str(candidates[0])
            try:
                Cls = SAC if algo == "sac" else TD3
                model = Cls.load(cand, env=env)
                print(f"[showcase] Loaded model: {cand}")
            except Exception as e:
                print(f"[showcase] Could not load model {cand}: {e}")

        for ep in range(args.episodes_per_task):
            obs = env.reset()
            for step in range(80):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, r, done, info = env.step(action)
                frame = env.envs[0].render()
                if frame is not None:
                    all_frames.append(frame)

                if done[0]:
                    break

        env.close()

    if all_frames:
        print(f"[showcase] Rendering {len(all_frames)} HD frames to {out_path} at {args.fps} FPS...")
        imageio.mimwrite(str(out_path), all_frames, fps=args.fps, quality=9)
        print(f"[showcase] Cinematic HD Video saved successfully: {out_path}")
    else:
        print("[showcase] No frames collected.")


if __name__ == "__main__":
    main()
