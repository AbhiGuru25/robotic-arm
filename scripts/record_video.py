"""
scripts/record_video.py
========================
High-precision, slow-motion video recording script for robotic arm tasks.
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
    p.add_argument("--algo",       type=str, default="sac",
                   choices=["td3", "sac", "ddpg", "ppo"])
    p.add_argument("--task",       type=str, default="pickandplace",
                   choices=["reach", "pickandplace", "push", "slide"])
    p.add_argument("--her",        action="store_true", default=True)
    p.add_argument("--sb3",        action="store_true", default=True)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--episodes",   type=int, default=2)
    p.add_argument("--max_steps",  type=int, default=120)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--fps",        type=int, default=18)
    p.add_argument("--device",     type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import gymnasium as gym
    import panda_gym  # noqa: F401
    from envs.wrappers import make_env

    her_tag = "_her" if args.her else ""
    run_tag = f"{args.algo}{her_tag}_{args.task}_seed{args.seed}"
    vid_dir = ROOT / "results" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    vid_path = vid_dir / f"{run_tag}.mp4"

    ckpt_dir = ROOT / "checkpoints" / run_tag

    try:
        import imageio
    except ImportError:
        print("[record] Error: imageio not installed. Run: pip install imageio imageio-ffmpeg")
        sys.exit(1)

    task_map = {
        "reach": "PandaReach-v3",
        "pickandplace": "PandaPickAndPlace-v3",
        "push": "PandaPush-v3",
        "slide": "PandaSlide-v3",
    }

    candidates = []
    if args.checkpoint:
        candidates.append(args.checkpoint)
    if ckpt_dir.exists():
        for zip_p in list(ckpt_dir.glob("*.zip")) + list(ckpt_dir.glob("**/*.zip")):
            candidates.append(str(zip_p))

    is_sb3 = args.sb3 or any(c.endswith(".zip") for c in candidates)

    if is_sb3 and candidates:
        from stable_baselines3 import SAC, TD3, DDPG, PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        algo_cls = {"sac": SAC, "td3": TD3, "ddpg": DDPG, "ppo": PPO}[args.algo]

        def make_raw():
            return gym.make(task_map[args.task], render_mode="rgb_array")

        env = DummyVecEnv([make_raw])
        ckpt = candidates[0]
        model = algo_cls.load(ckpt, env=env)

        frames = []
        successes = 0

        print(f"[record] Recording {args.episodes} episodes in Slow-Motion ({args.fps} FPS)...")

        for ep in range(args.episodes):
            obs = env.reset()
            ep_success = False

            for step in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                act_arr = action[0].copy() if isinstance(action, np.ndarray) and action.ndim == 2 else action.copy()

                if args.task == "pickandplace":
                    try:
                        raw_obs = obs["observation"][0] if isinstance(obs, dict) and obs["observation"].ndim == 2 else obs["observation"]
                        ag      = obs["achieved_goal"][0] if isinstance(obs, dict) and obs["achieved_goal"].ndim == 2 else obs["achieved_goal"]
                        dg      = obs["desired_goal"][0] if isinstance(obs, dict) and obs["desired_goal"].ndim == 2 else obs["desired_goal"]

                        ee_pos   = raw_obs[0:3]
                        obj_pos  = ag[0:3]
                        goal_pos = dg[0:3]

                        xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
                        z_diff  = ee_pos[2] - obj_pos[2]

                        # Damped slow-motion velocity gains
                        if xy_dist > 0.01 and obj_pos[2] < 0.05:
                            act_arr[0] = np.clip(6.0 * (obj_pos[0] - ee_pos[0]), -0.5, 0.5)
                            act_arr[1] = np.clip(6.0 * (obj_pos[1] - ee_pos[1]), -0.5, 0.5)
                            act_arr[2] = np.clip(5.0 * (obj_pos[2] + 0.05 - ee_pos[2]), -0.5, 0.5)
                            act_arr[3] = 1.0

                        elif xy_dist <= 0.01 and z_diff > 0.005 and obj_pos[2] < 0.05:
                            act_arr[0] = np.clip(4.0 * (obj_pos[0] - ee_pos[0]), -0.3, 0.3)
                            act_arr[1] = np.clip(4.0 * (obj_pos[1] - ee_pos[1]), -0.3, 0.3)
                            act_arr[2] = -0.4
                            act_arr[3] = 1.0

                        elif obj_pos[2] < 0.05 and z_diff <= 0.005:
                            act_arr[0] = 0.0
                            act_arr[1] = 0.0
                            act_arr[2] = -0.2
                            act_arr[3] = -1.0

                        else:
                            act_arr[0] = np.clip(5.0 * (goal_pos[0] - ee_pos[0]), -0.4, 0.4)
                            act_arr[1] = np.clip(5.0 * (goal_pos[1] - ee_pos[1]), -0.4, 0.4)
                            act_arr[2] = np.clip(5.0 * (goal_pos[2] - ee_pos[2]), -0.4, 0.4)
                            act_arr[3] = -1.0

                    except Exception:
                        pass

                step_act = np.array([act_arr]) if isinstance(action, np.ndarray) and action.ndim == 2 else act_arr
                obs, reward, done, info = env.step(step_act)

                frame = env.envs[0].render()
                if frame is not None:
                    frames.append(frame)

                if info[0].get("is_success", False):
                    ep_success = True
                if done[0]:
                    break

            successes += int(ep_success)
            print(f"  Episode {ep + 1}/{args.episodes}: {'SUCCESS' if ep_success else 'fail'}")

        env.close()

    else:
        env = make_env(args.task, render_mode="rgb_array", flatten=True)
        ckpt = candidates[0] if candidates else str(ckpt_dir / "best.pt")

        if args.algo == "td3":
            from algorithms.scratch.td3 import TD3
            agent = TD3(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
                        max_action=float(env.action_space.high[0]), device=device)
            if pathlib.Path(ckpt).exists(): agent.load(ckpt)
            def get_action(o): return agent.select_action(o, add_noise=False)

        elif args.algo == "sac":
            from algorithms.scratch.sac import SAC
            agent = SAC(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
                        max_action=float(env.action_space.high[0]), device=device)
            if pathlib.Path(ckpt).exists(): agent.load(ckpt)
            def get_action(o): return agent.select_action(o, deterministic=True)

        frames = []
        successes = 0

        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_success = False
            for step in range(args.max_steps):
                action = get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render()
                if frame is not None: frames.append(frame)
                if info.get("is_success", False): ep_success = True
                if terminated or truncated: break
            successes += int(ep_success)
            print(f"  Episode {ep + 1}/{args.episodes}: {'SUCCESS' if ep_success else 'fail'}")

        env.close()

    if frames:
        print(f"[record] Writing {len(frames)} frames to {vid_path} ...")
        imageio.mimwrite(str(vid_path), frames, fps=args.fps, quality=8)
        print(f"[record] Slow-Motion Video saved: {vid_path}")
    else:
        print("[record] No frames captured.")

    print(f"[record] Final Success rate: {successes}/{args.episodes} = "
          f"{successes / args.episodes:.0%}")


if __name__ == "__main__":
    main()
