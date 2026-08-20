"""
algorithms/baselines/ddpg_baseline.py
======================================
DDPG baseline using Stable-Baselines3.

DDPG (Lillicrap et al., 2016) is our library-provided reference baseline.
We use the SB3 implementation rather than writing DDPG from scratch because:

1. DDPG's weaknesses (overestimation bias, instability) are well understood
   and are exactly what TD3/SAC improve upon — observing DDPG's failures
   *is* part of the research contribution.
2. Using SB3 for DDPG allows fair comparison against a thoroughly validated
   implementation.

Reference
---------
Lillicrap et al. (2016). Continuous Control with Deep Reinforcement Learning.
ICLR 2016. arXiv:1509.02971.

Usage
-----
    from algorithms.baselines.ddpg_baseline import DDPGBaseline

    agent = DDPGBaseline(task="reach", seed=0, her=False)
    agent.train(total_timesteps=500_000)
    agent.save("checkpoints/ddpg_reach_seed0")
"""

import pathlib
from typing import Optional

import gymnasium as gym
import panda_gym  # noqa: F401

from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

import numpy as np


# Map short task names to panda-gym environment IDs
TASK_MAP = {
    "reach":         "PandaReach-v3",
    "pickandplace":  "PandaPickAndPlace-v3",
    "push":          "PandaPush-v3",
    "slide":         "PandaSlide-v3",
}


class DDPGBaseline:
    """Wrapper around SB3 DDPG for the robotic arm RL project.

    Parameters
    ----------
    task : str
        Task name: "reach" | "pickandplace" | "push" | "slide".
    seed : int
        Random seed for reproducibility.
    her : bool
        If True, use HER replay buffer (HerReplayBuffer from SB3).
    log_dir : str
        Directory for TensorBoard logs.
    checkpoint_dir : str
        Directory for model checkpoints.
    **sb3_kwargs
        Additional keyword arguments forwarded to the SB3 DDPG constructor.
    """

    def __init__(
        self,
        task:            str   = "reach",
        seed:            int   = 0,
        her:             bool  = False,
        log_dir:         str   = "logs",
        checkpoint_dir:  str   = "checkpoints",
        **sb3_kwargs,
    ) -> None:
        assert task in TASK_MAP, f"Unknown task '{task}'. Options: {list(TASK_MAP)}"
        self.task           = task
        self.seed           = seed
        self.her            = her
        self.log_dir        = pathlib.Path(log_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._env_id        = TASK_MAP[task]

        # ── Build environments ─────────────────────────────────────────
        self._env      = Monitor(gym.make(self._env_id))
        self._eval_env = Monitor(gym.make(self._env_id))

        # ── Action noise (Gaussian for DDPG) ──────────────────────────
        n_actions    = self._env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean  = np.zeros(n_actions),
            sigma = 0.1 * np.ones(n_actions),
        )

        # ── Construct SB3 DDPG ────────────────────────────────────────
        replay_buffer_class  = HerReplayBuffer if her else None
        replay_buffer_kwargs = (
            {"n_sampled_goal": 4, "goal_selection_strategy": "future"}
            if her else {}
        )

        run_tag = f"ddpg{'_her' if her else ''}_{task}_seed{seed}"

        default_kwargs = dict(
            policy               = "MultiInputPolicy" if her else "MlpPolicy",
            env                  = self._env,
            action_noise         = action_noise,
            replay_buffer_class  = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            learning_rate        = 1e-3,
            buffer_size          = 1_000_000,
            learning_starts      = 1000,
            batch_size           = 256,
            tau                  = 0.005,
            gamma                = 0.98,
            train_freq           = 1,
            gradient_steps       = 1,
            verbose              = 1,
            tensorboard_log      = str(self.log_dir),
            seed                 = seed,
        )
        default_kwargs.update(sb3_kwargs)

        self.model = DDPG(**default_kwargs)
        self._run_tag = run_tag

    def train(self, total_timesteps: int = 500_000, eval_freq: int = 5_000) -> None:
        """Train the DDPG agent.

        Parameters
        ----------
        total_timesteps : int
            Total environment interaction steps.
        eval_freq : int
            Evaluate on eval_env every this many steps.
        """
        ckpt_dir = self.checkpoint_dir / self._run_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EvalCallback(
                eval_env          = self._eval_env,
                best_model_save_path = str(ckpt_dir / "best"),
                log_path          = str(ckpt_dir),
                eval_freq         = eval_freq,
                n_eval_episodes   = 20,
                deterministic     = True,
                verbose           = 1,
            ),
            CheckpointCallback(
                save_freq       = 50_000,
                save_path       = str(ckpt_dir),
                name_prefix     = "ckpt",
                verbose         = 1,
            ),
        ]

        print(f"[DDPG Baseline] Starting training: {self._run_tag}")
        self.model.learn(
            total_timesteps   = total_timesteps,
            callback          = callbacks,
            tb_log_name       = self._run_tag,
            reset_num_timesteps = True,
        )

    def save(self, path: str) -> None:
        """Save the SB3 model."""
        self.model.save(path)
        print(f"[DDPG Baseline] Saved: {path}")

    @classmethod
    def load(cls, path: str, task: str, **kwargs) -> "DDPGBaseline":
        """Load a saved SB3 DDPG model."""
        instance = cls(task=task, **kwargs)
        instance.model = DDPG.load(
            path, env=instance._env
        )
        return instance

    def evaluate(self, n_episodes: int = 50, deterministic: bool = True) -> dict:
        """Evaluate the trained agent.

        Returns
        -------
        dict with keys "mean_reward", "std_reward", "success_rate".
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_r, std_r = evaluate_policy(
            self.model, self._eval_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
        )
        return {"mean_reward": mean_r, "std_reward": std_r}
