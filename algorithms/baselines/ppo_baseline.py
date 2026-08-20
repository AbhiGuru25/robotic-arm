"""
algorithms/baselines/ppo_baseline.py
=====================================
PPO (Proximal Policy Optimization) baseline using Stable-Baselines3.

PPO is our on-policy baseline.  It does not use a replay buffer and
cannot be combined with HER (HER requires off-policy learning), which
is a core finding of this research: comparing the off-policy + HER
paradigm against on-policy methods for sparse-reward manipulation.

Reference
---------
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
"Proximal Policy Optimization Algorithms."
arXiv:1707.06347.

Usage
-----
    from algorithms.baselines.ppo_baseline import PPOBaseline

    agent = PPOBaseline(task="reach", seed=0)
    agent.train(total_timesteps=1_000_000)
    agent.save("checkpoints/ppo_reach_seed0")
"""

import pathlib
from typing import Optional

import gymnasium as gym
import panda_gym  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


TASK_MAP = {
    "reach":         "PandaReach-v3",
    "pickandplace":  "PandaPickAndPlace-v3",
    "push":          "PandaPush-v3",
    "slide":         "PandaSlide-v3",
}


class PPOBaseline:
    """Wrapper around SB3 PPO for the robotic arm RL project.

    Parameters
    ----------
    task : str
        Task name: "reach" | "pickandplace" | etc.
    seed : int
        Random seed.
    n_envs : int
        Number of parallel environments (PPO benefits from parallelism).
    log_dir : str
        TensorBoard log directory.
    checkpoint_dir : str
        Checkpoint save directory.
    **sb3_kwargs
        Additional kwargs forwarded to the SB3 PPO constructor.
    """

    def __init__(
        self,
        task:           str = "reach",
        seed:           int = 0,
        n_envs:         int = 4,
        log_dir:        str = "logs",
        checkpoint_dir: str = "checkpoints",
        **sb3_kwargs,
    ) -> None:
        assert task in TASK_MAP, f"Unknown task '{task}'"
        self.task           = task
        self.seed           = seed
        self.log_dir        = pathlib.Path(log_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._env_id        = TASK_MAP[task]

        # panda-gym dict obs with FlattenObservation for PPO (which uses MlpPolicy)
        from gymnasium.wrappers import FlattenObservation

        def make_single_env():
            env = gym.make(self._env_id, render_mode=None)
            env = FlattenObservation(env)
            env = Monitor(env)
            return env

        # Vectorised envs for parallel rollout collection
        self._env = DummyVecEnv([make_single_env for _ in range(n_envs)])
        self._eval_env = Monitor(FlattenObservation(gym.make(self._env_id, render_mode=None)))

        run_tag = f"ppo_{task}_seed{seed}"
        self._run_tag = run_tag

        default_kwargs = dict(
            policy          = "MlpPolicy",
            env             = self._env,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.0,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            verbose         = 1,
            tensorboard_log = str(self.log_dir),
            seed            = seed,
        )
        default_kwargs.update(sb3_kwargs)

        self.model = PPO(**default_kwargs)

    def train(self, total_timesteps: int = 1_000_000, eval_freq: int = 10_000) -> None:
        """Train the PPO agent."""
        ckpt_dir = self.checkpoint_dir / self._run_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EvalCallback(
                eval_env             = self._eval_env,
                best_model_save_path = str(ckpt_dir / "best"),
                log_path             = str(ckpt_dir),
                eval_freq            = eval_freq,
                n_eval_episodes      = 20,
                deterministic        = True,
                verbose              = 1,
            ),
            CheckpointCallback(
                save_freq   = 100_000,
                save_path   = str(ckpt_dir),
                name_prefix = "ckpt",
                verbose     = 1,
            ),
        ]

        print(f"[PPO Baseline] Starting training: {self._run_tag}")
        self.model.learn(
            total_timesteps     = total_timesteps,
            callback            = callbacks,
            tb_log_name         = self._run_tag,
            reset_num_timesteps = True,
        )

    def save(self, path: str) -> None:
        self.model.save(path)
        print(f"[PPO Baseline] Saved: {path}")

    @classmethod
    def load(cls, path: str, task: str, **kwargs) -> "PPOBaseline":
        instance = cls(task=task, **kwargs)
        instance.model = PPO.load(path, env=instance._env)
        return instance

    def evaluate(self, n_episodes: int = 50, deterministic: bool = True) -> dict:
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_r, std_r = evaluate_policy(
            self.model, self._eval_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
        )
        return {"mean_reward": mean_r, "std_reward": std_r}
