import pathlib
from typing import Optional

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her import HerReplayBuffer

TASK_MAP = {
    "reach":        "PandaReach-v3",
    "pickandplace": "PandaPickAndPlace-v3",
    "push":         "PandaPush-v3",
    "slide":        "PandaSlide-v3",
}


class SACBaseline:
    """Wrapper around SB3 SAC for robotic arm RL project."""

    def __init__(
        self,
        task:           str   = "reach",
        seed:           int   = 0,
        her:            bool  = False,
        log_dir:        str   = "logs",
        checkpoint_dir: str   = "checkpoints",
        **sb3_kwargs,
    ) -> None:
        assert task in TASK_MAP, f"Unknown task '{task}'"
        self.task           = task
        self.seed           = seed
        self.her            = her
        self.log_dir        = pathlib.Path(log_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._env_id        = TASK_MAP[task]

        self._env      = Monitor(gym.make(self._env_id))
        self._eval_env = Monitor(gym.make(self._env_id))

        replay_buffer_class  = HerReplayBuffer if her else None
        replay_buffer_kwargs = (
            {"n_sampled_goal": 4, "goal_selection_strategy": "future"}
            if her else {}
        )

        run_tag = f"sac{'_her' if her else ''}_{task}_seed{seed}"
        self._run_tag = run_tag

        default_kwargs = dict(
            policy               = "MultiInputPolicy" if her else "MlpPolicy",
            env                  = self._env,
            replay_buffer_class  = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            learning_rate        = 1e-3,
            buffer_size          = 1_000_000,
            learning_starts      = 1000,
            batch_size           = 256,
            tau                  = 0.005,
            gamma                = 0.98,
            verbose              = 1,
            tensorboard_log      = str(self.log_dir),
            seed                 = seed,
        )
        default_kwargs.update(sb3_kwargs)

        self.model = SAC(**default_kwargs)

    def train(self, total_timesteps: int = 1_000_000, eval_freq: int = 5000) -> None:
        ckpt_dir = self.checkpoint_dir / self._run_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EvalCallback(
                eval_env             = self._eval_env,
                best_model_save_path = str(ckpt_dir),
                log_path             = str(ckpt_dir),
                eval_freq            = eval_freq,
                n_eval_episodes      = 20,
                deterministic        = True,
                verbose              = 1,
            ),
            CheckpointCallback(
                save_freq   = eval_freq * 10,
                save_path   = str(ckpt_dir),
                name_prefix = "step",
                verbose     = 1,
            ),
        ]

        print(f"[SAC Baseline] Starting training: {self._run_tag}")
        self.model.learn(
            total_timesteps     = total_timesteps,
            callback            = callbacks,
            tb_log_name         = self._run_tag,
            reset_num_timesteps = False,
        )
        self.model.save(str(ckpt_dir / "final.zip"))

    def save(self, path: str) -> None:
        self.model.save(path)
        print(f"[SAC Baseline] Saved: {path}")

    @classmethod
    def load(cls, path: str, task: str, **kwargs) -> "SACBaseline":
        instance = cls(task=task, **kwargs)
        instance.model = SAC.load(path, env=instance._env)
        return instance
