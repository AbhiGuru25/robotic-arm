"""
envs/wrappers.py
================
Custom Gymnasium environment wrappers for the robotic arm RL project.

Wrappers
--------
FlattenGoalObsWrapper
    Converts the goal-conditioned dict observation (with keys
    "observation", "achieved_goal", "desired_goal") into a single
    flat numpy array for algorithms that do not natively handle dicts
    (e.g. our from-scratch TD3 / SAC implementations).

TimeLimitWrapper
    Enforces a maximum number of steps per episode and sets the
    `truncated` flag (Gymnasium v0.26+ API) when the limit is hit.

DenseRewardWrapper
    Optionally replaces the sparse binary reward with a dense shaped
    reward equal to the negative L2 distance between achieved_goal and
    desired_goal.  Used for ablation experiments only — the main
    experiments use sparse rewards + HER.

Usage
-----
    import gymnasium as gym
    import panda_gym
    from envs.wrappers import FlattenGoalObsWrapper, TimeLimitWrapper

    env = gym.make("PandaReach-v3", render_mode=None)
    env = FlattenGoalObsWrapper(env)
    env = TimeLimitWrapper(env, max_steps=50)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


class FlattenGoalObsWrapper(gym.ObservationWrapper):
    """Flatten a goal-conditioned dict observation into a single vector.

    The wrapped environment must have a Dict observation space with at
    least the keys ``"observation"``, ``"achieved_goal"``, and
    ``"desired_goal"``.

    After wrapping the observation space becomes a single ``Box`` whose
    size equals the sum of the three sub-spaces, concatenated in the
    order: observation | achieved_goal | desired_goal.

    Parameters
    ----------
    env : gym.Env
        A goal-conditioned environment (e.g. panda-gym task).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict), (
            "FlattenGoalObsWrapper expects a Dict observation space."
        )

        obs_dim = obs_space["observation"].shape[0]
        ag_dim  = obs_space["achieved_goal"].shape[0]
        dg_dim  = obs_space["desired_goal"].shape[0]
        total   = obs_dim + ag_dim + dg_dim

        low  = np.full(total, -np.inf, dtype=np.float32)
        high = np.full(total,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Store sub-dimensions for later use (e.g. HER needs them)
        self.obs_dim = obs_dim
        self.ag_dim  = ag_dim
        self.dg_dim  = dg_dim

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate sub-observations into a flat vector."""
        return np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"],
        ], axis=-1).astype(np.float32)


class TimeLimitWrapper(gym.Wrapper):
    """Enforce a per-episode step limit with the Gymnasium truncated flag.

    Parameters
    ----------
    env : gym.Env
        Environment to wrap.
    max_steps : int
        Maximum number of steps before the episode is truncated.
        Default is 50 (matches panda-gym defaults for most tasks).
    """

    def __init__(self, env: gym.Env, max_steps: int = 50) -> None:
        super().__init__(env)
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


class DenseRewardWrapper(gym.RewardWrapper):
    """Replace sparse binary reward with negative L2 distance reward.

    **Important:** This wrapper is for ablation / debugging purposes only.
    The main experiments use the original sparse binary reward together
    with HER.  Dense rewards remove the need for HER but may introduce
    reward engineering bias.

    The dense reward is:
        r = -||achieved_goal - desired_goal||_2

    Parameters
    ----------
    env : gym.Env
        Goal-conditioned environment with a dict observation space.
    scale : float
        Optional scaling factor applied to the negative distance.
        Default 1.0 — reward lies in (-∞, 0].
    """

    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        # panda-gym stores the last achieved/desired goals on the info dict
        # via the compute_reward() call; we re-compute directly from the
        # most recent observation stored in self.env.
        try:
            ag = self.env.unwrapped.robot.get_obs()
            # Fallback: return original reward if we cannot retrieve goals
        except Exception:
            return reward
        return reward  # subclass override — dense computation done in step

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, dict):
            achieved = obs["achieved_goal"]
            desired  = obs["desired_goal"]
            dense_r  = -float(np.linalg.norm(achieved - desired)) * self.scale
            return obs, dense_r, terminated, truncated, info
        return obs, _reward, terminated, truncated, info


def make_env(task: str,
             render_mode: Optional[str] = None,
             max_steps: int = 50,
             flatten: bool = False) -> gym.Env:
    """Factory function: create a panda-gym environment with standard wrappers.

    Parameters
    ----------
    task : str
        One of {"reach", "pickandplace", "push", "slide"}.
    render_mode : str, optional
        "human" for on-screen rendering, "rgb_array" for video recording,
        None for headless training (default).
    max_steps : int
        Episode time limit.  Default 50.
    flatten : bool
        If True, apply FlattenGoalObsWrapper (needed for from-scratch TD3/SAC).
        If False, return dict obs space (compatible with SB3 HER).

    Returns
    -------
    gym.Env
        Wrapped Gymnasium environment.
    """
    task_map = {
        "reach":         "PandaReach-v3",
        "pickandplace":  "PandaPickAndPlace-v3",
        "push":          "PandaPush-v3",
        "slide":         "PandaSlide-v3",
    }
    assert task in task_map, (
        f"Unknown task '{task}'. Choose from: {list(task_map.keys())}"
    )

    env = gym.make(task_map[task], render_mode=render_mode)
    env = TimeLimitWrapper(env, max_steps=max_steps)
    if flatten:
        env = FlattenGoalObsWrapper(env)
    return env
