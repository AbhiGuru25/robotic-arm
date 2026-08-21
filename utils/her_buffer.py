"""
utils/her_buffer.py
====================
Hindsight Experience Replay (HER) buffer — implemented from scratch.

Reference
---------
Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R.,
Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017).
"Hindsight Experience Replay."
NeurIPS 2017. arXiv:1707.01495.

Algorithm
---------
HER is a replay buffer augmentation strategy for goal-conditioned RL.
It is agnostic to the RL algorithm — compatible with any off-policy
method (DDPG, TD3, SAC).

For every episode trajectory {(s_t, a_t, r_t, s_{t+1}) : t=0..T}
collected while pursuing goal g:

1. Store the original transitions (s_t, a_t, r(s_{t+1}, g), s_{t+1}, g)
   in the replay buffer.

2. For each step t, sample k hindsight goals g' from future states in
   the same episode: g' ∈ {s_{t+1}, ..., s_T}  (the "future" strategy).

3. Recompute reward: r' = r(s_{t+1}, g') — since g' was actually
   reached, this is a SUCCESS reward even though the episode "failed."

4. Store the k relabelled transitions (s_t, a_t, r', s_{t+1}, g') too.

Result: every episode generates learning signal, regardless of success.

Observation format
------------------
The HER buffer stores FLAT observations of the form:
    [observation | achieved_goal | desired_goal]
which is what FlattenGoalObsWrapper produces.

For relabelling, the goal portion (desired_goal) is replaced with the
hindsight goal g'.  The reward is recomputed using the binary sparse
reward function: r = 0 if ||ag - dg|| <= threshold else -1.

Parameters
----------
obs_dim : int
    Full flat observation dimension (obs + ag + dg concatenated).
act_dim : int
    Action dimension.
ag_dim : int
    Achieved goal dimension.
dg_dim : int
    Desired goal dimension.
capacity : int
    Maximum number of stored transitions (after HER relabelling).
k : int
    Number of hindsight goals to sample per transition.
max_eps_len : int
    Maximum episode length (for episode buffer pre-allocation).
device : str
    PyTorch device for sampled tensors.
goal_threshold : float
    Success threshold for sparse reward: ||ag - dg|| <= threshold -> r=0.
strategy : str
    HER goal sampling strategy. One of {"future", "episode", "final"}.
    "future" is the recommended default (best empirical performance).
"""

from typing import Optional

import numpy as np
import torch

from utils.replay_buffer import Batch, ReplayBuffer


class HERBuffer:
    """Hindsight Experience Replay buffer.

    Wraps a standard ReplayBuffer with episode-level storage and
    post-episode HER goal relabelling.

    Workflow
    --------
    1. Call ``add_step()`` at each environment step.
    2. Call ``finish_episode()`` at episode end — this triggers HER
       relabelling and flushes all transitions to the inner buffer.
    3. Call ``sample()`` to get a mini-batch (identical API to ReplayBuffer).
    """

    def __init__(
        self,
        obs_dim:        int,
        act_dim:        int,
        ag_dim:         int,
        dg_dim:         int,
        capacity:       int   = 1_000_000,
        k:              int   = 4,
        max_eps_len:    int   = 50,
        device:         str   = "cpu",
        goal_threshold: float = 0.05,
        strategy:       str   = "future",
    ) -> None:
        assert strategy in ("future", "episode", "final"),                 f"Unknown HER strategy '{strategy}'. Choose: future | episode | final"

        self.obs_dim        = obs_dim
        self.act_dim        = act_dim
        self.ag_dim         = ag_dim
        self.dg_dim         = dg_dim
        self.k              = k
        self.max_eps_len    = max_eps_len
        self.goal_threshold = goal_threshold
        self.strategy       = strategy

        # Compute sub-array offsets in the flat observation
        # flat obs = [raw_obs | achieved_goal | desired_goal]
        raw_dim = obs_dim - ag_dim - dg_dim
        self._raw_end  = raw_dim               # obs[:raw_end]     = raw obs
        self._ag_start = raw_dim               # obs[ag_start:ag_end] = ag
        self._ag_end   = raw_dim + ag_dim
        self._dg_start = raw_dim + ag_dim      # obs[dg_start:]    = dg

        # ── Inner replay buffer (stores all final transitions) ─────────
        self._buffer = ReplayBuffer(
            obs_dim  = obs_dim,
            act_dim  = act_dim,
            capacity = capacity,
            device   = device,
        )

        # ── Episode-level temporary storage ───────────────────────────
        self._ep_obs     : list = []   # flat observations
        self._ep_actions : list = []
        self._ep_rewards : list = []
        self._ep_next_obs: list = []
        self._ep_dones   : list = []

    # ------------------------------------------------------------------
    # Public API: step-by-step collection
    # ------------------------------------------------------------------

    def add_step(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
        ag:       Optional[np.ndarray] = None,
        dg:       Optional[np.ndarray] = None,
        next_ag:  Optional[np.ndarray] = None,
    ) -> None:
        """Record one environment transition.

        If ag / dg / next_ag are not provided, they are extracted from
        the flat ``obs`` and ``next_obs`` arrays using stored offsets.

        Parameters
        ----------
        obs, next_obs : np.ndarray (obs_dim,) — flat goal-conditioned obs
        action : np.ndarray (act_dim,)
        reward : float
        done : bool — True only for terminal (not time-limit) episodes
        ag, dg, next_ag : np.ndarray, optional — extracted if not provided
        """
        self._ep_obs.append(obs.copy())
        self._ep_actions.append(action.copy())
        self._ep_rewards.append(float(reward))
        self._ep_next_obs.append(next_obs.copy())
        self._ep_dones.append(float(done))

    def finish_episode(self) -> None:
        """Called at episode end — performs HER relabelling and stores all
        transitions into the inner replay buffer.

        This implements the HER "future" strategy:
        For each transition at step t, we sample k future achieved goals
        from steps {t+1, ..., T} in the same episode and store additional
        relabelled transitions.
        """
        T = len(self._ep_obs)
        if T == 0:
            return

        obs_arr      = np.array(self._ep_obs,      dtype=np.float32)  # (T, obs_dim)
        actions_arr  = np.array(self._ep_actions,  dtype=np.float32)  # (T, act_dim)
        rewards_arr  = np.array(self._ep_rewards,  dtype=np.float32)  # (T,)
        next_obs_arr = np.array(self._ep_next_obs, dtype=np.float32)  # (T, obs_dim)
        dones_arr    = np.array(self._ep_dones,    dtype=np.float32)  # (T,)

        # Extract achieved goals from next_obs (used as hindsight goals)
        # Shape: (T, ag_dim)
        next_ags = next_obs_arr[:, self._ag_start:self._ag_end]

        # ── Store original transitions ─────────────────────────────────
        for t in range(T):
            self._buffer.add(
                obs_arr[t], actions_arr[t], rewards_arr[t],
                next_obs_arr[t], bool(dones_arr[t])
            )

        # ── HER relabelling: store k additional relabelled transitions ─
        for t in range(T):
            if self.strategy == "future":
                # Sample k indices uniformly from [t+1, T)
                future_indices = np.random.randint(
                    t + 1, T,
                    size=min(self.k, T - 1 - t)
                ) if t < T - 1 else []
            elif self.strategy == "episode":
                future_indices = np.random.randint(0, T, size=self.k)
            elif self.strategy == "final":
                future_indices = [T - 1] * self.k

            for f_idx in future_indices:
                f_idx = min(int(f_idx), T - 1)
                # Hindsight goal = achieved goal at the future step
                her_goal = next_ags[f_idx]                            # (ag_dim,)

                # Build relabelled obs and next_obs by replacing desired_goal
                obs_relabelled      = obs_arr[t].copy()
                next_obs_relabelled = next_obs_arr[t].copy()
                obs_relabelled[self._dg_start:]      = her_goal
                next_obs_relabelled[self._dg_start:] = her_goal

                # Recompute sparse reward with hindsight goal
                her_reward = self._compute_reward(next_ags[t], her_goal)

                self._buffer.add(
                    obs_relabelled,
                    actions_arr[t],
                    her_reward,
                    next_obs_relabelled,
                    bool(dones_arr[t]),
                )

        # Clear episode buffer
        self._ep_obs.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        self._ep_next_obs.clear()
        self._ep_dones.clear()

    def sample(self, batch_size: int) -> Batch:
        """Sample a mini-batch (identical API to ReplayBuffer.sample)."""
        return self._buffer.sample(batch_size)

    @property
    def size(self) -> int:
        return self._buffer.size

    def __len__(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Sparse binary reward.

        Returns 0.0 (success) if ||ag - dg|| <= threshold, else -1.0.
        Matches panda-gym's internal reward function.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if distance <= self.goal_threshold else -1.0

    def __repr__(self) -> str:
        return (
            f"HERBuffer(size={self.size}, k={self.k}, "
            f"strategy={self.strategy!r}, "
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim})"
        )
