"""
utils/replay_buffer.py
======================
Standard off-policy experience replay buffer.

Stores (state, action, reward, next_state, done) transitions as
contiguous numpy arrays and samples random mini-batches for training.

Used by: TD3, SAC (without HER).
For HER-augmented training, see utils/her_buffer.py.

Design Notes
------------
- Pre-allocated numpy arrays — avoids per-step Python heap allocations.
- Ring-buffer semantics with a simple pointer + size counter.
- Returns torch Tensors on the correct device for direct use in loss
  computation without additional data movement.

Usage
-----
    buf = ReplayBuffer(obs_dim=25, act_dim=4, capacity=1_000_000, device="cuda")

    # Store a transition
    buf.add(obs, action, reward, next_obs, done)

    # Sample a mini-batch
    batch = buf.sample(256)
    # batch.obs, batch.actions, batch.rewards, batch.next_obs, batch.dones
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    """A mini-batch of transitions as PyTorch tensors."""
    obs:       torch.Tensor   # (B, obs_dim)
    actions:   torch.Tensor   # (B, act_dim)
    rewards:   torch.Tensor   # (B, 1)
    next_obs:  torch.Tensor   # (B, obs_dim)
    dones:     torch.Tensor   # (B, 1)  — float, 1.0 = terminal


class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector.
    act_dim : int
        Dimensionality of the action vector.
    capacity : int
        Maximum number of transitions stored.  Oldest transitions are
        overwritten when the buffer is full (ring buffer).
    device : str | torch.device
        PyTorch device for sampled tensors (e.g. "cpu", "cuda").
    """

    def __init__(
        self,
        obs_dim:  int,
        act_dim:  int,
        capacity: int = 1_000_000,
        device:   str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0   # write pointer (next slot to overwrite)
        self._size    = 0   # current number of stored transitions

        # Pre-allocated storage arrays
        self._obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._actions  = np.zeros((capacity, act_dim),  dtype=np.float32)
        self._rewards  = np.zeros((capacity, 1),         dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._dones    = np.zeros((capacity, 1),         dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Store a single transition.

        Parameters
        ----------
        obs, next_obs : np.ndarray of shape (obs_dim,)
        action : np.ndarray of shape (act_dim,)
        reward : float
        done : bool
            True if the episode terminated (not just truncated).
            For time-limit truncation, pass done=False so the
            bootstrap target is valid.
        """
        self._obs[self._ptr]      = obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr]    = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample a uniformly random mini-batch.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        Batch
            Named tuple of PyTorch tensors on self.device.

        Raises
        ------
        ValueError
            If batch_size > current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch_size={batch_size} > buffer size={self._size}. "
                "Fill the buffer more before sampling."
            )
        idxs = np.random.randint(0, self._size, size=batch_size)

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.FloatTensor(arr[idxs]).to(self.device)

        return Batch(
            obs      = to_tensor(self._obs),
            actions  = to_tensor(self._actions),
            rewards  = to_tensor(self._rewards),
            next_obs = to_tensor(self._next_obs),
            dones    = to_tensor(self._dones),
        )

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size}/{self.capacity}, "
            f"obs_dim={self._obs.shape[1]}, act_dim={self._actions.shape[1]})"
        )
