"""
algorithms/scratch/td3.py
==========================
TD3 (Twin Delayed DDPG) — implemented from scratch.

Reference
---------
Fujimoto, S., van Hoof, H., & Meger, D. (2018).
"Addressing Function Approximation Error in Actor-Critic Methods."
ICML 2018. arXiv:1802.09477.

Algorithm Summary
-----------------
TD3 improves upon DDPG with three targeted fixes to overestimation bias
and training instability:

1. **Clipped Double Q-Learning (Twin Critics)**
   Two independent Q-networks Q1, Q2.  Bellman target uses min(Q1, Q2)
   to reduce overestimation bias.

2. **Delayed Policy Updates**
   The actor and target networks are updated every `policy_delay` critic
   gradient steps (default: 2).  Actor updates are guided by more stable
   Q-estimates.

3. **Target Policy Smoothing**
   Gaussian noise ε ~ N(0, σ) is added to the target action when
   computing Bellman targets, and clipped to [-c, c].  This smooths the
   Q-function and prevents the policy from exploiting narrow Q peaks.

Update Equations
----------------
Target action (with smoothing):
    ã = clip(π_targ(s') + ε, a_low, a_high),  ε ~ clip(N(0,σ), -c, c)

Bellman target (clipped double Q):
    y = r + γ (1 - done) min_i Q_{i,targ}(s', ã)

Critic loss (MSE, each critic independently):
    L(φ_i) = (1/N) Σ (Q_i(s, a) - y)²

Actor loss (maximise Q1 w.r.t. actor parameters):
    L(θ) = -(1/N) Σ Q1(s, π_θ(s))

Soft target update (Polyak averaging):
    φ_i,targ ← τ φ_i + (1-τ) φ_i,targ
    θ_targ   ← τ θ   + (1-τ) θ_targ

Usage
-----
    from algorithms.scratch.td3 import TD3
    from utils.replay_buffer import ReplayBuffer

    agent = TD3(obs_dim=25, act_dim=4, max_action=1.0, device="cuda")
    buf   = ReplayBuffer(obs_dim=25, act_dim=4, capacity=1_000_000)

    # Training loop
    for step in range(total_steps):
        action = agent.select_action(obs, add_noise=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done_bool = terminated and not truncated
        buf.add(obs, action, reward, next_obs, done_bool)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

        if buf.size > start_steps:
            train_info = agent.train(buf, batch_size)
            logger.log_dict(train_info, step)
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from algorithms.scratch.networks import DeterministicActor, TwinCritic
from utils.replay_buffer import ReplayBuffer


class TD3:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the (flat) observation vector.
    act_dim : int
        Dimensionality of the action vector.
    max_action : float
        Absolute upper bound on action values (symmetric).
    device : str
        PyTorch device ("cpu" or "cuda").
    actor_lr : float
        Learning rate for the actor network.
    critic_lr : float
        Learning rate for both critic networks (shared optimiser).
    gamma : float
        Discount factor γ.  Default 0.98 (panda-gym tasks are short).
    tau : float
        Polyak averaging coefficient for soft target updates.
    policy_noise : float
        Standard deviation σ of target policy smoothing noise.
    noise_clip : float
        Clip bound c for target policy smoothing noise.
    policy_delay : int
        Actor update frequency relative to critic updates.
    exploration_noise : float
        Standard deviation of exploration noise added to actions.
    hidden_sizes : tuple
        Hidden layer sizes for actor and critic networks.
    """

    def __init__(
        self,
        obs_dim:           int,
        act_dim:           int,
        max_action:        float = 1.0,
        device:            str   = "cpu",
        actor_lr:          float = 3e-4,
        critic_lr:         float = 3e-4,
        gamma:             float = 0.98,
        tau:               float = 0.005,
        policy_noise:      float = 0.2,
        noise_clip:        float = 0.5,
        policy_delay:      int   = 2,
        exploration_noise: float = 0.1,
        hidden_sizes:      tuple = (256, 256),
    ) -> None:
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.max_action = max_action
        self.device     = device

        # Hyperparameters
        self.gamma             = gamma
        self.tau               = tau
        self.policy_noise      = policy_noise
        self.noise_clip        = noise_clip
        self.policy_delay      = policy_delay
        self.exploration_noise = exploration_noise

        # ── Networks ──────────────────────────────────────────────
        self.actor  = DeterministicActor(obs_dim, act_dim, max_action, hidden_sizes).to(device)
        self.critics = TwinCritic(obs_dim, act_dim, hidden_sizes).to(device)

        # Target networks — deep copies, no gradient
        self.actor_target   = copy.deepcopy(self.actor).to(device)
        self.critics_target = copy.deepcopy(self.critics).to(device)

        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critics_target.parameters():
            param.requires_grad = False

        # ── Optimisers ────────────────────────────────────────────
        self.actor_optim   = torch.optim.Adam(self.actor.parameters(),   lr=actor_lr)
        self.critics_optim = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)

        # Internal counter for delayed policy updates
        self._total_train_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select an action for the given observation.

        Parameters
        ----------
        obs : np.ndarray of shape (obs_dim,)
        add_noise : bool
            If True, adds Gaussian exploration noise (training mode).
            Set False during evaluation.

        Returns
        -------
        np.ndarray of shape (act_dim,) clipped to [-max_action, max_action]
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()

        if add_noise:
            noise   = np.random.normal(0, self.exploration_noise, size=action.shape)
            action  = (action + noise).clip(-self.max_action, self.max_action)

        return action

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """Perform one TD3 gradient update step.

        Parameters
        ----------
        replay_buffer : ReplayBuffer
            Filled replay buffer to sample from.
        batch_size : int

        Returns
        -------
        dict
            Training metrics for logging:
            ``critic_loss``, ``actor_loss`` (None if not updated this step).
        """
        self._total_train_steps += 1
        batch = replay_buffer.sample(batch_size)

        obs       = batch.obs
        actions   = batch.actions
        rewards   = batch.rewards
        next_obs  = batch.next_obs
        dones     = batch.dones

        with torch.no_grad():
            # ── Step 1: Target policy smoothing ───────────────────────
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_obs) + noise
            ).clamp(-self.max_action, self.max_action)

            # ── Step 2: Clipped double Q Bellman target ────────────────
            q1_target, q2_target = self.critics_target(next_obs, next_action)
            q_target = torch.min(q1_target, q2_target)                    # (B, 1)
            y = rewards + self.gamma * (1.0 - dones) * q_target           # (B, 1)

        # ── Step 3: Update both critics ────────────────────────────────
        q1, q2 = self.critics(obs, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        metrics = {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss":  float("nan"),
        }

        # ── Step 4: Delayed actor update ───────────────────────────────
        if self._total_train_steps % self.policy_delay == 0:
            # Actor loss: maximise Q1(s, π(s))  ≡  minimise -Q1(s, π(s))
            actor_loss = -self.critics.q1_forward(obs, self.actor(obs)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # ── Step 5: Soft Polyak target updates ────────────────────
            self._soft_update(self.critics, self.critics_target)
            self._soft_update(self.actor,   self.actor_target)

            metrics["train/actor_loss"] = actor_loss.item()

        return metrics

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all network weights to ``path``."""
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor":          self.actor.state_dict(),
            "critics":        self.critics.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critics_target": self.critics_target.state_dict(),
        }, path)
        print(f"[TD3] Saved checkpoint: {path}")

    def load(self, path: str) -> None:
        """Load network weights from ``path``."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critics.load_state_dict(ckpt["critics"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critics_target.load_state_dict(ckpt["critics_target"])
        print(f"[TD3] Loaded checkpoint: {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
        """Polyak averaging: θ_targ ← τ θ + (1-τ) θ_targ."""
        for p, p_targ in zip(online.parameters(), target.parameters()):
            p_targ.data.mul_(1.0 - self.tau)
            p_targ.data.add_(self.tau * p.data)

    def __repr__(self) -> str:
        return (
            f"TD3(obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"max_action={self.max_action}, device={self.device})"
        )
