"""
algorithms/scratch/sac.py
==========================
SAC (Soft Actor-Critic) — implemented from scratch.

References
----------
Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
 Learning with a Stochastic Actor."  ICML 2018. arXiv:1801.01290.

Haarnoja, T., et al. (2019).
"Soft Actor-Critic Algorithms and Applications."
arXiv:1812.05905.  (Adds automatic temperature tuning.)

Algorithm Overview
------------------
SAC solves the maximum entropy RL objective:

    J(π) = Σ_t E[r(s_t, a_t) + α H(π(·|s_t))]

where H(π(·|s)) = -E_{a~π}[log π(a|s)] is the policy entropy and
α is a temperature parameter balancing exploration vs exploitation.

Key design choices (vs DDPG/TD3):
1. **Stochastic policy**: actions are sampled from a Gaussian, enabling
   principled exploration without manually tuned noise.
2. **Entropy regularisation**: the agent is explicitly incentivised to
   stay uncertain — beneficial in contact-rich manipulation.
3. **Automatic temperature tuning**: α is optimised as a Lagrange
   multiplier to maintain a target entropy H̄ = -|A|.
4. **Twin critics** (borrowed from TD3): reduces overestimation bias.

Update Equations
----------------
Bellman target (soft, with entropy):
    y = r + γ (1-done) [min_i Q_{i,targ}(s', a') - α log π(a'|s')]
    a' ~ π_θ(·|s')

Critic loss:
    L(φ_i) = (1/N) Σ (Q_i(s,a) - y)²

Actor loss (maximise soft Q - entropy):
    L(θ) = (1/N) Σ [α log π_θ(a|s) - min_i Q_i(s, a_θ)]
    a_θ = μ_θ(s) + σ_θ(s) ⊙ ε   (reparameterisation trick)

Temperature (α) loss:
    L(α) = -α (log π_θ(a_t|s_t) + H̄)
    H̄ = -|A|  (target entropy = negative action dimensionality)
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from algorithms.scratch.networks import GaussianActor, TwinCritic
from utils.replay_buffer import ReplayBuffer


class SAC:
    """Soft Actor-Critic (SAC) agent — from scratch.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    max_action : float
        Absolute action bound (symmetric).
    device : str
        PyTorch device.
    actor_lr : float
        Learning rate for the actor network.
    critic_lr : float
        Learning rate for the twin critic networks.
    alpha_lr : float
        Learning rate for the temperature parameter α.
    gamma : float
        Discount factor γ.
    tau : float
        Polyak averaging coefficient for target networks.
    alpha_init : float
        Initial value of temperature α.
    auto_entropy_tuning : bool
        If True, automatically tune α to maintain target entropy H̄.
    target_entropy : float | None
        Target entropy H̄.  If None, defaults to -act_dim.
    hidden_sizes : tuple
        Hidden layer sizes for all networks.
    """

    def __init__(
        self,
        obs_dim:             int,
        act_dim:             int,
        max_action:          float = 1.0,
        device:              str   = "cpu",
        actor_lr:            float = 3e-4,
        critic_lr:           float = 3e-4,
        alpha_lr:            float = 3e-4,
        gamma:               float = 0.98,
        tau:                 float = 0.005,
        alpha_init:          float = 0.2,
        auto_entropy_tuning: bool  = True,
        target_entropy:      float = None,
        hidden_sizes:        tuple = (256, 256),
    ) -> None:
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.max_action = max_action
        self.device     = device
        self.gamma      = gamma
        self.tau        = tau
        self.auto_entropy_tuning = auto_entropy_tuning

        # ── Target entropy (Haarnoja et al. 2019: H_bar = -|A|) ───────
        self.target_entropy = target_entropy if target_entropy is not None                                   else -float(act_dim)

        # ── Temperature α ─────────────────────────────────────────────
        if auto_entropy_tuning:
            # log α is the optimised parameter (ensures α > 0)
            self.log_alpha = torch.tensor(
                np.log(alpha_init), dtype=torch.float32,
                device=device, requires_grad=True
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(alpha_init), device=device)

        # ── Networks ──────────────────────────────────────────────────
        self.actor   = GaussianActor(obs_dim, act_dim, max_action, hidden_sizes).to(device)
        self.critics = TwinCritic(obs_dim, act_dim, hidden_sizes).to(device)

        # Target critics (no gradient)
        self.critics_target = copy.deepcopy(self.critics).to(device)
        for p in self.critics_target.parameters():
            p.requires_grad = False

        # ── Optimisers ────────────────────────────────────────────────
        self.actor_optim   = torch.optim.Adam(self.actor.parameters(),   lr=actor_lr)
        self.critics_optim = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)

    # ------------------------------------------------------------------
    # Property: current α
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature α (always positive via exp)."""
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for the given observation.

        Parameters
        ----------
        obs : np.ndarray (obs_dim,)
        deterministic : bool
            If True, return the mean action (for evaluation).
            If False, sample stochastically (for training).

        Returns
        -------
        np.ndarray (act_dim,)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_action_deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """Perform one SAC gradient update.

        Unlike TD3, SAC updates actor and critics every step
        (no delayed actor updates).

        Parameters
        ----------
        replay_buffer : ReplayBuffer
        batch_size : int

        Returns
        -------
        dict
            Keys: critic_loss, actor_loss, alpha_loss, alpha.
        """
        batch = replay_buffer.sample(batch_size)
        obs       = batch.obs
        actions   = batch.actions
        rewards   = batch.rewards
        next_obs  = batch.next_obs
        dones     = batch.dones

        alpha = self.alpha

        # ── Step 1: Critic update ──────────────────────────────────────
        with torch.no_grad():
            # Sample next action from current policy (+ entropy term)
            next_action, next_log_prob = self.actor.sample(next_obs)

            # Soft Bellman target: min Q - α log π
            q1_t, q2_t = self.critics_target(next_obs, next_action)
            q_target    = torch.min(q1_t, q2_t) - alpha * next_log_prob
            y           = rewards + self.gamma * (1.0 - dones) * q_target

        q1, q2      = self.critics(obs, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # Soft target update for critics
        self._soft_update(self.critics, self.critics_target)

        # ── Step 2: Actor update ───────────────────────────────────────
        # Reparameterised sample from current policy
        pi, log_prob = self.actor.sample(obs)

        q1_pi, q2_pi = self.critics(obs, pi)
        q_pi         = torch.min(q1_pi, q2_pi)

        # Actor loss = α log π - Q  (minimise = maximise soft Q)
        actor_loss = (alpha.detach() * log_prob - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ── Step 3: Temperature α update ───────────────────────────────
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            # L(α) = -α (log π + H_bar)
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        return {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss":  actor_loss.item(),
            "train/alpha_loss":  alpha_loss.item(),
            "train/alpha":       self.alpha.item(),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor":          self.actor.state_dict(),
            "critics":        self.critics.state_dict(),
            "critics_target": self.critics_target.state_dict(),
            "log_alpha":      self.log_alpha,
        }, path)
        print(f"[SAC] Saved checkpoint: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critics.load_state_dict(ckpt["critics"])
        self.critics_target.load_state_dict(ckpt["critics_target"])
        self.log_alpha = ckpt["log_alpha"].to(self.device)
        print(f"[SAC] Loaded checkpoint: {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _soft_update(self, online, target):
        """Polyak averaging: θ_targ ← τ θ + (1-τ) θ_targ."""
        for p, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(1.0 - self.tau)
            p_t.data.add_(self.tau * p.data)

    def __repr__(self) -> str:
        return (
            f"SAC(obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"max_action={self.max_action}, device={self.device}, "
            f"auto_alpha={self.auto_entropy_tuning})"
        )
