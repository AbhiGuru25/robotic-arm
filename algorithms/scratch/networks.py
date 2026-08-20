"""
algorithms/scratch/networks.py
================================
Shared neural network architectures for TD3 and SAC.

All networks use multi-layer perceptrons (MLPs) with ReLU activations.
The output activations differ by role:

- Deterministic Actor (TD3): tanh output scaled to action bounds.
- Gaussian Actor (SAC): outputs mean + log_std of a Gaussian; actions
  sampled via the reparameterisation trick and squashed through tanh.
- Critic: scalar Q-value output, no final activation.

Design Notes
------------
- Networks accept flattened numpy-style observations (float tensors).
- Action scaling assumes symmetric bounds [-max_action, +max_action].
- Weight initialisation follows OpenAI Spinning Up convention
  (uniform, last layer scaled to ±3e-3 for stability).

References
----------
- Fujimoto et al. (2018): TD3 architecture details.
- Haarnoja et al. (2018, 2019): SAC architecture, log_std clamping.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# SAC log_std clamping bounds (Haarnoja et al. 2019)
LOG_STD_MAX =  2.0
LOG_STD_MIN = -20.0


def _mlp(layer_sizes: list, activation=nn.ReLU) -> nn.Sequential:
    """Build a Multi-Layer Perceptron.

    Parameters
    ----------
    layer_sizes : list of int
        Sizes including input and output, e.g. [obs_dim, 256, 256, act_dim].
    activation : nn.Module class
        Activation applied between layers (not after the last layer).

    Returns
    -------
    nn.Sequential
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:          # no activation after last layer
            layers.append(activation())
    return nn.Sequential(*layers)


# -------------------------------------------------------------------------
# Deterministic Actor  (used by TD3)
# -------------------------------------------------------------------------

class DeterministicActor(nn.Module):
    """Deterministic policy network for TD3.

    Maps state → action ∈ [-max_action, max_action] via tanh squashing.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    max_action : float
        Absolute upper bound on actions (assumed symmetric).
    hidden_sizes : tuple
        Hidden layer sizes.  Default (256, 256) as in Fujimoto et al.
    """

    def __init__(
        self,
        obs_dim:      int,
        act_dim:      int,
        max_action:   float = 1.0,
        hidden_sizes: tuple = (256, 256),
    ) -> None:
        super().__init__()
        self.max_action = max_action
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.net = _mlp(sizes)
        # Initialise last layer to small weights for training stability
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias,   -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor of shape (B, obs_dim)

        Returns
        -------
        Tensor of shape (B, act_dim) in [-max_action, max_action]
        """
        return self.max_action * torch.tanh(self.net(obs))


# -------------------------------------------------------------------------
# Critic  (used by both TD3 and SAC)
# -------------------------------------------------------------------------

class Critic(nn.Module):
    """Q-value network: Q(s, a) → scalar.

    Takes the concatenation of observation and action as input.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    hidden_sizes : tuple
    """

    def __init__(
        self,
        obs_dim:      int,
        act_dim:      int,
        hidden_sizes: tuple = (256, 256),
    ) -> None:
        super().__init__()
        sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        self.net = _mlp(sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs    : Tensor (B, obs_dim)
        action : Tensor (B, act_dim)

        Returns
        -------
        Tensor (B, 1) — Q-value
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# -------------------------------------------------------------------------
# Twin Critic  (used by TD3 — two independent Q networks sharing an interface)
# -------------------------------------------------------------------------

class TwinCritic(nn.Module):
    """Two independent critic networks for TD3 clipped double Q-learning.

    Instead of wrapping two Critic instances, we define two MLPs in one
    Module so they share a single optimiser.

    Parameters
    ----------
    obs_dim, act_dim, hidden_sizes : same as Critic.
    """

    def __init__(
        self,
        obs_dim:      int,
        act_dim:      int,
        hidden_sizes: tuple = (256, 256),
    ) -> None:
        super().__init__()
        in_dim = obs_dim + act_dim
        sizes  = [in_dim] + list(hidden_sizes) + [1]
        self.q1 = _mlp(sizes)
        self.q2 = _mlp(sizes)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q1 and Q2 values.

        Returns
        -------
        (q1_value, q2_value) — each of shape (B, 1)
        """
        x  = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 — used in the actor update (TD3 uses Q1 only)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


# -------------------------------------------------------------------------
# Gaussian Actor  (used by SAC)
# -------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """Stochastic policy network for SAC.

    Outputs the mean and log_std of a Gaussian distribution over actions.
    Actions are sampled via the reparameterisation trick and squashed
    through tanh, with the log probability corrected accordingly.

    The log probability of a tanh-squashed action is:
        log π(a|s) = log N(u; μ, σ) − Σ_i log(1 − tanh²(u_i))
    where u is the pre-squash sample.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    max_action : float
    hidden_sizes : tuple
    """

    def __init__(
        self,
        obs_dim:      int,
        act_dim:      int,
        max_action:   float = 1.0,
        hidden_sizes: tuple = (256, 256),
    ) -> None:
        super().__init__()
        self.max_action = max_action
        sizes = [obs_dim] + list(hidden_sizes)
        self.trunk = _mlp(sizes)

        # Separate heads for mean and log_std
        self.mean_head    = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)

        # Small last-layer init for stable start
        nn.init.uniform_(self.mean_head.weight,    -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias,      -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias,   -3e-3, 3e-3)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and log_std for the action distribution.

        Returns
        -------
        mean    : Tensor (B, act_dim)
        log_std : Tensor (B, act_dim)  clamped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        h       = F.relu(self.trunk(obs))
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action via reparameterisation + tanh squashing.

        Returns
        -------
        action  : Tensor (B, act_dim)  squashed and scaled to max_action
        log_prob: Tensor (B, 1)        log π(a|s) corrected for tanh
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterisation: u = mean + std * eps,  eps ~ N(0, I)
        dist = Normal(mean, std)
        u    = dist.rsample()                     # (B, act_dim)

        # Squash + scale
        a    = torch.tanh(u) * self.max_action

        # Log probability with tanh correction
        # log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        # Numerical stability: log(1 - tanh²(u)) = log(1 - tanh(u)²)
        log_prob -= (2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))).sum(
            dim=-1, keepdim=True
        )
        return a, log_prob

    def get_action_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation (use mean, no sampling)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.max_action
