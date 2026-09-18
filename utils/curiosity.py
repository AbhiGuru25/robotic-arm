"""
utils/curiosity.py
===================
Random Network Distillation (RND) Intrinsic Curiosity Exploration Module.

Provides self-supervised intrinsic curiosity rewards r_intrinsic = ||f_target(s) - f_predictor(s)||^2
to boost exploration in ultra-sparse long-horizon robotic manipulation tasks alongside HER.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

class RNDCuriosityModule(nn.Module):
    """Random Network Distillation Intrinsic Exploration Module."""

    def __init__(self, obs_dim: int, feature_dim: int = 64, lr: float = 1e-4):
        super().__init__()
        self.obs_dim = obs_dim
        
        # Target network (fixed random weights)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network (trained to match target)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic curiosity reward for given observation batch."""
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)
        intrinsic_reward = torch.mean((target_feat - pred_feat) ** 2, dim=-1)
        return intrinsic_reward

    def update(self, obs: torch.Tensor) -> float:
        """Train predictor network to minimize MSE against fixed target network."""
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)
        loss = nn.functional.mse_loss(pred_feat, target_feat)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
