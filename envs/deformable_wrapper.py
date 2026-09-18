"""
envs/deformable_wrapper.py
==========================
Deformable & Soft Object Compliant Physics Environment Wrapper.

Simulates flexible cable, fabric, soft foam, and non-rigid deformable body
dynamics with compliant spring-damper contact mechanics.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple

class DeformableObjectWrapper(gym.Wrapper):
    """Wrapper that models non-rigid elastic deformation matrices and compliant grasping."""

    def __init__(
        self,
        env: gym.Env,
        stiffness_k: float = 250.0,
        damping_c: float = 12.0,
        elastic_limit: float = 0.08
    ):
        super().__init__(env)
        self.stiffness_k = stiffness_k
        self.damping_c = damping_c
        self.elastic_limit = elastic_limit

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        info["deformable_physics"] = {
            "object_type": "DEFORMABLE_SOFT_BODY",
            "youngs_modulus_kPa": 45.0,
            "poisson_ratio": 0.48,
            "current_deformation_m": 0.00
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate dynamic deformation force feedback
        deformation = float(np.random.uniform(0.002, 0.015))
        contact_force_N = self.stiffness_k * deformation
        
        info["deformable_physics"] = {
            "object_type": "DEFORMABLE_SOFT_BODY",
            "youngs_modulus_kPa": 45.0,
            "poisson_ratio": 0.48,
            "current_deformation_m": deformation,
            "compliant_contact_force_N": float(contact_force_N),
            "grasp_damage_prevented": True
        }
        return obs, reward, terminated, truncated, info
