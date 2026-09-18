"""
Dual-Arm Cooperative Bimanual Manipulation Environment Wrapper.

Implements multi-agent master-slave impedance coupling and dynamic Control Barrier Functions (CBF)
for collaborative heavy-load transport and mid-air handover without inter-arm collision.
"""

import numpy as np
from typing import Dict, Tuple, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    BaseEnv = gym.Env
except ImportError:
    class BaseEnv:
        pass
    class DummySpace:
        def __init__(self, shape):
            self.shape = shape
    class spaces:
        @staticmethod
        def Box(low, high, shape, dtype=np.float32):
            return DummySpace(shape)


class DualArmCoopEnv(BaseEnv):
    """
    Gymnasium-compliant multi-agent dual robotic arm environment.
    Arm 1 (Left/Master) and Arm 2 (Right/Slave) coordinate to move a shared rigid payload.
    """
    def __init__(self, min_arm_distance: float = 0.25):
        super().__init__()
        self.min_arm_distance = min_arm_distance

        # Action space: 6D action (Arm 1: 3D dx, dy, dz; Arm 2: 3D dx, dy, dz)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation space: 18D state vector
        # [Arm1_pos(3), Arm1_vel(3), Arm2_pos(3), Arm2_vel(3), Payload_pos(3), Target_pos(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # Environment state variables
        self.arm1_pos = np.array([-0.3, 0.0, 0.4], dtype=np.float32)
        self.arm1_vel = np.zeros(3, dtype=np.float32)
        self.arm2_pos = np.array([0.3, 0.0, 0.4], dtype=np.float32)
        self.arm2_vel = np.zeros(3, dtype=np.float32)
        
        self.payload_pos = np.array([0.0, 0.0, 0.4], dtype=np.float32)
        self.target_pos = np.array([0.0, 0.4, 0.5], dtype=np.float32)
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        if hasattr(super(), "reset"):
            try:
                super().reset(seed=seed)
            except Exception:
                pass
        self.arm1_pos = np.array([-0.3, 0.0, 0.4], dtype=np.float32)
        self.arm1_vel = np.zeros(3, dtype=np.float32)
        self.arm2_pos = np.array([0.3, 0.0, 0.4], dtype=np.float32)
        self.arm2_vel = np.zeros(3, dtype=np.float32)
        self.payload_pos = (self.arm1_pos + self.arm2_pos) / 2.0
        self.target_pos = np.array([0.0, 0.35, 0.5], dtype=np.float32) + np.random.uniform(-0.05, 0.05, 3)
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.arm1_pos, self.arm1_vel,
            self.arm2_pos, self.arm2_vel,
            self.payload_pos, self.target_pos
        ]).astype(np.float32)

    def _enforce_inter_arm_cbf(self, proposed_act1: np.ndarray, proposed_act2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Calculates Control Barrier Function (CBF) safety filter for dual-arm collision avoidance:
        h(p1, p2) = ||p1 - p2||^2 - d_min^2 >= 0
        """
        next_p1 = self.arm1_pos + proposed_act1 * 0.05
        next_p2 = self.arm2_pos + proposed_act2 * 0.05

        dist = np.linalg.norm(next_p1 - next_p2)
        cbf_violation = dist < self.min_arm_distance

        safe_act1 = np.copy(proposed_act1)
        safe_act2 = np.copy(proposed_act2)

        if cbf_violation:
            # Apply repulsive vector along inter-arm axis
            direction = (next_p1 - next_p2) / max(dist, 1e-6)
            correction = (self.min_arm_distance - dist) * direction * 2.0
            safe_act1 += correction
            safe_act2 -= correction

        return safe_act1, safe_act2, cbf_violation

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Extract individual actions
        act1 = np.clip(action[:3], -1.0, 1.0)
        act2 = np.clip(action[3:6], -1.0, 1.0)

        # Apply inter-arm CBF collision barrier
        safe_act1, safe_act2, cbf_triggered = self._enforce_inter_arm_cbf(act1, act2)

        # Master-Slave impedance coupling on payload
        # Arm 1 moves according to safe_act1
        self.arm1_vel = safe_act1 * 0.05
        self.arm1_pos += self.arm1_vel

        # Arm 2 follows master arm with impedance compliance to maintain payload distance
        desired_arm2_pos = self.arm1_pos + np.array([0.6, 0.0, 0.0]) # Desired 60cm offset
        impedance_error = desired_arm2_pos - self.arm2_pos
        self.arm2_vel = safe_act2 * 0.05 + impedance_error * 0.2
        self.arm2_pos += self.arm2_vel

        # Shared payload center of mass moves as midpoint of dual arms
        self.payload_pos = (self.arm1_pos + self.arm2_pos) / 2.0

        # Reward formulation
        dist_to_target = float(np.linalg.norm(self.payload_pos - self.target_pos))
        reward = -dist_to_target

        if cbf_triggered:
            reward -= 5.0 # Penalty for entering CBF safety margin zone

        terminated = bool(dist_to_target < 0.05)
        if terminated:
            reward += 100.0

        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()

        info = {
            "dist_to_target": dist_to_target,
            "cbf_triggered": cbf_triggered,
            "inter_arm_distance": float(np.linalg.norm(self.arm1_pos - self.arm2_pos)),
            "payload_pos": self.payload_pos.tolist()
        }

        return obs, reward, terminated, truncated, info
