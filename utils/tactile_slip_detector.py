"""
Tactile Slip Detector and In-Hand Micro-Adjustment Controller.

Provides high-frequency tactile force ratio monitoring and micro-grip adjustments
to prevent object drops under unpredictable payload friction and external vibration.
"""

import numpy as np
from typing import Dict, Tuple, Any

class TactileSlipDetector:
    """
    Tactile slip detection and micro-adjustment system using friction cone bounds.
    Monitors shear-to-normal force ratio and computes instant force boost.
    """
    def __init__(
        self,
        friction_coeff: float = 0.5,
        safety_margin: float = 0.85,
        max_grip_force: float = 50.0,
        response_gain: float = 12.0
    ):
        """
        Args:
            friction_coeff: Estimated static friction coefficient (mu) between finger and object.
            safety_margin: Threshold ratio (F_shear / (mu * F_normal)) triggering slip mitigation.
            max_grip_force: Maximum allowable normal force (N).
            response_gain: Proportional gain for grip force boost.
        """
        self.mu = friction_coeff
        self.safety_margin = safety_margin
        self.max_grip_force = max_grip_force
        self.response_gain = response_gain
        self.current_grip_force = 10.0  # Initial normal force in Newtons
        self.slip_event_count = 0

    def evaluate_tactile_feedback(
        self,
        normal_force: float,
        shear_force_vec: np.ndarray,
        object_mass: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluates current 3D tactile sensor wrench and determines slip state.

        Args:
            normal_force: Measured normal force F_n (N).
            shear_force_vec: 2D shear force vector [F_x, F_y] (N).
            object_mass: Estimated mass of object (kg).

        Returns:
            Dictionary containing slip status, force ratio, and recommended grip adjustment.
        """
        F_n = max(float(normal_force), 1e-3)
        F_s = float(np.linalg.norm(shear_force_vec))

        # Friction utilization ratio (lambda)
        # lambda = F_s / (mu * F_n)
        slip_ratio = F_s / (self.mu * F_n)
        is_slipping = slip_ratio >= self.safety_margin

        grip_boost = 0.0
        if is_slipping:
            self.slip_event_count += 1
            # Compute required additional normal force to bring slip_ratio back to 0.70
            target_F_n = F_s / (self.mu * 0.70)
            grip_boost = min(self.response_gain * (target_F_n - F_n), self.max_grip_force - F_n)
            self.current_grip_force = float(np.clip(F_n + grip_boost, 0.0, self.max_grip_force))

        return {
            "is_slipping": bool(is_slipping),
            "slip_ratio": float(slip_ratio),
            "shear_force_magnitude": float(F_s),
            "normal_force": float(F_n),
            "recommended_grip_force": float(self.current_grip_force),
            "grip_boost": float(grip_boost),
            "slip_events_detected": int(self.slip_event_count)
        }

    def adjust_action_for_slip(
        self,
        action: np.ndarray,
        tactile_status: Dict[str, Any]
    ) -> np.ndarray:
        """
        Modifies arm joint/cartesion control action to incorporate grip boost and dampen tangential velocity.

        Args:
            action: Original 4D or 7D action array (e.g. [dx, dy, dz, grip_action]).
            tactile_status: Result from evaluate_tactile_feedback.

        Returns:
            Adjusted action array.
        """
        adjusted_action = np.copy(action)
        if tactile_status["is_slipping"]:
            # Boost gripper action index (assuming last element is gripper control)
            adjusted_action[-1] = np.clip(adjusted_action[-1] + 0.5, -1.0, 1.0)
            # Dampen spatial translational velocity to reduce inertia during slip recovery
            adjusted_action[:3] *= 0.4
        return adjusted_action
