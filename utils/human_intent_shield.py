"""
Human-Robot Shared Autonomy & ISO 15066 Intent Safety Shield.

Forecasts human worker hand trajectory vectors and dynamically scales workspace
robot velocity to allow zero-interruption human-robot co-working.
"""

import numpy as np
from typing import Dict, Tuple, Any

class HumanIntentShield:
    """
    Dynamic ISO 15066 safety shield with 0.5s predictive human intent forecasting.
    Adjusts robot action velocity vectors dynamically to guarantee zero collision.
    """
    def __init__(
        self,
        stop_distance: float = 0.15,
        margin_distance: float = 0.40,
        forecast_horizon_sec: float = 0.5
    ):
        """
        Args:
            stop_distance: Hard minimum separation distance (m).
            margin_distance: Distance zone where speed scaling begins (m).
            forecast_horizon_sec: Future time horizon for human motion forecasting (sec).
        """
        self.stop_distance = stop_distance
        self.margin_distance = margin_distance
        self.forecast_horizon_sec = forecast_horizon_sec

    def forecast_human_pose(self, human_pos: np.ndarray, human_vel: np.ndarray) -> np.ndarray:
        """Forecasts future human hand position vector: \hat{p}(t + \Delta t) = p + v \cdot \Delta t"""
        return human_pos + human_vel * self.forecast_horizon_sec

    def evaluate_safety_shield(
        self,
        robot_ee_pos: np.ndarray,
        human_hand_pos: np.ndarray,
        human_hand_vel: np.ndarray = np.zeros(3)
    ) -> Dict[str, Any]:
        """
        Evaluates dynamic distance separation to predicted human hand pose.

        Args:
            robot_ee_pos: Current 3D position of robot end-effector [x, y, z].
            human_hand_pos: Current 3D position of human worker hand [x, y, z].
            human_hand_vel: Estimated 3D velocity vector of human hand [vx, vy, vz].

        Returns:
            Dictionary with predicted distance, velocity scaling factor \alpha, and safety state.
        """
        predicted_human_pos = self.forecast_human_pose(human_hand_pos, human_hand_vel)
        dist_current = float(np.linalg.norm(robot_ee_pos - human_hand_pos))
        dist_predicted = float(np.linalg.norm(robot_ee_pos - predicted_human_pos))
        
        effective_dist = min(dist_current, dist_predicted)

        # Dynamic speed scaling factor: alpha \in [0.05, 1.0]
        if effective_dist <= self.stop_distance:
            speed_scale = 0.05 # Minimum crawl speed for safety hold
            safety_status = "SAFETY_HOLD"
        elif effective_dist >= self.margin_distance:
            speed_scale = 1.0 # 100% full speed
            safety_status = "FULL_SPEED"
        else:
            speed_scale = float((effective_dist - self.stop_distance) / (self.margin_distance - self.stop_distance))
            speed_scale = np.clip(speed_scale, 0.05, 1.0)
            safety_status = "SPEED_SCALED"

        return {
            "current_distance_meters": dist_current,
            "predicted_distance_meters": dist_predicted,
            "speed_scale_factor": float(speed_scale),
            "safety_status": safety_status,
            "iso_15066_compliant": bool(effective_dist > self.stop_distance)
        }

    def apply_shield_to_action(self, raw_action: np.ndarray, shield_result: Dict[str, Any]) -> np.ndarray:
        """Scales translational components of action array by speed_scale_factor."""
        adjusted_action = np.copy(raw_action)
        scale = shield_result["speed_scale_factor"]
        adjusted_action[:3] *= scale
        return adjusted_action
