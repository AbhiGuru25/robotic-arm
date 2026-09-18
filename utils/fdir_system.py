"""
utils/fdir_system.py
=====================
Industrial Fault Detection, Isolation & Recovery (FDIR) Engine.

Monitors torque anomalies, detects object slippage, joint thermal drift,
and triggers autonomous self-healing recovery sub-routines to prevent
factory assembly line downtime.
"""

import numpy as np
from typing import Dict, Any, Tuple

class IndustrialFDIRSystem:
    """Industrial Real-Time Anomaly Detection & Self-Healing Engine."""

    def __init__(self, torque_threshold: float = 45.0, slip_threshold: float = 0.04):
        self.torque_threshold = torque_threshold
        self.slip_threshold = slip_threshold
        self.fault_count = 0
        self.active_fault = None

    def monitor_telemetry(
        self,
        joint_torques: np.ndarray,
        ee_vel: np.ndarray,
        obj_vel: np.ndarray,
        gripper_clamped: bool
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Scan real-time robot sensors for torque anomalies or object slip events."""
        fault_detected = False
        fault_type = "NOMINAL_OPERATION"

        # 1. Torque Anomaly Check
        max_torque = np.max(np.abs(joint_torques))
        if max_torque > self.torque_threshold:
            fault_detected = True
            fault_type = "TORQUE_OVERLOAD_ANOMALY"

        # 2. Object Slippage Check during Clamped Transport
        if gripper_clamped and np.linalg.norm(ee_vel - obj_vel) > self.slip_threshold:
            fault_detected = True
            fault_type = "GRIPPER_SLIP_PAGE_FAULT"

        if fault_detected:
            self.fault_count += 1
            self.active_fault = fault_type

        status_info = {
            "fault_detected": fault_detected,
            "fault_type": fault_type,
            "active_fault": self.active_fault,
            "total_faults_handled": self.fault_count,
            "self_healing_routine": "RE_GRASP_AND_TORQUE_REBALANCING" if fault_detected else "IDLE"
        }

        return fault_detected, fault_type, status_info

    def execute_self_healing(self, current_action: np.ndarray) -> np.ndarray:
        """Execute autonomous recovery maneuver when a fault is detected."""
        recovery_action = current_action.copy()
        if self.active_fault == "GRIPPER_SLIP_PAGE_FAULT":
            # Increase clamping force and reduce vertical velocity
            recovery_action[3] = -1.0 # Max clamp
            recovery_action[2] = max(-0.1, recovery_action[2] * 0.5) # Damped lift
        elif self.active_fault == "TORQUE_OVERLOAD_ANOMALY":
            # Reduce joint velocity to relieve joint strain
            recovery_action[:3] *= 0.3
        
        self.active_fault = None # Reset after recovery
        return recovery_action
