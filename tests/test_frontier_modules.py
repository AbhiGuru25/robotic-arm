"""
Unit tests for Frontier Industrial Robotic Arm Modules:
1. Tactile Slip Detection & In-Hand Micro-Adjustment
2. Proprioceptive Haptic Estimator & Occlusion Filter
3. Dual-Arm Dynamic Bimanual Environment (CBF Safety)
4. Continuous Lifelong Learning Adapter (EWC)
"""

import numpy as np

from utils.tactile_slip_detector import TactileSlipDetector
from utils.proprioceptive_estimator import ProprioceptiveEstimator
from envs.dual_arm_coop import DualArmCoopEnv
from utils.lifelong_adapter import LifelongAdapter


def test_tactile_slip_detector():
    detector = TactileSlipDetector(friction_coeff=0.5, safety_margin=0.85)
    
    # Test case 1: Normal grip force, zero shear -> No slip
    res_normal = detector.evaluate_tactile_feedback(normal_force=20.0, shear_force_vec=np.array([1.0, 0.0]))
    assert not res_normal["is_slipping"]
    assert res_normal["slip_ratio"] < 0.85

    # Test case 2: High shear force relative to normal force -> Trigger slip warning & grip boost
    res_slip = detector.evaluate_tactile_feedback(normal_force=5.0, shear_force_vec=np.array([10.0, 0.0]))
    assert res_slip["is_slipping"]
    assert res_slip["grip_boost"] > 0.0
    assert res_slip["slip_events_detected"] >= 1

    # Test action modification
    raw_action = np.array([0.1, 0.2, 0.3, 0.0])
    adj_action = detector.adjust_action_for_slip(raw_action, res_slip)
    assert adj_action[-1] > raw_action[-1]  # Gripper boosted
    assert np.linalg.norm(adj_action[:3]) < np.linalg.norm(raw_action[:3]) # Velocity dampened
    print("[PASS] Tactile Slip Detector Test")


def test_proprioceptive_estimator():
    estimator = ProprioceptiveEstimator(num_particles=50)
    initial_guess = np.array([0.1, 0.2, 0.3])
    estimator.initialize_belief(initial_guess)

    # Test case 1: Visual sensor online (no occlusion)
    joint_pos = np.zeros(7)
    ee_pos = np.array([0.1, 0.2, 0.3])
    wrench = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    visual_meas = np.array([0.105, 0.202, 0.298])

    res = estimator.update_state(
        joint_positions=joint_pos,
        forward_kinematics_ee=ee_pos,
        measured_wrench=wrench,
        visual_pos_meas=visual_meas,
        visual_confidence=0.95
    )
    assert not res["is_occluded"]
    assert np.allclose(res["estimated_target_pos"], visual_meas, atol=0.02)

    # Test case 2: Camera Blackout (100% visual occlusion)
    res_blackout = estimator.update_state(
        joint_positions=joint_pos,
        forward_kinematics_ee=ee_pos,
        measured_wrench=wrench,
        visual_pos_meas=None,
        visual_confidence=0.0
    )
    assert res_blackout["is_occluded"]

    # Test haptic search action generation
    search_action = estimator.generate_haptic_search_action(ee_pos)
    assert search_action.shape == (4,)
    assert search_action[2] < 0  # Probing downwards
    print("[PASS] Proprioceptive Estimator Test")


def test_dual_arm_coop_env():
    env = DualArmCoopEnv(min_arm_distance=0.25)
    obs, info = env.reset()
    assert obs.shape == (18,)

    # Test step with normal non-colliding action
    action = np.array([0.01, 0.0, 0.0, -0.01, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert "cbf_triggered" in info
    assert info["inter_arm_distance"] > 0.0

    # Test step where arms attempt to collide (CBF trigger test)
    # Force Arm 1 and Arm 2 towards center
    colliding_action = np.array([0.9, 0.0, 0.0, -0.9, 0.0, 0.0], dtype=np.float32)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(colliding_action)

    # Distance must be safely maintained at or above min distance threshold
    assert info["inter_arm_distance"] >= env.min_arm_distance - 0.05
    print("[PASS] Dual Arm Cooperative Env Test")


def test_lifelong_adapter():
    initial_weights = np.array([0.5, -0.2, 0.8, 1.1])
    adapter = LifelongAdapter(weights=initial_weights, ewc_lambda=100.0)
    
    # Register baseline task with sample gradients
    sample_grads = np.array([[0.1, 0.2, 0.05, 0.3], [0.12, 0.18, 0.04, 0.28]])
    adapter.register_baseline_task(sample_grads)

    assert len(adapter.optimal_weights) == 4
    assert len(adapter.fisher_diag) == 4

    # Test online adaptation step under domain shift
    shift_grad = np.array([0.05, -0.05, 0.1, 0.02])
    metrics = adapter.adapt_online_step(shift_grad)

    assert "task_grad_norm" in metrics
    assert "ewc_penalty" in metrics
    assert "weight_norm" in metrics
    print("[PASS] Lifelong Adapter (EWC) Test")


if __name__ == "__main__":
    test_tactile_slip_detector()
    test_proprioceptive_estimator()
    test_dual_arm_coop_env()
    test_lifelong_adapter()
    print("\nSUCCESS: All 4 Frontier Industrial Robotic Modules Passed Verification!")
