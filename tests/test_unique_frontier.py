"""
Unit tests for Unique Frontier Robotic Arm Breakthroughs:
1. Multimodal VLM Vision-Language Reasoning & Task Decomposition
2. Deformable Cable Harness Routing Dynamics
3. Human-Robot Shared Autonomy & ISO 15066 Intent Safety Shield
"""

import numpy as np

from utils.vlm_reasoning_agent import VLMReasoningAgent
from envs.cable_routing_env import CableRoutingEnv
from utils.human_intent_shield import HumanIntentShield


def test_vlm_reasoning_agent():
    agent = VLMReasoningAgent()
    
    # Test case 1: Cable routing prompt
    plan_cable = agent.reason_and_plan("Pick up fragile cable harness and route into assembly port")
    assert plan_cable["parsed_task"] == "Deformable Cable Harness Routing"
    assert plan_cable["total_subgoals"] == 4
    assert plan_cable["vlm_confidence"] >= 0.90

    # Test case 2: Sorting prompt
    plan_sort = agent.reason_and_plan("Sort red items into designated bin")
    assert plan_sort["parsed_task"] == "Zero-Shot Multi-Object Sorting"
    assert plan_sort["total_subgoals"] == 3

    # Test execution anomaly detection
    anomaly = agent.evaluate_execution_anomaly(
        target_pos=np.array([0.1, 0.58, 0.3]),
        current_pos=np.array([0.1, 0.58, 0.3])
    )
    assert not anomaly["has_anomaly"]
    print("[PASS] VLM Reasoning Agent Test")


def test_cable_routing_env():
    env = CableRoutingEnv(num_nodes=8, cable_length=0.60)
    obs = env.reset()
    assert obs.shape == (24,) # 8 nodes * 3D

    # Step action moving gripper node
    action = np.array([0.02, 0.01, 0.0])
    obs, reward, done, info = env.step(action)

    assert "max_cable_tension" in info
    assert "dist_to_target" in info
    assert isinstance(reward, float)
    assert info["max_cable_tension"] > 0.0
    print("[PASS] Cable Routing Environment Test")


def test_human_intent_shield():
    shield = HumanIntentShield(stop_distance=0.15, margin_distance=0.40)
    
    # Test case 1: Human far away -> Full speed
    robot_ee = np.array([0.0, 0.5, 0.3])
    human_far = np.array([1.0, 0.5, 0.3])
    res_far = shield.evaluate_safety_shield(robot_ee, human_far)
    assert res_far["safety_status"] == "FULL_SPEED"
    assert res_far["speed_scale_factor"] == 1.0

    # Test case 2: Human hand approaching -> Speed scaled down
    human_close = np.array([0.25, 0.5, 0.3])
    human_vel = np.array([-0.2, 0.0, 0.0]) # Moving towards robot
    res_close = shield.evaluate_safety_shield(robot_ee, human_close, human_vel)
    assert res_close["speed_scale_factor"] < 1.0
    assert res_close["safety_status"] in ["SPEED_SCALED", "SAFETY_HOLD"]

    # Test action modification
    raw_action = np.array([0.1, 0.2, 0.3, 0.0])
    adj_action = shield.apply_shield_to_action(raw_action, res_close)
    assert np.linalg.norm(adj_action[:3]) < np.linalg.norm(raw_action[:3])
    print("[PASS] Human Intent Safety Shield Test")


if __name__ == "__main__":
    test_vlm_reasoning_agent()
    test_cable_routing_env()
    test_human_intent_shield()
    print("\nSUCCESS: All 3 Unique Frontier Robotic Modules Passed Verification!")
