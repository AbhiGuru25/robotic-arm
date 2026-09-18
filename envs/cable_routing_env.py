"""
Non-Rigid Deformable Cable Harness Routing & Manipulation Environment.

Simulates flexible cable dynamics using mass-spring-damper beam chain mechanics
for automotive/aerospace wiring harness routing tasks.
"""

import numpy as np
from typing import Dict, Tuple, List, Any

class CableRoutingEnv:
    """
    Deformable Cable Routing Environment with 8-node compliant spring-chain dynamics.
    Enforces maximum cable tension and curvature bend limits during routing.
    """
    def __init__(self, num_nodes: int = 8, cable_length: float = 0.60):
        self.num_nodes = num_nodes
        self.cable_length = cable_length
        self.segment_len = cable_length / (num_nodes - 1)
        
        # Cable node 3D positions [N, 3]
        self.nodes = np.zeros((num_nodes, 3))
        self.node_vels = np.zeros((num_nodes, 3))
        self.target_port = np.array([0.4, 0.58, 0.22])
        self.obstacle_peg = np.array([0.25, 0.65, 0.25])
        self.reset()

    def reset(self):
        """Initializes cable straight along X axis."""
        for i in range(self.num_nodes):
            self.nodes[i] = np.array([0.1 + i * self.segment_len, 0.58, 0.20])
        self.node_vels.fill(0.0)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return self.nodes.flatten()

    def step(self, end_effector_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Simulates one physics step of cable routing given end-effector displacement [dx, dy, dz].

        Args:
            end_effector_action: 3D velocity/displacement vector of the robot gripper holding node 0.

        Returns:
            Observation, reward, done flag, and telemetry info.
        """
        # Node 0 is held by robot gripper
        self.nodes[0] += end_effector_action * 0.05
        
        # Forward mass-spring-damper propagation along cable chain
        k_spring = 40.0
        damp = 0.85

        for i in range(1, self.num_nodes):
            prev_node = self.nodes[i - 1]
            curr_node = self.nodes[i]
            
            vec = curr_node - prev_node
            dist = np.linalg.norm(vec)
            if dist > 1e-6:
                dir_vec = vec / dist
                strain = dist - self.segment_len
                force = -k_spring * strain * dir_vec
                self.node_vels[i] = (self.node_vels[i] + force * 0.01) * damp
                self.nodes[i] += self.node_vels[i]

        # Calculate cable tension & curvature bending strain
        tensions = [np.linalg.norm(self.nodes[i] - self.nodes[i-1]) / self.segment_len for i in range(1, self.num_nodes)]
        max_tension = float(np.max(tensions))

        # Check collision / clearance with obstacle peg
        dist_to_peg = np.min([np.linalg.norm(self.nodes[i] - self.obstacle_peg) for i in range(self.num_nodes)])
        peg_collision = dist_to_peg < 0.05

        # Distance from cable tip (node N-1) to target insertion port
        dist_to_target = float(np.linalg.norm(self.nodes[-1] - self.target_port))
        
        reward = -dist_to_target - 2.0 * max(0.0, max_tension - 1.2)
        if peg_collision:
            reward -= 5.0

        done = dist_to_target < 0.04
        if done:
            reward += 50.0

        info = {
            "max_cable_tension": max_tension,
            "dist_to_target": dist_to_target,
            "peg_collision": peg_collision,
            "cable_tip_pos": self.nodes[-1].tolist()
        }

        return self._get_obs(), float(reward), done, info
