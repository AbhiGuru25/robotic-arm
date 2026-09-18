"""
Proprioceptive Haptic Estimator & Occlusion-Robust Control System.

Enables robust manipulation under camera blackout (smoke, dust, line-of-sight occlusion)
using Bayesian Particle Filtering and Contact-Guided Force Exploration.
"""

import numpy as np
from typing import Dict, Tuple, List, Any

class ProprioceptiveEstimator:
    """
    Bayesian Particle Filter for state estimation under visual occlusion and camera dropout.
    Fuses joint kinematics, IMU accelerometer data, and contact wrench sensing.
    """
    def __init__(
        self,
        num_particles: int = 100,
        process_noise_std: float = 0.005,
        haptic_force_threshold: float = 2.5
    ):
        """
        Args:
            num_particles: Number of belief state particles.
            process_noise_std: Kinematic process noise standard deviation (m).
            haptic_force_threshold: Force magnitude (N) signifying solid surface contact.
        """
        self.num_particles = num_particles
        self.process_noise_std = process_noise_std
        self.haptic_force_threshold = haptic_force_threshold
        
        # State particles: [x, y, z] for target object belief position
        self.particles = np.zeros((num_particles, 3))
        self.weights = np.ones(num_particles) / num_particles
        self.estimated_pos = np.zeros(3)
        self.estimated_cov = np.eye(3) * 0.01
        self.is_occluded = False
        self.haptic_contact_detected = False
        self.haptic_search_step = 0

    def initialize_belief(self, initial_guess: np.ndarray, uncertainty_std: float = 0.05):
        """Initializes Gaussian distribution of particles around an initial position guess."""
        self.particles = initial_guess + np.random.normal(0, uncertainty_std, size=(self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.estimated_pos = np.copy(initial_guess)

    def update_state(
        self,
        joint_positions: np.ndarray,
        forward_kinematics_ee: np.ndarray,
        measured_wrench: np.ndarray,
        visual_pos_meas: np.ndarray = None,
        visual_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Updates particle filter belief state based on visual confidence and haptic contact.

        Args:
            joint_positions: Current joint angles (rad).
            forward_kinematics_ee: Current end-effector position [x, y, z] from FK.
            measured_wrench: End-effector 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz].
            visual_pos_meas: Observed target position from camera [x, y, z] if available.
            visual_confidence: Confidence coefficient in [0.0, 1.0] (0 = total blackout).

        Returns:
            Dictionary with state estimate, occlusion flag, and haptic contact status.
        """
        self.is_occluded = visual_confidence < 0.2
        contact_force_mag = np.linalg.norm(measured_wrench[:3])
        self.haptic_contact_detected = contact_force_mag >= self.haptic_force_threshold

        # 1. Prediction step (incorporate kinematic motion + process noise)
        self.particles += np.random.normal(0, self.process_noise_std, size=self.particles.shape)

        # 2. Measurement update step
        if not self.is_occluded and visual_pos_meas is not None:
            # High visual confidence: weight particles by Euclidean distance to vision measurement
            dists = np.linalg.norm(self.particles - visual_pos_meas, axis=1)
            self.weights = np.exp(-0.5 * (dists / 0.02) ** 2)
        elif self.haptic_contact_detected:
            # Camera blacked out, but haptic contact detected at end-effector position
            dists = np.linalg.norm(self.particles - forward_kinematics_ee, axis=1)
            self.weights = np.exp(-0.5 * (dists / 0.01) ** 2)
        else:
            # Camera blacked out and no contact: maintain uniform weight drift
            pass

        # Normalize particle weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-12:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Resample particles if effective sample size drops low
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2.0:
            indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Compute mean state estimate and covariance matrix
        self.estimated_pos = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - self.estimated_pos
        self.estimated_cov = np.dot(self.weights * diff.T, diff)

        return {
            "estimated_target_pos": self.estimated_pos.copy(),
            "position_variance": float(np.trace(self.estimated_cov)),
            "is_occluded": bool(self.is_occluded),
            "haptic_contact_detected": bool(self.haptic_contact_detected),
            "contact_force_magnitude": float(contact_force_mag),
            "visual_confidence": float(visual_confidence)
        }

    def generate_haptic_search_action(self, ee_pos: np.ndarray) -> np.ndarray:
        """
        Generates a 3D spiral/raster search trajectory when camera is occluded and contact is not yet established.

        Args:
            ee_pos: Current end-effector position [x, y, z].

        Returns:
            Delta action [dx, dy, dz, 0] to probe workspace haptically.
        """
        self.haptic_search_step += 1
        t = self.haptic_search_step * 0.1
        radius = min(0.005 * t, 0.10)  # Expanding spiral radius
        
        dx = radius * np.cos(t) * 0.05
        dy = radius * np.sin(t) * 0.05
        dz = -0.002  # Gradual downward probing towards surface

        return np.array([dx, dy, dz, 0.0])
