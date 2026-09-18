"""
Lifelong Learning & Online Model Adaptation Engine with EWC.

Protects robotic policy networks against catastrophic forgetting during continuous deployment
under mechanical degradation, temperature-induced joint friction shifts, and payload drift.
Implements Elastic Weight Consolidation (EWC) using lightweight NumPy / matrix operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

class LifelongAdapter:
    """
    Elastic Weight Consolidation (EWC) and Online Residual Adaptation engine.
    Maintains diagonal Fisher Information Matrix across operational tasks to preserve baseline skills.
    """
    def __init__(
        self,
        weights: np.ndarray,
        ewc_lambda: float = 400.0,
        adaptation_lr: float = 1e-3
    ):
        """
        Args:
            weights: Flattened initial parameter vector or model weight matrix.
            ewc_lambda: Regularization weight balancing target adaptivity vs task retention.
            adaptation_lr: Learning rate for streaming online gradient steps.
        """
        self.weights = np.array(weights, dtype=np.float64)
        self.ewc_lambda = ewc_lambda
        self.adaptation_lr = adaptation_lr
        
        # Optimal baseline parameters: \theta^*
        self.optimal_weights = np.copy(self.weights)
        # Diagonal Fisher Information Matrix entries: F_i
        self.fisher_diag = np.ones_like(self.weights) * 1.0

    def register_baseline_task(self, sample_gradients: np.ndarray):
        """
        Calculates diagonal Fisher Information Matrix over empirical sample gradients of baseline task.

        Fisher = E [ (d/d\theta log p(y|x, \theta))^2 ]
        """
        self.optimal_weights = np.copy(self.weights)
        # Diagonal Fisher estimation via squared sample gradients
        self.fisher_diag = np.mean(np.square(sample_gradients), axis=0) + 1e-5

    def compute_ewc_loss(self) -> float:
        """
        Calculates EWC quadratic loss penalty:
        L_ewc = (\lambda / 2) * \sum_i F_i (\theta_i - \theta_i^*)^2
        """
        diff = self.weights - self.optimal_weights
        ewc_loss = (self.ewc_lambda / 2.0) * np.sum(self.fisher_diag * (diff ** 2))
        return float(ewc_loss)

    def adapt_online_step(
        self,
        current_grad: np.ndarray
    ) -> Dict[str, float]:
        """
        Performs a single online adaptation step using Task Gradient + EWC Penalty Gradient.

        Args:
            current_grad: Task loss gradient wrt model parameters.

        Returns:
            Dictionary containing task loss norm, EWC penalty, and updated weight norm.
        """
        # EWC gradient: \lambda * F_i * (\theta_i - \theta_i^*)
        ewc_grad = self.ewc_lambda * self.fisher_diag * (self.weights - self.optimal_weights)
        
        total_grad = current_grad + ewc_grad
        self.weights -= self.adaptation_lr * total_grad

        ewc_penalty = self.compute_ewc_loss()
        
        return {
            "task_grad_norm": float(np.linalg.norm(current_grad)),
            "ewc_penalty": float(ewc_penalty),
            "weight_norm": float(np.linalg.norm(self.weights))
        }
