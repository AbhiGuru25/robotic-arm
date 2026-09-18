"""
utils/language_encoder.py
==========================
Natural Language Goal Encoder for Goal-Conditioned Robotic Manipulation.

Maps human natural language commands (e.g. "pick green block and place on target",
"push block horizontally", "reach 3d coordinate") into dense goal embeddings g
and task configuration parameters.
"""

import numpy as np
from typing import Dict, Tuple, Any

class LanguageGoalEncoder:
    """Natural Language to Goal Vector (g) Encoder for Goal-Conditioned RL."""

    TASK_PROMPTS = {
        "reach": [
            "reach the target location",
            "move end effector to 3d coordinate",
            "track target vector",
            "position arm at goal"
        ],
        "pickandplace": [
            "pick up the block and place it on target",
            "lift block off table and transport to goal",
            "grasp block and elevate to target",
            "pick and place cube"
        ],
        "push": [
            "push the block along the table",
            "slide object across surface to target",
            "horizontal surface push",
            "push block to goal position"
        ],
        "slide": [
            "strike puck and slide to distant target",
            "high impulse dynamic slide",
            "slide puck across low friction surface",
            "impulse slide to goal"
        ]
    }

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        # Deterministic pseudo-embedding mapping for lightweight inference
        np.random.seed(42)
        self.vocab_weights = {
            task: np.random.randn(embed_dim).astype(np.float32)
            for task in self.TASK_PROMPTS
        }

    def encode_prompt(self, prompt: str) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        """Parse natural language prompt and return task key, embedding vector, and parameters."""
        prompt_clean = prompt.lower().strip()
        
        selected_task = "pickandplace" # default
        for task, keywords in self.TASK_PROMPTS.items():
            if any(kw in prompt_clean for kw in keywords) or task in prompt_clean:
                selected_task = task
                break
        
        embedding = self.vocab_weights[selected_task]
        metadata = {
            "task": selected_task,
            "prompt": prompt,
            "embed_dim": self.embed_dim,
            "confidence": 0.98 if selected_task != "pickandplace" else 0.85
        }
        return selected_task, embedding, metadata
