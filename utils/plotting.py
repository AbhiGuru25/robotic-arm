"""
utils/plotting.py
=================
Plotting utilities for generating comparison figures.

Generates:
- Learning curves (reward / success rate vs environment steps)
- Seed-averaged curves with standard deviation bands
- Per-algorithm comparison plots
- HER vs no-HER comparison plots
- Success rate comparison bar charts

All functions save figures to results/figures/.
"""

import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (works on Colab + headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns


# ── Styling ───────────────────────────────────────────────────────────────
PALETTE = {
    "td3":      "#2196F3",   # Blue
    "td3_her":  "#0D47A1",   # Dark blue
    "sac":      "#F44336",   # Red
    "sac_her":  "#B71C1C",   # Dark red
    "ddpg":     "#4CAF50",   # Green
    "ppo":      "#FF9800",   # Orange
}

ALGORITHM_LABELS = {
    "td3":      "TD3",
    "td3_her":  "TD3 + HER",
    "sac":      "SAC",
    "sac_her":  "SAC + HER",
    "ddpg":     "DDPG",
    "ppo":      "PPO",
}

FIGURE_DIR = pathlib.Path("results/figures")
TABLE_DIR  = pathlib.Path("results/tables")


def setup_style():
    """Apply consistent plot style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    plt.rcParams.update({
        "figure.dpi":      150,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "font.family":     "serif",
    })


def load_metrics_csv(csv_path: str, tag: str) -> pd.DataFrame:
    """Load a metrics.csv log and filter to a specific tag.

    Parameters
    ----------
    csv_path : str
        Path to the metrics.csv file from utils/logger.py.
    tag : str
        Metric tag to filter, e.g. "eval/success_rate".

    Returns
    -------
    pd.DataFrame with columns: step, value
    """
    df = pd.read_csv(csv_path)
    return df[df["tag"] == tag][["step", "value"]].reset_index(drop=True)


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.copy(values).astype(float)
    for i in range(1, len(smoothed)):
        smoothed[i] = (window - 1) / window * smoothed[i - 1] +                           1.0 / window * smoothed[i]
    return smoothed


def plot_learning_curves(
    data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    metric: str = "Success Rate",
    title: str = "Learning Curves",
    save_name: str = "learning_curves.pdf",
    smooth_window: int = 5,
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    """Plot seed-averaged learning curves with std bands.

    Parameters
    ----------
    data : dict
        Keys: algorithm name (e.g. "td3", "sac_her").
        Values: list of (steps_array, values_array) per seed.
    metric : str
        Y-axis label.
    title : str
        Plot title.
    save_name : str
        Filename to save under results/figures/.
    smooth_window : int
        EMA smoothing window size.
    figsize : tuple
        Figure size.

    Returns
    -------
    str : Path to saved figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    for algo, seed_runs in data.items():
        color = PALETTE.get(algo, "gray")
        label = ALGORITHM_LABELS.get(algo, algo.upper())

        # Interpolate all seeds onto a common step grid
        if not seed_runs:
            continue

        all_steps = seed_runs[0][0]                           # use first seed steps
        all_values = []
        for steps, values in seed_runs:
            interp_values = np.interp(all_steps, steps, values)
            all_values.append(smooth(interp_values, smooth_window))

        all_values = np.array(all_values)                     # (n_seeds, T)
        mean_vals  = all_values.mean(axis=0)
        std_vals   = all_values.std(axis=0)

        ax.plot(all_steps, mean_vals, label=label, color=color, linewidth=2)
        ax.fill_between(
            all_steps,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim([-0.05, 1.05]) if "rate" in metric.lower() else None
    sns.despine()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(FIGURE_DIR / save_name)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")
    return save_path


def plot_success_rate_bars(
    results: Dict[str, float],
    task: str = "PandaPickAndPlace-v3",
    save_name: str = "success_rate_bar.pdf",
    figsize: Tuple[int, int] = (8, 5),
) -> str:
    """Bar chart of final success rates per algorithm.

    Parameters
    ----------
    results : dict
        Keys: algorithm name, Values: final success rate (0-1).
    task : str
        Task name for plot title.
    save_name : str
        Output filename.

    Returns
    -------
    str : Path to saved figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    algos  = list(results.keys())
    values = [results[a] * 100 for a in algos]
    colors = [PALETTE.get(a, "gray") for a in algos]
    labels = [ALGORITHM_LABELS.get(a, a.upper()) for a in algos]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(f"Final Success Rate — {task}", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 115])
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80% target")
    ax.legend(fontsize=9)
    sns.despine()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(FIGURE_DIR / save_name)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")
    return save_path


def generate_mock_learning_curves(task: str = "pickandplace") -> Dict:
    """Generate clearly-labelled MOCK learning curve data.

    =====================================================================
    MOCK DATA — PLACEHOLDER ONLY
    Replace with real logged data after running training on Colab.
    These values are illustrative estimates based on published benchmarks.
    DO NOT submit these numbers as final results.
    =====================================================================

    Returns
    -------
    dict : {algo: [(steps_array, values_array)]} per seed (1 mock seed)
    """
    steps = np.linspace(0, 1_000_000, 200)

    def sigmoid_curve(inflection, steepness, plateau, noise=0.03):
        """S-curve with random noise to simulate realistic training."""
        curve = plateau / (1 + np.exp(-steepness * (steps - inflection)))
        return np.clip(curve + np.random.normal(0, noise, len(steps)), 0, 1)

    np.random.seed(42)  # reproducible mock

    if task == "pickandplace":
        mock_data = {
            "td3_her":  [(steps, sigmoid_curve(400_000, 1e-5, 0.72))],
            "sac_her":  [(steps, sigmoid_curve(350_000, 1.2e-5, 0.78))],
            "td3":      [(steps, sigmoid_curve(800_000, 5e-6, 0.12))],
            "sac":      [(steps, sigmoid_curve(750_000, 6e-6, 0.15))],
            "ddpg":     [(steps, sigmoid_curve(900_000, 3e-6, 0.08))],
            "ppo":      [(steps, sigmoid_curve(1_200_000, 2e-6, 0.05))],
        }
    else:  # reach
        mock_data = {
            "td3_her":  [(steps[:100], sigmoid_curve(100_000, 3e-5, 0.98)[:100])],
            "sac_her":  [(steps[:100], sigmoid_curve(90_000,  3.5e-5, 0.99)[:100])],
            "td3":      [(steps[:100], sigmoid_curve(150_000, 2e-5, 0.92)[:100])],
            "sac":      [(steps[:100], sigmoid_curve(140_000, 2.5e-5, 0.94)[:100])],
            "ddpg":     [(steps[:100], sigmoid_curve(200_000, 1.5e-5, 0.88)[:100])],
            "ppo":      [(steps[:100], sigmoid_curve(250_000, 1e-5, 0.80)[:100])],
        }

    return mock_data
