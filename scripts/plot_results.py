"""
scripts/plot_results.py
========================
Generate all final comparison plots and tables for the research paper.

Run after training experiments:
    python scripts/plot_results.py --log_dir logs

Or to generate placeholder plots:
    python scripts/plot_results.py --mock
"""

import argparse
import glob
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from utils.plotting import (
    plot_learning_curves,
    plot_success_rate_bars,
    generate_mock_learning_curves,
    FIGURE_DIR,
    TABLE_DIR,
)

MOCK_BANNER = """
==============================================================
  MOCK DATA — PLACEHOLDER FIGURES
  Replace with real results after Colab training.
==============================================================
"""


def generate_mock_results_table() -> pd.DataFrame:
    data = {
        "Algorithm": [
            "DDPG",
            "TD3",
            "TD3 + HER",
            "SAC",
            "SAC + HER",
            "PPO",
        ],
        "Task": [
            "PickAndPlace",
            "PickAndPlace",
            "PickAndPlace",
            "PickAndPlace",
            "PickAndPlace",
            "PickAndPlace",
        ],
        "HER": [False, False, True, False, True, False],
        "Success Rate (%) [MOCK]": [
            8.0,
            12.0,
            72.0,
            15.0,
            78.0,
            5.0,
        ],
        "Steps to 50% Success [MOCK]": [
            "N/A",
            "N/A",
            "~550k",
            "N/A",
            "~480k",
            "N/A",
        ],
        "Avg Reward [MOCK]": [
            -48.2,
            -46.8,
            -14.2,
            -45.1,
            -12.8,
            -49.5,
        ],
    }
    return pd.DataFrame(data)


def load_real_results(log_dir: pathlib.Path):
    """Load real metrics.csv files from experiment runs."""
    csv_files = glob.glob(str(log_dir / "**" / "metrics.csv"), recursive=True)
    runs = {}
    for csv_file in csv_files:
        p_file = pathlib.Path(csv_file)
        run_name = p_file.parent.name
        try:
            df = pd.read_csv(csv_file)
            runs[run_name] = df
        except Exception as e:
            print(f"[plot] Warning: Failed to read {csv_file}: {e}")
    return runs


def process_real_logs(log_dir: pathlib.Path):
    runs = load_real_results(log_dir)
    if not runs:
        print(f"[plot] No metrics.csv files found in {log_dir}")
        return False

    print(f"[plot] Processing {len(runs)} experiment runs...")

    grouped_data = {}
    final_success = {}

    for run_name, df in runs.items():
        eval_df = df[df["tag"] == "eval/success_rate"].sort_values("step")
        if eval_df.empty:
            continue

        steps = eval_df["step"].values
        values = eval_df["value"].values

        parts = run_name.split("_")
        if "her" in parts:
            algo_tag = f"{parts[0]}_her"
        else:
            algo_tag = parts[0]

        if algo_tag not in grouped_data:
            grouped_data[algo_tag] = []
        grouped_data[algo_tag].append((steps, values))

        if algo_tag not in final_success:
            final_success[algo_tag] = []
        final_success[algo_tag].append(values[-1] if len(values) > 0 else 0.0)

    if grouped_data:
        plot_learning_curves(
            data=grouped_data,
            metric="Success Rate",
            title="PickAndPlace — Learning Curves",
            save_name="learning_curves_real.pdf",
        )

    if final_success:
        avg_final = {k: float(np.mean(v)) for k, v in final_success.items()}
        plot_success_rate_bars(
            results=avg_final,
            task="PandaPickAndPlace-v3",
            save_name="success_rate_bars_real.pdf",
        )
        summary_df = pd.DataFrame(
            [{"Algorithm": k, "Mean Success Rate": v} for k, v in avg_final.items()]
        )
        table_path = TABLE_DIR / "results_summary.csv"
        summary_df.to_csv(table_path, index=False)
        print(f"[plot] Saved summary table: {table_path}")

    return True


def main():
    p = argparse.ArgumentParser(description="Generate result plots and tables.")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Generate placeholder plots with MOCK DATA.",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory containing experiment logs.",
    )
    args = p.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    log_path = ROOT / args.log_dir
    if not args.mock and log_path.exists():
        found = process_real_logs(log_path)
        if found:
            print("[plot] Real plots and tables generated successfully!")
            return

    print(MOCK_BANNER)
    mock_pp = generate_mock_learning_curves("pickandplace")
    plot_learning_curves(
        data=mock_pp,
        metric="Success Rate",
        title="PandaPickAndPlace-v3 — All Algorithms [MOCK DATA]",
        save_name="pickandplace_learning_curves_MOCK.pdf",
    )
    mock_pp_final = {
        "ddpg": 0.08,
        "td3": 0.12,
        "td3_her": 0.72,
        "sac": 0.15,
        "sac_her": 0.78,
        "ppo": 0.05,
    }
    plot_success_rate_bars(
        results=mock_pp_final,
        task="PandaPickAndPlace-v3",
        save_name="success_rate_bars_pickandplace_MOCK.pdf",
    )
    df = generate_mock_results_table()
    table_path = TABLE_DIR / "results_table_MOCK.csv"
    df.to_csv(table_path, index=False)
    print(f"[plot] Saved mock table: {table_path}")
    print("[plot] All plots generated successfully!")


if __name__ == "__main__":
    main()
