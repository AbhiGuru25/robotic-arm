"""
scripts/run_all_experiments.py
==============================
Batch runner: executes all algorithm x task x HER x seed combinations.

Usage
-----
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --task reach
    python scripts/run_all_experiments.py --her-only
    python scripts/run_all_experiments.py --dry-run
"""

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
PYTHON = sys.executable

EXPERIMENTS = [
    dict(algo="ddpg", task="reach", her=False, seeds=[0, 1, 2]),
    dict(algo="td3", task="reach", her=False, seeds=[0, 1, 2]),
    dict(algo="sac", task="reach", her=False, seeds=[0, 1, 2]),
    dict(algo="ppo", task="reach", her=False, seeds=[0, 1, 2]),
    dict(algo="ddpg", task="pickandplace", her=False, seeds=[0, 1, 2]),
    dict(algo="td3", task="pickandplace", her=False, seeds=[0, 1, 2]),
    dict(algo="td3", task="pickandplace", her=True, seeds=[0, 1, 2]),
    dict(algo="sac", task="pickandplace", her=False, seeds=[0, 1, 2]),
    dict(algo="sac", task="pickandplace", her=True, seeds=[0, 1, 2]),
    dict(algo="ppo", task="pickandplace", her=False, seeds=[0, 1, 2]),
]


def build_command(algo: str, task: str, her: bool, seed: int) -> list:
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "train.py"),
        "--algo",
        algo,
        "--task",
        task,
        "--seed",
        str(seed),
    ]
    if her:
        cmd.append("--her")
    return cmd


def run_experiment(cmd: list, dry_run: bool = False) -> bool:
    cmd_str = " ".join(cmd)
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Running: {cmd_str}")
    if dry_run:
        return True
    try:
        t0 = time.time()
        subprocess.run(cmd, check=True)
        elapsed = (time.time() - t0) / 60
        print(f"  Done in {elapsed:.1f} min.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED (exit code {e.returncode})")
        return False


def main():
    p = argparse.ArgumentParser(description="Batch RL experiment runner.")
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter to specific task (reach | pickandplace).",
    )
    p.add_argument(
        "--algo", type=str, default=None, help="Filter to specific algorithm."
    )
    p.add_argument(
        "--her-only", action="store_true", help="Only run HER experiments."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = p.parse_args()

    exps = EXPERIMENTS
    if args.task:
        exps = [e for e in exps if e["task"] == args.task]
    if args.algo:
        exps = [e for e in exps if e["algo"] == args.algo]
    if args.her_only:
        exps = [e for e in exps if e["her"]]

    total_runs = sum(len(e["seeds"]) for e in exps)
    print(f"Experiment matrix: {len(exps)} configs x seeds = {total_runs} total runs")
    print(f"Dry run: {args.dry_run}\n")

    results = []
    for exp in exps:
        algo, task, her = exp["algo"], exp["task"], exp["her"]
        for seed in exp["seeds"]:
            cmd = build_command(algo, task, her, seed)
            success = run_experiment(cmd, dry_run=args.dry_run)
            results.append(
                {
                    "algo": algo,
                    "task": task,
                    "her": her,
                    "seed": seed,
                    "success": success,
                }
            )

    n_success = sum(r["success"] for r in results)
    print(f"\n{'=' * 50}")
    print(f"Completed: {n_success}/{len(results)} runs successful.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
