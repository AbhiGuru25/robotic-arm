    """
    scripts/run_all_experiments.py
    ==============================
    Batch runner: executes all algorithm x task x HER x seed combinations.

    This script runs every experiment defined in EXPERIMENTS systematically.
    It is designed to be run on Google Colab where each run is ~1-2 hours.

    Usage
    -----
        # Run everything (full research matrix)
        python scripts/run_all_experiments.py

        # Run only reach task experiments
        python scripts/run_all_experiments.py --task reach

        # Run only HER experiments
        python scripts/run_all_experiments.py --her-only

        # Dry run (print commands without executing)
        python scripts/run_all_experiments.py --dry-run

    Experiment Matrix
    -----------------
    Task: PandaReach-v3
        DDPG (SB3)        — no HER
        TD3  (scratch)    — no HER
        SAC  (scratch)    — no HER
        PPO  (SB3)        — no HER

    Task: PandaPickAndPlace-v3
        DDPG (SB3)        — no HER
        TD3  (scratch)    — no HER     [ablation: HER matters]
        TD3  (scratch)    — with HER   [key experiment]
        SAC  (scratch)    — no HER     [ablation: HER matters]
        SAC  (scratch)    — with HER   [key experiment, best expected]
        PPO  (SB3)        — no HER     [on-policy baseline]

    Seeds: 0, 1, 2  (3 seeds minimum per config)
    """

    import argparse
    import subprocess
    import sys
    import pathlib
    import time

    ROOT = pathlib.Path(__file__).parent.parent
    PYTHON = sys.executable


    # ── Full experiment matrix ────────────────────────────────────────────────
    EXPERIMENTS = [
        # ── PandaReach: validate all 4 algorithms ─────────────────────────
        dict(algo="ddpg", task="reach",        her=False, seeds=[0, 1, 2]),
        dict(algo="td3",  task="reach",        her=False, seeds=[0, 1, 2]),
        dict(algo="sac",  task="reach",        her=False, seeds=[0, 1, 2]),
        dict(algo="ppo",  task="reach",        her=False, seeds=[0, 1, 2]),

        # ── PandaPickAndPlace: HER vs no-HER comparison ───────────────────
        dict(algo="ddpg", task="pickandplace", her=False, seeds=[0, 1, 2]),
        dict(algo="td3",  task="pickandplace", her=False, seeds=[0, 1, 2]),   # ablation
        dict(algo="td3",  task="pickandplace", her=True,  seeds=[0, 1, 2]),   # KEY
        dict(algo="sac",  task="pickandplace", her=False, seeds=[0, 1, 2]),   # ablation
        dict(algo="sac",  task="pickandplace", her=True,  seeds=[0, 1, 2]),   # KEY (best)
        dict(algo="ppo",  task="pickandplace", her=False, seeds=[0, 1, 2]),   # on-policy
    ]


    def build_command(algo: str, task: str, her: bool, seed: int) -> list:
        cmd = [
            PYTHON, str(ROOT / "scripts" / "train.py"),
            "--algo", algo,
            "--task", task,
            "--seed", str(seed),
        ]
        if her:
            cmd.append("--her")
        return cmd


    def run_experiment(cmd: list, dry_run: bool = False) -> bool:
        """Run a single training command, return True on success."""
        cmd_str = " ".join(cmd)
        print(f"
{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")
        if dry_run:
            return True
        try:
            t0 = time.time()
            result = subprocess.run(cmd, check=True)
            elapsed = (time.time() - t0) / 60
            print(f"  Done in {elapsed:.1f} min.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  FAILED (exit code {e.returncode})")
            return False


    def main():
        p = argparse.ArgumentParser(description="Batch RL experiment runner.")
        p.add_argument("--task",     type=str, default=None,
                       help="Filter to specific task (reach | pickandplace).")
        p.add_argument("--algo",     type=str, default=None,
                       help="Filter to specific algorithm.")
        p.add_argument("--her-only", action="store_true",
                       help="Only run HER experiments.")
        p.add_argument("--dry-run",  action="store_true",
                       help="Print commands without executing.")
        args = p.parse_args()

        # Filter experiments
        exps = EXPERIMENTS
        if args.task:
            exps = [e for e in exps if e["task"] == args.task]
        if args.algo:
            exps = [e for e in exps if e["algo"] == args.algo]
        if args.her_only:
            exps = [e for e in exps if e["her"]]

        total_runs = sum(len(e["seeds"]) for e in exps)
        print(f"Experiment matrix: {len(exps)} configs x seeds = {total_runs} total runs")
        print(f"Dry run: {args.dry_run}
")

        results = []
        for exp in exps:
            algo, task, her = exp["algo"], exp["task"], exp["her"]
            for seed in exp["seeds"]:
                cmd = build_command(algo, task, her, seed)
                success = run_experiment(cmd, dry_run=args.dry_run)
                results.append({
                    "algo": algo, "task": task,
                    "her": her, "seed": seed,
                    "success": success
                })

        # Summary
        n_success = sum(r["success"] for r in results)
        print(f"
{'=' * 50}")
        print(f"Completed: {n_success}/{len(results)} runs successful.")
        print(f"{'=' * 50}")

        failed = [r for r in results if not r["success"]]
        if failed:
            print("
Failed runs:")
            for r in failed:
                her_tag = "_her" if r["her"] else ""
                print(f"  {r['algo']}{her_tag}_{r['task']}_seed{r['seed']}")


    if __name__ == "__main__":
        main()
