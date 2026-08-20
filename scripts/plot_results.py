    """
    scripts/plot_results.py
    ========================
    Generate all final comparison plots and tables for the research paper.

    Run after all training experiments are complete:
        python scripts/plot_results.py

    Or to generate placeholder plots immediately (with MOCK DATA label):
        python scripts/plot_results.py --mock

    Output
    ------
    results/figures/
        reach_learning_curves.pdf
        pickandplace_learning_curves.pdf
        her_vs_noher_comparison.pdf
        success_rate_bars_reach.pdf
        success_rate_bars_pickandplace.pdf

    results/tables/
        results_table.csv    — full results (success rate, sample eff., time)
        her_ablation.csv     — HER vs no-HER per algorithm
    """

    import argparse
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
        FIGURE_DIR, TABLE_DIR,
    )

    MOCK_BANNER = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  MOCK DATA — PLACEHOLDER FIGURES                            ║
    ║  Replace with real results after Colab training.            ║
    ║  DO NOT submit mock figures in your final paper.            ║
    ╚══════════════════════════════════════════════════════════════╝
    """


    def generate_mock_results_table() -> pd.DataFrame:
        """
        ==================================================================
        MOCK RESULTS TABLE — PLACEHOLDER ONLY
        Replace all values with real results from your Colab training runs.
        Success rates, sample efficiency, and training times are approximate
        estimates based on published panda-gym benchmarks.
        ==================================================================
        """
        data = {
            "Algorithm": [
                "DDPG",
                "TD3",  "TD3 + HER",
                "SAC",  "SAC + HER",
                "PPO",
            ],
            "Task": [
                "PickAndPlace",
                "PickAndPlace", "PickAndPlace",
                "PickAndPlace", "PickAndPlace",
                "PickAndPlace",
            ],
            "HER": [False, False, True, False, True, False],
            "Success Rate (%) [MOCK]": [
                 8.0,
                12.0,  72.0,
                15.0,  78.0,
                 5.0,
            ],
            "Steps to 50% Success [MOCK]": [
                "N/A",
                "N/A",  "~550k",
                "N/A",  "~480k",
                "N/A",
            ],
            "Avg Reward [MOCK]": [
                -48.2,
                -46.8, -14.2,
                -45.1, -12.8,
                -49.5,
            ],
            "Train Time (hrs, Colab T4) [MOCK]": [
                1.2, 1.4, 1.8,
                1.6, 2.0,
                2.4,
            ],
        }
        return pd.DataFrame(data)


    def load_real_results(log_dir: pathlib.Path):
        """
        Load real experiment results from CSV logs.
        Called when --mock is NOT set.
        """
        import glob
        results = {}
        for csv_file in glob.glob(str(log_dir / "**" / "metrics.csv"), recursive=True):
            parts = pathlib.Path(csv_file).parent.name  # e.g. "td3_her_pickandplace_seed0"
            results[parts] = csv_file
        return results


    def main():
        p = argparse.ArgumentParser(description="Generate result plots and tables.")
        p.add_argument("--mock", action="store_true",
                       help="Generate placeholder plots with MOCK DATA (no training needed).")
        p.add_argument("--log_dir", type=str, default="logs",
                       help="Directory containing experiment logs.")
        args = p.parse_args()

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

        if args.mock:
            print(MOCK_BANNER)

            # ── Mock learning curves: PandaReach ─────────────────────────
            mock_reach = generate_mock_learning_curves("reach")
            plot_learning_curves(
                data      = mock_reach,
                metric    = "Success Rate",
                title     = "PandaReach-v3 — All Algorithms [MOCK DATA]",
                save_name = "reach_learning_curves_MOCK.pdf",
            )

            # ── Mock learning curves: PandaPickAndPlace ───────────────────
            mock_pp = generate_mock_learning_curves("pickandplace")
            plot_learning_curves(
                data      = mock_pp,
                metric    = "Success Rate",
                title     = "PandaPickAndPlace-v3 — All Algorithms [MOCK DATA]",
                save_name = "pickandplace_learning_curves_MOCK.pdf",
            )

            # ── Mock HER vs no-HER comparison ────────────────────────────
            her_comparison = {k: v for k, v in mock_pp.items()
                              if k in ("td3", "td3_her", "sac", "sac_her")}
            plot_learning_curves(
                data      = her_comparison,
                metric    = "Success Rate",
                title     = "HER vs No-HER — PandaPickAndPlace-v3 [MOCK DATA]",
                save_name = "her_vs_noher_MOCK.pdf",
            )

            # ── Mock success rate bars ────────────────────────────────────
            mock_pp_final = {
                "ddpg": 0.08, "td3": 0.12, "td3_her": 0.72,
                "sac":  0.15, "sac_her": 0.78, "ppo": 0.05,
            }
            plot_success_rate_bars(
                results   = mock_pp_final,
                task      = "PandaPickAndPlace-v3",
                save_name = "success_rate_bars_pickandplace_MOCK.pdf",
            )

            # ── Mock results table ────────────────────────────────────────
            df = generate_mock_results_table()
            table_path = TABLE_DIR / "results_table_MOCK.csv"
            df.to_csv(table_path, index=False)
            print(f"[plot] Saved mock table: {table_path}")

            print("
[plot] All MOCK plots generated.")
            print("[plot] Run real training on Colab, then re-run without --mock.")

        else:
            log_dir = ROOT / args.log_dir
            print(f"[plot] Loading real results from: {log_dir}")
            real_results = load_real_results(log_dir)
            if not real_results:
                print("[plot] No log files found. Run training first, or use --mock.")
                sys.exit(1)
            print(f"[plot] Found {len(real_results)} experiment logs.")
            # TODO: parse real CSV logs and call plot_learning_curves()
            # (Implement after first Colab training run)
            print("[plot] Real result plotting: implement after Colab training.")


    if __name__ == "__main__":
        main()
