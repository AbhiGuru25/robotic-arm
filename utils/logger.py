"""
utils/logger.py
===============
Unified logging utility combining TensorBoard and CSV output.

The Logger class provides a simple interface for recording scalar metrics
(reward, success rate, actor/critic loss, etc.) to:

1. **TensorBoard** — for real-time visualisation during training.
2. **CSV file** — for post-hoc analysis and plot generation.

Usage
-----
    from utils.logger import Logger

    logger = Logger(
        log_dir="logs/td3_reach_seed0",
        algo="td3",
        task="reach",
        seed=0,
    )

    for step in training_loop():
        logger.log_scalar("train/reward",       reward,       step)
        logger.log_scalar("train/success_rate", success_rate, step)
        logger.log_scalar("train/actor_loss",   actor_loss,   step)
        logger.log_scalar("train/critic_loss",  critic_loss,  step)

    logger.close()
"""

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    print("[Logger] WARNING: TensorBoard not available. CSV logging only.")


class Logger:
    """Dual TensorBoard + CSV logger for RL training metrics.

    Parameters
    ----------
    log_dir : str | Path
        Directory where logs are written.  Created if it does not exist.
    algo : str
        Algorithm name (e.g. "td3", "sac").  Used in CSV header.
    task : str
        Task name (e.g. "reach", "pickandplace").
    seed : int
        Random seed.  Used in CSV header.
    use_tensorboard : bool
        Enable TensorBoard logging.  Default True.
    use_csv : bool
        Enable CSV logging.  Default True.
    """

    def __init__(
        self,
        log_dir: str,
        algo: str = "unknown",
        task: str = "unknown",
        seed: int = 0,
        use_tensorboard: bool = True,
        use_csv: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.algo = algo
        self.task = task
        self.seed = seed
        self._start_time = time.time()

        # TensorBoard writer
        self._tb_writer = None
        if use_tensorboard and _TB_AVAILABLE:
            self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))

        # CSV writer
        self._csv_file    = None
        self._csv_writer  = None
        self._csv_columns: list = []
        if use_csv:
            csv_path = self.log_dir / "metrics.csv"
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")

        print(f"[Logger] Logging to: {self.log_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value.

        Parameters
        ----------
        tag : str
            Metric name, e.g. "train/reward" or "eval/success_rate".
        value : float
            Scalar value to record.
        step : int
            Global environment step (x-axis).
        """
        # TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, global_step=step)

        # CSV
        if self._csv_file is not None:
            row = {
                "step":    step,
                "algo":    self.algo,
                "task":    self.task,
                "seed":    self.seed,
                "tag":     tag,
                "value":   value,
                "elapsed": round(time.time() - self._start_time, 2),
            }
            self._write_csv_row(row)

    def log_dict(self, metrics: Dict[str, Any], step: int) -> None:
        """Log multiple scalar values at once.

        Parameters
        ----------
        metrics : dict
            Mapping of tag -> value.
        step : int
            Global environment step.
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, float(value), step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string to TensorBoard (useful for hyperparameter logging)."""
        if self._tb_writer is not None:
            self._tb_writer.add_text(tag, text, global_step=step)

    def close(self) -> None:
        """Flush and close all writers."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
        if self._csv_file is not None:
            self._csv_file.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, row: Dict[str, Any]) -> None:
        """Lazily initialise CSV writer and write a row."""
        if self._csv_writer is None:
            self._csv_columns = list(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_columns
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
