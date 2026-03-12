"""
Compatibility runner for the full analysis pipeline.

This script matches the README entrypoint and orchestrates the actual
step scripts that live in data_analysis/steps/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS_DIR = ROOT / "data_analysis" / "steps"


def run_step(script: str, *extra_args: str) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "funding-rate-analysis-mpl"))

    cmd = [sys.executable, script, *extra_args]
    print(f"\n==> Running {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=STEPS_DIR, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the funding-rate analysis pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "test"],
        default="full",
        help="Pipeline size: full for paper-grade outputs, quick/test for validation",
    )
    args = parser.parse_args()

    print(f"Pipeline root: {ROOT}")
    print(f"Mode: {args.mode}")

    run_step("01_data_preprocessing.py")
    run_step("03_calibration.py")
    run_step("05_experiments.py", "--mode", args.mode)
    run_step("06_visualization.py")

    print("\nPipeline complete. Outputs are in data_analysis/steps/results/.")


if __name__ == "__main__":
    main()
