"""Test: the benchmark.py CLI entry point runs and reports a win."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent.parent / "benchmark.py"


def test_benchmark_cli_runs_and_reports_savings():
    result = subprocess.run(
        [sys.executable, str(BENCHMARK)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Workload:" in result.stdout
    assert "optimisation wins" in result.stdout
