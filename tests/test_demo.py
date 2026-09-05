"""Tests for the PyCharm-friendly non-training demo."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_a_forward_pass_without_training() -> None:
    completed = subprocess.run(
        [sys.executable, "demo.py", "--batch-size", "3", "--seed", "13"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "inputs shape: (3, 10)" in completed.stdout
    assert "fixed queries: [[1.0], [1.0]]" in completed.stdout
    assert "attention weights shape: (3, 2, 10)" in completed.stdout
    assert "summed attention output shape: (3, 3)" in completed.stdout
    assert "logits shape: (3, 2, 100)" in completed.stdout
    assert "quantization step: 0.125" in completed.stdout
    assert "No optimizer step or training was performed" in completed.stdout
