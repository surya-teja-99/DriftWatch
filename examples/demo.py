"""Runnable DriftWatch demo.

Generates a synthetic reference set and a drifted current set
(pricing-change scenario, drift=0.6), runs the drift check, prints a
summary and writes ``examples/sample_report.md``.

Run from the project root::

    python3 examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the demo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftwatch.core import compare_datasets
from driftwatch.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANGES,
    generate_current,
    generate_reference,
)
from driftwatch.report import write_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    print("Generating reference data (n=5000)...")
    reference = generate_reference(n=5000, seed=42)

    print("Generating current data with drift=0.6, missing_rate=0.02 ...")
    current = generate_current(n=5000, seed=7, drift=0.6, missing_rate=0.02)

    comparison = compare_datasets(
        reference,
        current,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        ranges=RANGES,
    )

    print(f"\nOverall verdict: {comparison['overall_verdict'].upper()}")
    print(f"{'Feature':<20} {'PSI':>7} {'KS':>7}  Verdict")
    print("-" * 48)
    for name, result in comparison["features"].items():
        ks = "—" if result["ks"] is None else f"{result['ks']:.3f}"
        print(f"{name:<20} {result['psi']:>7.3f} {ks:>7}  {result['verdict']}")

    quality = comparison["quality"]
    print(f"\nWorst missing rate: {quality['worst_missing_rate']:.1%}")
    print(f"Range violations: {quality['total_range_violations']}")
    print(f"Unseen categories: {quality['unseen_categories'] or 'none'}")

    out = root / "examples" / "sample_report.md"
    write_report(
        comparison, str(out),
        reference_name="reference (seed 42)",
        current_name="current (drift 0.6, seed 7)",
    )
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
