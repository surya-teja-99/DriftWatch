"""Command-line interface for DriftWatch.

Usage::

    driftwatch --version
    driftwatch generate --kind reference --out data/reference.csv
    driftwatch generate --kind current --drift 0.6 --out data/current.csv
    driftwatch compare --reference data/reference.csv --current data/current.csv
    driftwatch report --reference data/reference.csv --current data/current.csv \
        --out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core import compare_datasets
from .data import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANGES,
    generate_current,
    generate_reference,
    load_csv,
    save_csv,
)
from .report import generate_markdown_report


def _compare(reference_path: str, current_path: str) -> dict:
    reference = load_csv(reference_path)
    current = load_csv(current_path)
    missing = [f for f in ALL_FEATURES if f not in reference or f not in current]
    if missing:
        raise SystemExit(f"Missing feature(s) in input files: {', '.join(missing)}")
    return compare_datasets(
        reference,
        current,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        ranges=RANGES,
    )


def _print_summary(comparison: dict) -> None:
    print(f"Overall verdict: {comparison['overall_verdict']}")
    print(f"{'Feature':<20} {'PSI':>7} {'KS':>7}  Verdict")
    print("-" * 48)
    for name, result in comparison["features"].items():
        ks = "—" if result["ks"] is None else f"{result['ks']:.3f}"
        print(f"{name:<20} {result['psi']:>7.3f} {ks:>7}  {result['verdict']}")
    q = comparison["quality"]
    print(f"\nWorst missing rate: {q['worst_missing_rate']:.1%}")
    print(f"Range violations: {q['total_range_violations']}")
    print(f"Unseen categories: {q['unseen_categories'] or 'none'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftwatch", description="Lightweight ML drift & data-quality monitor."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic churn-like data.")
    gen.add_argument("--kind", choices=["reference", "current"], required=True)
    gen.add_argument("--out", required=True, help="Output CSV path.")
    gen.add_argument("--n", type=int, default=5000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--drift", type=float, default=0.0,
                     help="Drift strength 0-1 (current only).")
    gen.add_argument("--missing-rate", type=float, default=0.0)

    cmp_ = sub.add_parser("compare", help="Compare two CSVs and print a summary.")
    cmp_.add_argument("--reference", required=True)
    cmp_.add_argument("--current", required=True)
    cmp_.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    cmp_.add_argument("--out", help="Also write a Markdown report to this path.")

    rep = sub.add_parser("report", help="Compare two CSVs and write a Markdown report.")
    rep.add_argument("--reference", required=True)
    rep.add_argument("--current", required=True)
    rep.add_argument("--out", required=True)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        if args.kind == "reference":
            data = generate_reference(n=args.n, seed=args.seed)
        else:
            data = generate_current(
                n=args.n, seed=args.seed,
                drift=args.drift, missing_rate=args.missing_rate,
            )
        save_csv(data, args.out)
        print(f"Wrote {args.n} rows to {args.out}")
        return 0

    comparison = _compare(args.reference, args.current)

    if args.command == "compare":
        if args.json:
            print(json.dumps(comparison, indent=2, default=str))
        else:
            _print_summary(comparison)
        if args.out:
            Path(args.out).write_text(
                generate_markdown_report(
                    comparison, reference_name=args.reference, current_name=args.current
                ),
                encoding="utf-8",
            )
            print(f"Report written to {args.out}")
        return 0

    # report
    text = generate_markdown_report(
        comparison, reference_name=args.reference, current_name=args.current
    )
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"Report written to {args.out}")
    _print_summary(comparison)
    return 0


if __name__ == "__main__":
    sys.exit(main())
