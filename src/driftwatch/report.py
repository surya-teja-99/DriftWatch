"""Markdown report generation for DriftWatch."""

from __future__ import annotations

from datetime import date
from typing import Dict, Optional


def _fmt(value: Optional[float], ndigits: int = 3) -> str:
    return "—" if value is None else f"{value:.{ndigits}f}"


def generate_markdown_report(
    comparison: Dict,
    title: str = "DriftWatch Report",
    reference_name: str = "reference",
    current_name: str = "current",
) -> str:
    """Render a comparison dict (from ``core.compare_datasets``) as Markdown."""
    features = comparison["features"]
    quality = comparison["quality"]

    lines = [
        f"# {title}",
        "",
        f"Date: {date.today().isoformat()}  ",
        f"Reference: `{reference_name}` — Current: `{current_name}`",
        "",
        f"**Overall verdict: {comparison['overall_verdict'].upper()}**",
        "",
    ]

    if comparison["drifted_features"]:
        lines.append(
            "Drifted features: "
            + ", ".join(f"`{name}`" for name in comparison["drifted_features"])
        )
        lines.append("")
    else:
        lines.append("No feature crossed a drift threshold.")
        lines.append("")

    lines += [
        "## Per-feature drift",
        "",
        "| Feature | PSI | KS | Verdict | Missing (current) |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for name, result in features.items():
        lines.append(
            f"| {name} | {_fmt(result['psi'])} | {_fmt(result['ks'])} "
            f"| {result['verdict']} | {_fmt(result['missing_rate_current'], 3)} |"
        )

    lines += [
        "",
        "## Data quality",
        "",
        f"- Worst missing rate (current): {quality['worst_missing_rate']:.1%}",
        f"- Total range violations: {quality['total_range_violations']}",
    ]
    unseen = quality["unseen_categories"]
    if unseen:
        lines.append("- Unseen categories: " + ", ".join(f"`{c}`" for c in unseen))
    else:
        lines.append("- Unseen categories: none")

    lines += [
        "",
        "## How to read this",
        "",
        "PSI < 0.10 is stable, 0.10–0.25 is a moderate shift worth "
        "investigating, and ≥ 0.25 is significant drift. KS is the largest "
        "gap between the reference and current CDFs (0 = identical, "
        "1 = completely separated).",
        "",
    ]
    return "\n".join(lines)


def write_report(comparison: Dict, path: str, **kwargs) -> str:
    """Write the Markdown report to *path* and return its text."""
    text = generate_markdown_report(comparison, **kwargs)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
