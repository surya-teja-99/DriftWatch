"""DriftWatch — lightweight ML drift & data-quality monitoring for tabular data."""

from .core import (
    check_categorical_column,
    check_numeric_column,
    compare_datasets,
    drift_verdict,
    ks_statistic,
    population_stability_index,
    psi_verdict,
)
from .data import generate_current, generate_reference
from .report import generate_markdown_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "check_categorical_column",
    "check_numeric_column",
    "compare_datasets",
    "drift_verdict",
    "generate_current",
    "generate_markdown_report",
    "generate_reference",
    "ks_statistic",
    "population_stability_index",
    "psi_verdict",
]
