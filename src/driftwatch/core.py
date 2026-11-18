"""Core drift and data-quality calculations for DriftWatch.

Only NumPy and the standard library are used at runtime.

Metrics
-------
PSI (Population Stability Index)
    Compares the binned distribution of a feature in a reference
    population with the same feature in a current population::

        PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Conventional interpretation (used by :func:`psi_verdict`)::

        PSI < 0.10   stable
        0.10 - 0.25  moderate shift — investigate
        PSI >= 0.25  significant drift — act

KS statistic
    Two-sample Kolmogorov–Smirnov statistic: the largest absolute
    difference between the two empirical CDFs. It is 0 for identical
    distributions and 1 for completely separated ones. It needs no
    binning, so it complements PSI on numeric features.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PSI_STABLE = 0.10
PSI_MODERATE = 0.25

KS_STABLE = 0.10
KS_MODERATE = 0.20

MISSING_WARN = 0.05
MISSING_FAIL = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float_array(values: Sequence) -> np.ndarray:
    """Convert *values* to a float array, treating None as NaN."""
    arr = np.asarray(
        [np.nan if v is None else v for v in values], dtype=float
    )
    return arr


def _drop_nan(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def population_stability_index(
    expected: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
) -> float:
    """Return the PSI between a reference (``expected``) and current (``actual``) sample.

    Bin edges are quantile edges of the reference, so each reference bin
    holds ~1/``bins`` of the reference data. Proportions are clipped to
    1e-4 to avoid log(0). Identical distributions return 0.0.
    """
    ref = _drop_nan(_as_float_array(expected))
    cur = _drop_nan(_as_float_array(actual))
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples")

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if edges.size < 3:
        # (Near-)constant feature: fall back to a single split at the median.
        edges = np.array([ref.min(), np.median(ref), ref.max()], dtype=float)
        edges = np.unique(edges)
        if edges.size < 2:
            return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    # np.histogram drops values equal to the last edge from the final bin
    # in older edge cases; count them explicitly for exactness.
    ref_counts[-1] += int(np.sum(ref == edges[-1]))
    cur_counts[-1] += int(np.sum(cur == edges[-1]))

    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-4, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_verdict(psi: float) -> str:
    """Return ``"stable"``, ``"moderate"`` or ``"significant"`` for a PSI value."""
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "significant"


# ---------------------------------------------------------------------------
# KS statistic
# ---------------------------------------------------------------------------

def ks_statistic(expected: Sequence[float], actual: Sequence[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max CDF difference)."""
    ref = np.sort(_drop_nan(_as_float_array(expected)))
    cur = np.sort(_drop_nan(_as_float_array(actual)))
    if ref.size == 0 or cur.size == 0:
        raise ValueError("KS needs non-empty reference and current samples")

    combined = np.sort(np.concatenate([ref, cur]))
    ref_cdf = np.searchsorted(ref, combined, side="right") / ref.size
    cur_cdf = np.searchsorted(cur, combined, side="right") / cur.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def ks_verdict(ks: float) -> str:
    """Return ``"stable"``, ``"moderate"`` or ``"significant"`` for a KS value."""
    if ks < KS_STABLE:
        return "stable"
    if ks < KS_MODERATE:
        return "moderate"
    return "significant"


def drift_verdict(psi: Optional[float], ks: Optional[float] = None) -> str:
    """Overall verdict for one feature: the worst of its PSI / KS verdicts."""
    verdicts = []
    if psi is not None:
        verdicts.append(psi_verdict(psi))
    if ks is not None:
        verdicts.append(ks_verdict(ks))
    if not verdicts:
        return "stable"
    rank = {"stable": 0, "moderate": 1, "significant": 2}
    return max(verdicts, key=rank.get)


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def missing_rate(values: Sequence) -> float:
    """Fraction of values that are missing (None or NaN)."""
    arr = np.asarray([v is None for v in values]) if not _is_numeric(values) else None
    if arr is not None:
        return float(arr.mean()) if len(values) else 0.0
    float_arr = _as_float_array(values)
    return float(np.isnan(float_arr).mean()) if float_arr.size else 0.0


def _is_numeric(values: Sequence) -> bool:
    return all(v is None or isinstance(v, (int, float, np.integer, np.floating)) for v in values)


def range_violations(
    values: Sequence[float],
    low: float,
    high: float,
) -> Dict[str, float]:
    """Count / rate of non-missing values outside ``[low, high]``."""
    arr = _drop_nan(_as_float_array(values))
    if arr.size == 0:
        return {"count": 0, "rate": 0.0}
    count = int(np.sum((arr < low) | (arr > high)))
    return {"count": count, "rate": count / arr.size}


def unseen_categories(
    reference: Sequence[str], current: Sequence[str]
) -> List[str]:
    """Categories present in ``current`` but never seen in ``reference``."""
    ref_set = {str(v) for v in reference if v is not None}
    cur_set = {str(v) for v in current if v is not None}
    return sorted(cur_set - ref_set)


def check_numeric_column(
    reference: Sequence[float],
    current: Sequence[float],
    expected_range: Optional[Tuple[float, float]] = None,
) -> Dict:
    """PSI + KS + quality checks for one numeric column."""
    psi = population_stability_index(reference, current)
    ks = ks_statistic(reference, current)
    result: Dict = {
        "psi": psi,
        "ks": ks,
        "verdict": drift_verdict(psi, ks),
        "missing_rate_current": missing_rate(current),
        "missing_rate_reference": missing_rate(reference),
    }
    if expected_range is not None:
        result["range_violations"] = range_violations(
            current, expected_range[0], expected_range[1]
        )
    return result


def check_categorical_column(
    reference: Sequence[str], current: Sequence[str]
) -> Dict:
    """PSI (over category proportions) + unseen-category check for one column."""
    ref = [str(v) for v in reference if v is not None]
    cur = [str(v) for v in current if v is not None]
    categories = sorted(set(ref) | set(cur))
    # PSI on category proportions, with the same 1e-4 clipping as numeric PSI.
    psi = 0.0
    for cat in categories:
        expected_pct = max(ref.count(cat) / max(len(ref), 1), 1e-4)
        actual_pct = max(cur.count(cat) / max(len(cur), 1), 1e-4)
        psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
    return {
        "psi": float(psi),
        "ks": None,
        "verdict": psi_verdict(psi),
        "missing_rate_current": missing_rate(current),
        "missing_rate_reference": missing_rate(reference),
        "unseen_categories": unseen_categories(reference, current),
    }


# ---------------------------------------------------------------------------
# Whole-dataset comparison
# ---------------------------------------------------------------------------

def compare_datasets(
    reference: Dict[str, Sequence],
    current: Dict[str, Sequence],
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] = (),
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """Compare every listed feature and summarise the overall drift.

    Returns a dict with ``features`` (one entry per feature), an
    ``overall_verdict`` (worst feature verdict), the names of any
    ``drifted_features``, and a ``quality`` summary (worst missing rate,
    total range violations, all unseen categories).
    """
    ranges = ranges or {}
    features: Dict[str, Dict] = {}

    for name in numeric_features:
        features[name] = check_numeric_column(
            reference[name], current[name], ranges.get(name)
        )
    for name in categorical_features:
        features[name] = check_categorical_column(reference[name], current[name])

    rank = {"stable": 0, "moderate": 1, "significant": 2}
    overall = "stable"
    drifted: List[str] = []
    for name, result in features.items():
        if rank[result["verdict"]] > rank[overall]:
            overall = result["verdict"]
        if result["verdict"] != "stable":
            drifted.append(name)

    quality = {
        "worst_missing_rate": max(
            (r["missing_rate_current"] for r in features.values()), default=0.0
        ),
        "total_range_violations": sum(
            r.get("range_violations", {}).get("count", 0)
            for r in features.values()
        ),
        "unseen_categories": sorted(
            {c for r in features.values() for c in r.get("unseen_categories", [])}
        ),
    }

    return {
        "features": features,
        "overall_verdict": overall,
        "drifted_features": drifted,
        "quality": quality,
    }
