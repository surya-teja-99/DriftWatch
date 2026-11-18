"""Tests for DriftWatch core metrics, quality checks and reporting."""

import numpy as np
import pytest

from driftwatch.core import (
    check_categorical_column,
    check_numeric_column,
    compare_datasets,
    ks_statistic,
    missing_rate,
    population_stability_index,
    range_violations,
    unseen_categories,
)
from driftwatch.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANGES,
    generate_current,
    generate_reference,
)
from driftwatch.report import generate_markdown_report


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


def _normal(rng, mean=0.0, sd=1.0, n=5000):
    return rng.normal(mean, sd, n)


# --- PSI ---------------------------------------------------------------------

def test_psi_identical_distributions_is_zero(rng):
    x = _normal(rng)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_no_drift_is_near_zero(rng):
    ref = _normal(rng)
    cur = _normal(rng)  # same distribution, different sample
    assert population_stability_index(ref, cur) < 0.05


def test_psi_strong_drift_is_high(rng):
    ref = _normal(rng)
    cur = _normal(rng, mean=2.0)  # large mean shift
    assert population_stability_index(ref, cur) > 0.25


def test_psi_moderate_drift_between_thresholds(rng):
    ref = _normal(rng)
    cur = _normal(rng, mean=0.5)
    psi = population_stability_index(ref, cur)
    assert 0.05 < psi < 1.0


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError):
        population_stability_index([], [1, 2, 3])


# --- KS ----------------------------------------------------------------------

def test_ks_identical_is_zero(rng):
    x = _normal(rng)
    assert ks_statistic(x, x) == pytest.approx(0.0, abs=1e-9)


def test_ks_no_drift_is_small(rng):
    ref = _normal(rng)
    cur = _normal(rng)
    assert ks_statistic(ref, cur) < 0.05


def test_ks_strong_drift_is_large(rng):
    ref = _normal(rng)
    cur = _normal(rng, mean=2.0)
    assert ks_statistic(ref, cur) > 0.5


# --- Data quality ------------------------------------------------------------

def test_missing_rate_counts_none_and_nan():
    assert missing_rate([1.0, None, float("nan"), 4.0]) == pytest.approx(0.5)
    assert missing_rate(["a", None, "b"]) == pytest.approx(1 / 3)


def test_range_violations():
    result = range_violations([0, 5, 10, 15, 20], low=0, high=10)
    assert result == {"count": 2, "rate": pytest.approx(0.4)}


def test_unseen_categories():
    assert unseen_categories(["a", "b"], ["b", "c", "d"]) == ["c", "d"]
    assert unseen_categories(["a", "b"], ["a", "b"]) == []


def test_numeric_check_flags_drift_and_missing(rng):
    ref = _normal(rng)
    cur = np.append(_normal(rng, mean=2.0), [np.nan, np.nan])
    result = check_numeric_column(ref, cur, expected_range=(-3, 3))
    assert result["verdict"] == "significant"
    assert result["missing_rate_current"] > 0
    assert result["range_violations"]["count"] > 0


def test_categorical_check_finds_unseen():
    ref = ["a", "a", "b", "b"]
    cur = ["a", "b", "c", "c"]
    result = check_categorical_column(ref, cur)
    assert result["unseen_categories"] == ["c"]
    assert result["psi"] > 0


# --- End-to-end comparison ---------------------------------------------------

def test_compare_no_drift_is_stable():
    ref = generate_reference(n=2000, seed=1)
    cur = generate_current(n=2000, seed=2, drift=0.0)
    comparison = compare_datasets(
        ref, cur, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANGES
    )
    assert comparison["overall_verdict"] == "stable"
    assert comparison["drifted_features"] == []


def test_compare_strong_drift_is_significant():
    ref = generate_reference(n=2000, seed=1)
    cur = generate_current(n=2000, seed=2, drift=1.0)
    comparison = compare_datasets(
        ref, cur, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANGES
    )
    assert comparison["overall_verdict"] == "significant"
    assert "monthly_charges" in comparison["drifted_features"]
    assert "tenure_months" in comparison["drifted_features"]


def test_compare_reports_quality_issues():
    ref = generate_reference(n=5000, seed=1)
    cur = generate_current(n=5000, seed=2, drift=0.0, missing_rate=0.1)
    comparison = compare_datasets(
        ref, cur, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANGES
    )
    assert comparison["quality"]["worst_missing_rate"] == pytest.approx(0.1, abs=0.03)


def test_report_generation():
    ref = generate_reference(n=1000, seed=1)
    cur = generate_current(n=1000, seed=2, drift=1.0)
    comparison = compare_datasets(
        ref, cur, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANGES
    )
    text = generate_markdown_report(comparison)
    assert "# DriftWatch Report" in text
    assert "SIGNIFICANT" in text
    assert "monthly_charges" in text
    assert "| PSI |" in text or "PSI" in text
