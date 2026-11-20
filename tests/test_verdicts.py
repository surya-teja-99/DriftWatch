"""Tests for the verdict helpers in driftwatch.core.

These were untested before: psi_verdict / ks_verdict / drift_verdict,
and the empty-input behaviour of missing_rate and range_violations.
"""

from driftwatch.core import (
    drift_verdict,
    ks_verdict,
    missing_rate,
    psi_verdict,
    range_violations,
)


def test_psi_verdict_boundaries():
    assert psi_verdict(0.0) == "stable"
    assert psi_verdict(0.09) == "stable"
    assert psi_verdict(0.10) == "moderate"
    assert psi_verdict(0.24) == "moderate"
    assert psi_verdict(0.25) == "significant"
    assert psi_verdict(1.5) == "significant"


def test_ks_verdict_boundaries():
    assert ks_verdict(0.0) == "stable"
    assert ks_verdict(0.09) == "stable"
    assert ks_verdict(0.10) == "moderate"
    assert ks_verdict(0.19) == "moderate"
    assert ks_verdict(0.20) == "significant"
    assert ks_verdict(0.9) == "significant"


def test_drift_verdict_takes_worst_of_psi_and_ks():
    assert drift_verdict(0.05, 0.05) == "stable"
    assert drift_verdict(0.05, 0.15) == "moderate"
    assert drift_verdict(0.30, 0.05) == "significant"
    assert drift_verdict(0.05, 0.30) == "significant"


def test_drift_verdict_with_one_metric_missing():
    assert drift_verdict(0.30, None) == "significant"
    assert drift_verdict(None, 0.05) == "stable"
    assert drift_verdict(None, None) == "stable"


def test_missing_rate_empty_input_is_zero():
    assert missing_rate([]) == 0.0


def test_range_violations_empty_input_is_zero():
    assert range_violations([], low=0, high=10) == {"count": 0, "rate": 0.0}
