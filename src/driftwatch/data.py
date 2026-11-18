"""Synthetic churn-like tabular data for DriftWatch.

The reference data mimics a telecom churn table: a mix of numeric and
categorical customer features. ``generate_current`` applies controllable
drift so PSI / KS behaviour can be demonstrated and tested without
shipping any real customer data:

* ``drift=0.0`` — same distribution as the reference (different seed only)
* ``drift=1.0`` — strong drift: higher monthly charges, shorter tenure,
  and a shift towards month-to-month contracts

Missing values can be injected with ``missing_rate``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

NUMERIC_FEATURES: List[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_FEATURES: List[str] = [
    "contract_type",
    "payment_method",
    "internet_service",
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plausible ranges for the numeric features; used by the quality checks.
RANGES: Dict[str, Tuple[float, float]] = {
    "tenure_months": (0.0, 72.0),
    "monthly_charges": (18.0, 120.0),
    "total_charges": (0.0, 9000.0),
}

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer",
    "Credit card",
]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]


def _inject_missing(values: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a fraction ``rate`` of values with NaN (numeric) / None."""
    if rate <= 0:
        return values
    out = values.astype(object) if values.dtype.kind in "OUS" else values.copy()
    mask = rng.random(values.shape[0]) < rate
    if out.dtype.kind == "f":
        out[mask] = np.nan
    else:
        out[mask] = None
    return out


def generate_reference(n: int = 5000, seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate ``n`` reference rows from the baseline churn distribution."""
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.normal(32, 18, n), 0, 72)
    monthly = np.clip(rng.normal(65, 18, n), 18, 120)
    total = np.clip(tenure * monthly * rng.uniform(0.85, 1.05, n), 0, 9000)

    return {
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract_type": rng.choice(
            CONTRACT_TYPES, size=n, p=[0.55, 0.25, 0.20]
        ),
        "payment_method": rng.choice(
            PAYMENT_METHODS, size=n, p=[0.35, 0.25, 0.20, 0.20]
        ),
        "internet_service": rng.choice(
            INTERNET_SERVICES, size=n, p=[0.35, 0.45, 0.20]
        ),
    }


def generate_current(
    n: int = 5000,
    seed: int = 7,
    drift: float = 0.0,
    missing_rate: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Generate ``n`` current rows, with ``drift`` in [0, 1] controlling shift.

    Drift moves monthly charges up (+0 to +25), tenure down (0 to -12
    months) and the contract mix towards month-to-month, i.e. the kind of
    change a churn model would actually see after a pricing change.
    """
    if not 0.0 <= drift <= 1.0:
        raise ValueError("drift must be between 0.0 and 1.0")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be between 0.0 and < 1.0")

    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.normal(32 - 12 * drift, 18, n), 0, 72)
    monthly = np.clip(rng.normal(65 + 25 * drift, 18, n), 18, 120)
    total = np.clip(tenure * monthly * rng.uniform(0.85, 1.05, n), 0, 9000)

    # Contract mix: month-to-month rises from 55% to 80% as drift grows.
    p_month = 0.55 + 0.25 * drift
    p_one = 0.25 - 0.15 * drift
    p_two = 1.0 - p_month - p_one

    data: Dict[str, np.ndarray] = {
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract_type": rng.choice(
            CONTRACT_TYPES, size=n, p=[p_month, p_one, p_two]
        ),
        "payment_method": rng.choice(
            PAYMENT_METHODS, size=n, p=[0.35, 0.25, 0.20, 0.20]
        ),
        "internet_service": rng.choice(
            INTERNET_SERVICES, size=n, p=[0.35, 0.45, 0.20]
        ),
    }

    if missing_rate > 0:
        for name in NUMERIC_FEATURES:
            data[name] = _inject_missing(data[name], missing_rate, rng)
        for name in CATEGORICAL_FEATURES:
            data[name] = _inject_missing(data[name], missing_rate, rng)

    return data


# ---------------------------------------------------------------------------
# CSV round-trip (used by the CLI)
# ---------------------------------------------------------------------------

def save_csv(data: Dict[str, Sequence], path: str | Path) -> Path:
    """Write a feature dict to CSV; NaN / None become empty cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(next(iter(data.values())))
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(data.keys()))
        for i in range(n):
            row = []
            for name in data:
                v = data[name][i]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    row.append("")
                elif isinstance(v, (np.floating, np.integer)):
                    row.append(float(v) if isinstance(v, np.floating) else int(v))
                else:
                    row.append(v)
            writer.writerow(row)
    return path


def load_csv(path: str | Path) -> Dict[str, np.ndarray]:
    """Read a CSV written by :func:`save_csv` back into a feature dict.

    Numeric columns are returned as float arrays (empty cells -> NaN);
    the known categorical columns are returned as object arrays.
    """
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        columns: Dict[str, list] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(row[name])

    out: Dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if name in NUMERIC_FEATURES:
            out[name] = np.array(
                [float(v) if v != "" else np.nan for v in values], dtype=float
            )
        else:
            out[name] = np.array(
                [v if v != "" else None for v in values], dtype=object
            )
    return out
