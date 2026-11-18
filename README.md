# DriftWatch

Lightweight ML drift & data-quality monitor for tabular models — no
pandas, no scikit-learn at runtime, just NumPy and the standard library.

DriftWatch compares a **reference** dataset (the data a model was trained
or validated on) with a **current** dataset (what the model is seeing now)
and flags when the world has moved underneath the model. It is a natural
companion to a churn-prediction project: churn behaviour changes after
pricing changes, new plans, or new customer segments, and a silent
distribution shift is usually what degrades a model first.

## What it checks

| Check | What it catches |
| --- | --- |
| **PSI** (Population Stability Index) | Distribution shift per feature |
| **KS statistic** | Largest CDF gap per numeric feature (no binning needed) |
| **Missing rate** | Sudden gaps in a column |
| **Range violations** | Values outside the plausible range, e.g. tenure > 72 months |
| **Unseen categories** | New contract / payment / service values the model never saw |

## PSI / KS, in plain English

**PSI** bins a feature by the reference distribution (deciles by default),
then compares the share of current data in each bin:

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

- **PSI < 0.10** — stable, no action
- **0.10 – 0.25** — moderate shift, investigate
- **PSI ≥ 0.25** — significant drift, act (retrain, re-validate, or roll back)

**KS** is the largest absolute difference between the reference and
current cumulative distributions. It is 0 for identical distributions
and 1 for completely separated ones, and it needs no bins — so it
catches shifts PSI's binning might smooth over. DriftWatch reports the
worst of the two verdicts per feature.

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Generate synthetic churn-like data
driftwatch generate --kind reference --out data/reference.csv
driftwatch generate --kind current --drift 0.6 --missing-rate 0.02 \
    --out data/current.csv

# 2. Compare
driftwatch compare --reference data/reference.csv --current data/current.csv

# 3. Write a Markdown report
driftwatch report --reference data/reference.csv --current data/current.csv \
    --out report.md
```

Or run the demo (no install needed):

```bash
python3 examples/demo.py
```

## Example output

```
Overall verdict: SIGNIFICANT
Feature              PSI      KS  Verdict
------------------------------------------------
tenure_months        0.167   0.171  moderate
monthly_charges      0.657   0.333  significant
total_charges        0.050   0.070  stable
contract_type        0.099   —      stable
payment_method       0.000   —      stable
internet_service     0.001   —      stable

Worst missing rate: 2.3%
Range violations: 0
Unseen categories: none
```

A full report is generated at [`examples/sample_report.md`](examples/sample_report.md).

## Use it from Python

```python
from driftwatch import compare_datasets, generate_current, generate_reference
from driftwatch.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANGES

ref = generate_reference(n=5000, seed=42)
cur = generate_current(n=5000, seed=7, drift=0.6, missing_rate=0.02)

result = compare_datasets(
    ref, cur, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANGES
)
print(result["overall_verdict"])       # 'significant'
print(result["drifted_features"])      # ['tenure_months', 'monthly_charges', ...]
```

## How to run tests

```bash
python3 -m pytest -q
```

Tests cover PSI (identical ≈ 0, no drift < 0.05, strong drift > 0.25),
KS, missing-rate / range / unseen-category checks, end-to-end
comparison on synthetic data, and report generation.

## Project layout

```
src/driftwatch/
  core.py    PSI, KS, verdicts, data-quality checks
  data.py    synthetic churn-like generator (controllable drift)
  report.py  Markdown report generator
  cli.py     `driftwatch generate | compare | report`
tests/test_drift.py
examples/demo.py
.github/workflows/ci.yml
```

## License

MIT — see [LICENSE](LICENSE).
