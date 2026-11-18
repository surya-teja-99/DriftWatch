# DriftWatch Report

Date: 2026-11-18  
Reference: `reference (seed 42)` — Current: `current (drift 0.6, seed 7)`

**Overall verdict: SIGNIFICANT**

Drifted features: `tenure_months`, `monthly_charges`

## Per-feature drift

| Feature | PSI | KS | Verdict | Missing (current) |
| --- | ---: | ---: | --- | ---: |
| tenure_months | 0.167 | 0.171 | moderate | 0.022 |
| monthly_charges | 0.657 | 0.333 | significant | 0.020 |
| total_charges | 0.050 | 0.070 | stable | 0.023 |
| contract_type | 0.099 | — | stable | 0.018 |
| payment_method | 0.000 | — | stable | 0.018 |
| internet_service | 0.001 | — | stable | 0.021 |

## Data quality

- Worst missing rate (current): 2.3%
- Total range violations: 0
- Unseen categories: none

## How to read this

PSI < 0.10 is stable, 0.10–0.25 is a moderate shift worth investigating, and ≥ 0.25 is significant drift. KS is the largest gap between the reference and current CDFs (0 = identical, 1 = completely separated).
