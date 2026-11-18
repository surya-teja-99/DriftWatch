# DriftWatch usage guide

## Quick check

```bash
driftwatch generate --kind reference --out data/reference.csv
driftwatch generate --kind current --drift 0.6 --out data/current.csv
driftwatch compare --reference data/reference.csv --current data/current.csv
```

## Reading the verdicts

- PSI < 0.10: stable, no action
- PSI 0.10-0.25: moderate shift, investigate the feature
- PSI >= 0.25: significant drift, retrain or roll back

KS is the largest CDF gap, no binning needed. DriftWatch reports
the worst verdict of PSI and KS per feature.

## Tips

- Compare like with like: same columns, same units, same time window.
- A sudden missing-rate jump is usually a pipeline bug, not drift.
- Re-run `driftwatch report` after a fix to confirm the verdict clears.
