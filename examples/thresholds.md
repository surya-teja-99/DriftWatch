# Drift thresholds

DriftWatch uses these defaults, from `core.py`:

- PSI < 0.10: stable
- PSI 0.10-0.25: moderate
- PSI >= 0.25: significant

KS follows the same verdict ladder, using the largest CDF gap.
When PSI and KS disagree, DriftWatch reports the worse verdict.

Tune thresholds only after checking them against a labelled
backtest window — a threshold that is too sensitive will page
you on normal weekly variation.
