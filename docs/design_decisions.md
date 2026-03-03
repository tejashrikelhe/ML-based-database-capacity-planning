# Design Decisions 

## Goal
Predict baseline and peak compute needs for a heterogeneous database fleet while reducing overprovisioning and avoiding slowdowns.

## Why windowed telemetry
Per-minute telemetry is noisy. Aggregating into fixed windows (default 15 minutes) yields stable, comparable features:
- mean/max/std/q95 of workload and resource signals
- tail ratios (p95 latency vs p50 latency)
- simple derived ratios (read/write skew, total IO)

This keeps the pipeline interpretable and production-friendly.

## Why clustering workload patterns
Databases run different workload modes (read-heavy OLTP, write-heavy batch, cache-heavy, IO-bound).
Clustering helps the model learn consistent regimes and improves calibration.

Implementation:
- preprocessed window features -> KMeans
- cluster id appended as a feature (simple, effective)
- production option: train per-cluster models

## Why quantile models (P50/P95)
Capacity planning is risk-based:
- P50: typical demand (baseline allocation)
- P95: high demand (headroom for spikes)

Quantile regression (pinball loss) models these directly without assuming normal errors.

## Why boosted trees
Boosted trees are strong for tabular telemetry:
- fast training/inference
- handle non-linear interactions
- pair well with explainability tooling

This repo uses scikit-learn GradientBoostingRegressor with quantile loss for a lightweight reference build.

## Explainability
Two layers:
- global: permutation importance
- local: SHAP (why this specific DB-window got this prediction)

SHAP is integrated as an optional dependency to keep the base repo lean.

## Drift-aware retraining
Telemetry shifts over time (new releases, migrations, seasonal usage).
We run PSI + KS tests; if thresholds are exceeded, we retrain from latest features.

## What to swap in for production
- LightGBM/XGBoost quantile objectives for scale
- SHAP TreeExplainer (faster and richer) where supported
- model registry + CI checks + automated deployment hooks
