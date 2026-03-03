from __future__ import annotations
import pandas as pd
from pathlib import Path
from .drift import drift_report
from ..utils.io import write_csv
from ..modeling.train import train_all

def maybe_retrain(
    df_features: pd.DataFrame,
    model_dir: str,
    reports_dir: str,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.01,
    recent_days: int = 14,
    test_size_days: int = 14,
    clustering_enabled: bool = True,
    n_clusters: int = 6,
    random_state: int = 42
) -> dict:
    df = df_features.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    cutoff = df["window_start"].max() - pd.Timedelta(days=recent_days)
    train_df = df[df["window_start"] <= cutoff].copy()
    recent_df = df[df["window_start"] > cutoff].copy()

    drop = ["compute_need_vcpu_p50","compute_need_vcpu_p95","db_id","window_start","db_type","region","storage_type"]
    numeric_cols = [c for c in df.columns if c not in drop and df[c].dtype != "object"]

    rep = drift_report(train_df, recent_df, numeric_cols)
    write_csv(rep, Path(reports_dir) / "drift_report.csv")

    drifted = (rep["psi"] >= psi_threshold).any() or (rep["ks_pvalue"] <= ks_pvalue_threshold).any()
    if drifted:
        out = train_all(df, model_dir, reports_dir, clustering_enabled, n_clusters, random_state, [0.5,0.95], test_size_days)
        out["retrained"] = True
        return out
    return {"retrained": False, "reason": "No drift thresholds exceeded"}
