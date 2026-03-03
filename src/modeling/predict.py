from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils.io import load_joblib
from ..features.windowing import build_window_features

def predict_capacity(raw_telemetry: pd.DataFrame, artifacts_path: str, window_minutes: int = 15, min_rows_per_db_window: int = 5) -> pd.DataFrame:
    artifacts = load_joblib(artifacts_path)
    pre = artifacts["preprocessor"]
    km = artifacts.get("kmeans")
    models = artifacts["models"]

    feats = build_window_features(raw_telemetry, window_minutes=window_minutes, min_rows_per_db_window=min_rows_per_db_window)
    drop_cols = ["compute_need_vcpu_p50","compute_need_vcpu_p95","db_id","window_start"]
    X = feats.drop(columns=[c for c in drop_cols if c in feats.columns])
    Xt = pre.transform(X)

    if km is not None:
        clusters = km.predict(Xt)
        Xt = np.hstack([Xt.toarray() if hasattr(Xt,"toarray") else Xt, clusters.reshape(-1,1)])
        feats["cluster"] = clusters
    else:
        Xt = Xt.toarray() if hasattr(Xt,"toarray") else Xt
        feats["cluster"] = -1

    for q, m in models.items():
        col = "pred_vcpu_p50" if abs(q-0.5) < 1e-9 else f"pred_vcpu_p{int(q*100)}"
        feats[col] = m.predict(Xt)

    if "pred_vcpu_p50" in feats.columns and "pred_vcpu_p95" in feats.columns:
        feats["recommended_vcpu"] = feats["pred_vcpu_p50"] + 0.7*(feats["pred_vcpu_p95"] - feats["pred_vcpu_p50"])
    return feats
