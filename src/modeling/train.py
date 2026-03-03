from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

from .preprocess import make_preprocessor
from .clustering import fit_kmeans
from .quantile_models import make_quantile_gbr
from ..utils.io import save_joblib, write_csv

def _time_split(df: pd.DataFrame, test_size_days: int = 14):
    df = df.sort_values("window_start")
    cutoff = df["window_start"].max() - pd.Timedelta(days=test_size_days)
    return df[df["window_start"] <= cutoff].copy(), df[df["window_start"] > cutoff].copy()

def train_all(
    df_features: pd.DataFrame,
    model_dir: str,
    reports_dir: str,
    clustering_enabled: bool = True,
    n_clusters: int = 6,
    random_state: int = 42,
    quantiles: list[float] = [0.5, 0.95],
    test_size_days: int = 14,
) -> dict:
    df = df_features.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)

    label_cols = {0.5:"compute_need_vcpu_p50", 0.95:"compute_need_vcpu_p95"}
    drop_cols = ["compute_need_vcpu_p50","compute_need_vcpu_p95","db_id","window_start"]

    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns])
    cat_cols = ["db_type","region","storage_type"]
    num_cols = [c for c in X_all.columns if c not in cat_cols]

    pre = make_preprocessor(cat_cols=cat_cols, num_cols=num_cols)

    train_df, test_df = _time_split(df, test_size_days=test_size_days)
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    Xt_train = pre.fit_transform(X_train)
    Xt_test = pre.transform(X_test)

    artifacts = {"preprocessor": pre}
    if clustering_enabled:
        km = fit_kmeans(Xt_train, n_clusters=n_clusters, random_state=random_state)
        artifacts["kmeans"] = km
        tr_c = km.predict(Xt_train)
        te_c = km.predict(Xt_test)
        Xt_train = np.hstack([Xt_train.toarray() if hasattr(Xt_train,"toarray") else Xt_train, tr_c.reshape(-1,1)])
        Xt_test = np.hstack([Xt_test.toarray() if hasattr(Xt_test,"toarray") else Xt_test, te_c.reshape(-1,1)])
    else:
        Xt_train = Xt_train.toarray() if hasattr(Xt_train,"toarray") else Xt_train
        Xt_test = Xt_test.toarray() if hasattr(Xt_test,"toarray") else Xt_test

    models, metrics = {}, []
    for q in quantiles:
        y_tr = train_df[label_cols[q]].values
        y_te = test_df[label_cols[q]].values
        m = make_quantile_gbr(alpha=q, random_state=random_state)
        m.fit(Xt_train, y_tr)
        pred = m.predict(Xt_test)
        metrics.append({"quantile": q, "mae": float(mean_absolute_error(y_te, pred)), "n_train": len(train_df), "n_test": len(test_df)})
        models[q] = m

    artifacts["models"] = models
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_joblib(artifacts, Path(model_dir) / "artifacts.joblib")
    write_csv(pd.DataFrame(metrics), Path(reports_dir) / "metrics.csv")


    # Save a lightweight test-set prediction file for visualization / debugging
    try:
        out_rows = []
        for q in quantiles:
            y_true = test_df[label_cols[q]].values
            y_pred = models[q].predict(Xt_test)
            tmp = pd.DataFrame({
            "db_id": test_df["db_id"].values,
            "window_start": test_df["window_start"].astype(str).values,
            "quantile": q,
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
            })
            out_rows.append(tmp)
        preds_test = pd.concat(out_rows, ignore_index=True)
        write_csv(preds_test, Path(reports_dir) / "preds_test.csv")
    except Exception:
        pass

    # best-effort global explainability via permutation importances on P95
    try:
        q_imp = 0.95 if 0.95 in models else quantiles[-1]
        perm = permutation_importance(models[q_imp], Xt_test, test_df[label_cols[q_imp]].values, n_repeats=7, random_state=random_state)
        imp = pd.DataFrame({"feature_idx": np.arange(len(perm.importances_mean)),
                            "importance_mean": perm.importances_mean,
                            "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
        write_csv(imp, Path(reports_dir) / "permutation_importance.csv")
    except Exception:
        pass

    return {"metrics": metrics, "model_path": str(Path(model_dir) / "artifacts.joblib")}
