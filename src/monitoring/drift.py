from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    e_counts, _ = np.histogram(expected, bins=qs)
    a_counts, _ = np.histogram(actual, bins=qs)
    e = np.clip(e_counts / max(e_counts.sum(), 1), eps, None)
    a = np.clip(a_counts / max(a_counts.sum(), 1), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))

def drift_report(train_df: pd.DataFrame, recent_df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric_cols:
        e = train_df[c].to_numpy(dtype=float)
        a = recent_df[c].to_numpy(dtype=float)
        ks = ks_2samp(e[~np.isnan(e)], a[~np.isnan(a)])
        rows.append({"feature": c, "psi": psi(e, a), "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)})
    return pd.DataFrame(rows).sort_values(["psi","ks_stat"], ascending=False)
