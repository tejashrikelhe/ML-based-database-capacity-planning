from __future__ import annotations
import pandas as pd
import numpy as np

NUM_COLS = [
    "qps","read_ratio","write_ratio","txn_bytes",
    "p50_latency_ms","p95_latency_ms","errors_per_min","connections",
    "cpu_util","mem_util","io_read_mb_s","io_write_mb_s","cache_hit_ratio",
    "vcpu_allocated","true_required_vcpu"
]
CAT_COLS = ["db_type","region","storage_type"]

def build_window_features(df: pd.DataFrame, window_minutes: int = 15, min_rows_per_db_window: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["db_id","timestamp"])
    df["window_start"] = df["timestamp"].dt.floor(f"{window_minutes}min")

    agg = {c:["mean","max","min","std"] for c in NUM_COLS}
    agg["p95_latency_ms"].append(lambda x: np.quantile(x, 0.95))
    agg["qps"].append(lambda x: np.quantile(x, 0.95))
    agg["cpu_util"].append(lambda x: np.quantile(x, 0.95))

    g = df.groupby(["db_id","window_start"], observed=True)
    feats = g.agg(agg)
    feats.columns = ["__".join([a, b if isinstance(b, str) else "q95"]) for a, b in feats.columns]
    feats = feats.reset_index()

    meta = df.groupby("db_id", observed=True)[CAT_COLS + ["size_gb"]].agg(
        {**{c:(lambda s: s.mode().iloc[0]) for c in CAT_COLS}, "size_gb":"max"}
    ).reset_index()

    out = feats.merge(meta, on="db_id", how="left")
    out["rw_skew"] = out["read_ratio__mean"] - out["write_ratio__mean"]
    out["io_total_mb_s_mean"] = out["io_read_mb_s__mean"] + out["io_write_mb_s__mean"]
    out["latency_tail_ratio"] = out["p95_latency_ms__mean"] / (out["p50_latency_ms__mean"] + 1e-6)

    counts = g.size().reset_index(name="rows_in_window")
    out = out.merge(counts, on=["db_id","window_start"], how="left")
    out = out[out["rows_in_window"] >= min_rows_per_db_window].reset_index(drop=True)

    label = df.groupby(["db_id","window_start"], observed=True)["true_required_vcpu"].agg(
        compute_need_vcpu_p50=lambda x: np.quantile(x, 0.50),
        compute_need_vcpu_p95=lambda x: np.quantile(x, 0.95),
    ).reset_index()
    out = out.merge(label, on=["db_id","window_start"], how="left")
    return out
