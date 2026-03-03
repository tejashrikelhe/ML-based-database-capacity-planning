from __future__ import annotations
import argparse
from pathlib import Path
from src.utils.config import load_config, ensure_dirs
from src.utils.io import read_csv, write_csv
from src.features.windowing import build_window_features
from src.modeling.train import train_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--telemetry", default="data/raw/telemetry.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw = read_csv(args.telemetry)
    feats = build_window_features(raw, cfg["feature_engineering"]["window_minutes"], cfg["feature_engineering"]["min_rows_per_db_window"])
    write_csv(feats, Path(cfg["data"]["processed_dir"]) / "features.csv")

    out = train_all(
        feats,
        cfg["data"]["model_dir"],
        cfg["data"]["reports_dir"],
        cfg["clustering"]["enabled"],
        cfg["clustering"]["n_clusters"],
        cfg["clustering"]["random_state"],
        cfg["modeling"]["quantiles"],
        cfg["modeling"]["test_size_days"],
    )
    print(out)

if __name__ == "__main__":
    main()
