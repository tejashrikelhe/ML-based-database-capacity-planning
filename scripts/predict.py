from __future__ import annotations
import argparse
from src.utils.config import load_config, ensure_dirs
from src.utils.io import read_csv, write_csv
from src.modeling.predict import predict_capacity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--input", default="data/raw/telemetry.csv")
    ap.add_argument("--artifacts", default="models/artifacts.joblib")
    ap.add_argument("--output", default="reports/predictions.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw = read_csv(args.input)
    preds = predict_capacity(raw, args.artifacts, cfg["feature_engineering"]["window_minutes"], cfg["feature_engineering"]["min_rows_per_db_window"])
    write_csv(preds, args.output)
    print(f"Wrote {args.output} rows={len(preds):,}")

if __name__ == "__main__":
    main()
