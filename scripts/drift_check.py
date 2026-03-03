from __future__ import annotations
import argparse
from src.utils.config import load_config, ensure_dirs
from src.utils.io import read_csv
from src.monitoring.retrain import maybe_retrain

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--maybe_retrain", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    df = read_csv(args.features)
    out = maybe_retrain(
        df,
        cfg["data"]["model_dir"],
        cfg["data"]["reports_dir"],
        psi_threshold=cfg["drift"]["psi_threshold"],
        ks_pvalue_threshold=cfg["drift"]["ks_pvalue_threshold"],
        test_size_days=cfg["modeling"]["test_size_days"],
        clustering_enabled=cfg["clustering"]["enabled"],
        n_clusters=cfg["clustering"]["n_clusters"],
        random_state=cfg["clustering"]["random_state"],
    )
    print(out)

if __name__ == "__main__":
    main()
