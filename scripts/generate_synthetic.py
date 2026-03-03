from __future__ import annotations
import argparse
from src.data.synthetic import SyntheticSpec, generate_telemetry
from src.utils.config import load_config, ensure_dirs
from src.utils.io import write_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--n_dbs", type=int, default=120)
    ap.add_argument("--out", default="data/raw/telemetry.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    df = generate_telemetry(SyntheticSpec(days=args.days, n_dbs=args.n_dbs, seed=cfg["project"]["seed"]))
    #write_csv(df.drop(columns=["true_required_vcpu"]), args.out)
    write_csv(df, args.out)
    print(f"Wrote {args.out} rows={len(df):,}")

if __name__ == "__main__":
    main()
