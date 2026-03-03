from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.config import load_config, ensure_dirs
from src.utils.io import read_csv, load_joblib, write_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--artifacts", default="models/artifacts.joblib")
    ap.add_argument("--max_rows", type=int, default=2000)
    ap.add_argument("--quantile", type=float, default=0.95)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    try:
        import shap
    except Exception:
        raise SystemExit("SHAP is not installed. Run: pip install -r requirements.txt")

    df = read_csv(args.features)
    artifacts = load_joblib(args.artifacts)
    pre = artifacts["preprocessor"]
    km = artifacts.get("kmeans")
    models = artifacts["models"]

    if args.quantile not in models:
        raise SystemExit(f"Quantile {args.quantile} model not found. Available: {list(models.keys())}")

    drop_cols = ["compute_need_vcpu_p50","compute_need_vcpu_p95","db_id","window_start"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    Xt = pre.transform(X)

    if km is not None:
        clusters = km.predict(Xt)
        Xt = np.hstack([Xt.toarray() if hasattr(Xt,'toarray') else Xt, clusters.reshape(-1,1)])
    else:
        Xt = Xt.toarray() if hasattr(Xt,'toarray') else Xt

    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    if len(idx) > args.max_rows:
        idx = rng.choice(idx, size=args.max_rows, replace=False)
    Xt_s = Xt[idx]

    model = models[args.quantile]

    # KernelExplainer for broad compatibility with sklearn GBR (keeps this repo dependency-light).
    background = shap.sample(Xt_s, min(200, len(Xt_s)), random_state=42)
    explainer = shap.KernelExplainer(model.predict, background)

    explain_rows = min(200, len(Xt_s))
    shap_values = explainer.shap_values(Xt_s[:explain_rows], nsamples=200)

    out_dir = Path(cfg["data"]["reports_dir"]) / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(pd.DataFrame(shap_values), out_dir / f"shap_values_q{int(args.quantile*100)}.csv")

    # Plots (best-effort)
    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, Xt_s[:explain_rows], show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_q{int(args.quantile*100)}.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=Xt_s[0]), show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_waterfall_row0_q{int(args.quantile*100)}.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    print(f"Wrote SHAP artifacts to: {out_dir}")

if __name__ == "__main__":
    main()
