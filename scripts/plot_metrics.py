from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.config import load_config, ensure_dirs
from src.utils.io import read_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--metrics", default="reports/metrics.csv")
    ap.add_argument("--preds_test", default="reports/preds_test.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    out_dir = Path(cfg["data"]["reports_dir"]) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    m = read_csv(args.metrics)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar([str(q) for q in m["quantile"]], m["mae"])
    ax.set_title("Test MAE by Quantile")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("MAE (vCPU)")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_quantile.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Pred vs true scatter
    try:
        p = read_csv(args.preds_test)
        q = 0.95 if (p["quantile"] == 0.95).any() else p["quantile"].iloc[0]
        pp = p[p["quantile"] == q]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pp["y_true"], pp["y_pred"], s=6)
        ax.set_title(f"Predicted vs True (q={q})")
        ax.set_xlabel("True vCPU need")
        ax.set_ylabel("Predicted vCPU need")
        plt.tight_layout()
        plt.savefig(out_dir / f"pred_vs_true_q{int(q*100)}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    print(f"Wrote plots to: {out_dir}")

if __name__ == "__main__":
    main()
