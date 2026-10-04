"""Generate REPORT.md + figures from real train/eval logs (dev-plot style)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imlfsr.plot_style import (
    FIGSIZE,
    LABEL_SIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    TITLE_SIZE,
    apply_rcparams,
    apply_style,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    hist_path = log_dir / "train_history.json"
    eval_path = log_dir / "eval_results.json"
    sum_path = log_dir / "train_summary.json"

    if not hist_path.is_file():
        print("No train_history.json — skip report figures")
        return

    hist = json.loads(hist_path.read_text())
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hist["epochs"], hist["train_loss"], color=MULTI_SERIES_COLORS[0], lw=LINEWIDTH_MAIN, label="Train loss")
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("Loss", fontsize=LABEL_SIZE)
    ax.set_title("IML-FaceSR training loss", fontsize=TITLE_SIZE)
    ax.legend()
    apply_style(ax)
    fig.savefig(fig_dir / "train_loss.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hist["epochs"], hist["val_psnr"], color=MULTI_SERIES_COLORS[1], lw=LINEWIDTH_MAIN, label="Val PSNR")
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("PSNR (dB)", fontsize=LABEL_SIZE)
    ax.set_title("IML-FaceSR validation PSNR", fontsize=TITLE_SIZE)
    ax.legend()
    apply_style(ax)
    fig.savefig(fig_dir / "val_psnr.svg", bbox_inches="tight")
    plt.close(fig)

    summary = json.loads(sum_path.read_text()) if sum_path.is_file() else {}
    eval_js = json.loads(eval_path.read_text()) if eval_path.is_file() else {}

    def fmt(v, nd=4):
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return "n/a"

    report = f"""# IML Dual-Branch Face SR — Results Report

Paper: Implicit Mutual Learning With Dual-Branch Networks for Face Super-Resolution (TBIOM 2024).

## Training

| Metric | Value |
|--------|------:|
| Protocol | {summary.get('protocol', 'n/a')} |
| Last epoch | {summary.get('epoch', 'n/a')} |
| Train loss | {fmt(summary.get('train_loss'))} |
| Val PSNR | {fmt(summary.get('val_psnr'), 3)} dB |
| Val SSIM | {fmt(summary.get('val_ssim'))} |
| Params (M) | {summary.get('n_params_m', 'n/a')} |

![train loss](outputs/figures/train_loss.svg)

![val psnr](outputs/figures/val_psnr.svg)

## Test evaluation

| Metric | Value |
|--------|------:|
| Split | {eval_js.get('split', 'n/a')} |
| PSNR | {fmt(eval_js.get('psnr'), 3)} dB |
| SSIM | {fmt(eval_js.get('ssim'))} |
| N | {eval_js.get('n', 'n/a')} |

Metrics are from our runs on the paper’s CelebA protocol (real data only).
"""
    Path("REPORT.md").write_text(report)
    print("Wrote REPORT.md and figures")


if __name__ == "__main__":
    main()
