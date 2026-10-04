"""Evaluate IML Face-SR (PSNR / SSIM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imlfsr.data import build_dataloader
from imlfsr.metrics.psnr_ssim import calc_psnr_ssim
from imlfsr.models import build_model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("device") == "cuda" and cfg.get("cuda_device") is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(cfg['cuda_device'])}")

    model = build_model(cfg).to(device)
    ckpt = args.checkpoint or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    blob = torch.load(ckpt, map_location=device)
    model.load_state_dict(blob["model"])
    model.eval()

    loader = build_dataloader(cfg, args.split)
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            st_lr = batch["st_lr"].to(device)
            sr = model(lr, st_lr)["sr"].clamp(0, 1)
            for i in range(sr.size(0)):
                p, s = calc_psnr_ssim(sr[i], hr[i], crop_border=4)
                psnr_sum += p
                ssim_sum += s
                n += 1
    results = {"split": args.split, "psnr": psnr_sum / max(n, 1), "ssim": ssim_sum / max(n, 1), "n": n, "checkpoint": ckpt}
    out = Path(cfg["paths"]["log_dir"]) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
