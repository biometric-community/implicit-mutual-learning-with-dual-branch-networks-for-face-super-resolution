"""Save SR predictions for a few test images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imlfsr.data import build_dataloader
from imlfsr.models import build_model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = args.checkpoint or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = build_dataloader(cfg, "test")
    saved = 0
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            st_lr = batch["st_lr"].to(device)
            sr = model(lr, st_lr)["sr"].clamp(0, 1)
            for i in range(sr.size(0)):
                name = batch["name"][i] if isinstance(batch["name"], (list, tuple)) else f"{saved:04d}.png"
                stem = Path(name).stem
                grid = torch.cat(
                    [
                        torch.nn.functional.interpolate(lr[i : i + 1], size=hr.shape[-2:], mode="nearest"),
                        sr[i : i + 1],
                        hr[i : i + 1],
                    ],
                    dim=0,
                )
                vutils.save_image(grid, out_dir / f"{stem}_lr_sr_hr.png", nrow=3)
                saved += 1
                if saved >= args.n:
                    print(f"saved {saved} to {out_dir}")
                    return
    print(f"saved {saved} to {out_dir}")


if __name__ == "__main__":
    main()
