"""Train IML dual-branch face SR (Adam 1e-4, batch 8)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imlfsr.data import build_dataloader
from imlfsr.losses import IMLLoss
from imlfsr.metrics.psnr_ssim import calc_psnr_ssim
from imlfsr.models import build_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        gpu = cfg.get("cuda_device")
        return torch.device(f"cuda:{int(gpu)}" if gpu is not None else "cuda")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
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
    model.train()
    return {"psnr": psnr_sum / max(n, 1), "ssim": ssim_sum / max(n, 1), "n": n}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train IML Face-SR")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    torch.manual_seed(int(cfg["train"].get("seed", 42)))

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(cfg, "train")
    val_split = "val" if cfg["data"].get("dataset") == "celeba" else "test"
    try:
        val_loader = build_dataloader(cfg, val_split)
    except Exception:
        val_loader = build_dataloader(cfg, "test")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"IML-FaceSR params={n_params:.2f}M device={device} n_train={len(train_loader.dataset)}")

    criterion = IMLLoss(
        w_rec=float(cfg["train"].get("w_rec", 1.0)),
        w_st=float(cfg["train"].get("w_st", 1.0)),
        w_edge=float(cfg["train"].get("w_edge", 0.1)),
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=(0.9, 0.99),
    )

    epochs = int(cfg["train"]["epochs"])
    max_steps = args.max_steps if args.max_steps is not None else cfg["train"].get("max_steps")
    log_every = int(cfg["train"].get("log_every", 50))
    history = {"epochs": [], "train_loss": [], "val_psnr": [], "val_ssim": []}
    global_step = 0
    start_epoch = 0
    latest = ckpt_dir / "latest.pt"

    if cfg["train"].get("protocol") == "full" and latest.is_file():
        blob = torch.load(latest, map_location=device)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        start_epoch = int(blob.get("epoch", -1)) + 1
        hist_path = log_dir / "train_history.json"
        if hist_path.is_file():
            history = json.loads(hist_path.read_text())
        print(f"Resumed from {latest} epoch={start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running, count = 0.0, 0
        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            st_lr = batch["st_lr"].to(device)
            st_hr = batch["st_hr"].to(device)
            out = model(lr, st_lr)
            loss, parts = criterion(out, hr, st_hr)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print(f"epoch={epoch} step={global_step} skip non-finite loss")
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1
            running += float(loss.detach())
            count += 1
            if global_step % log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} loss={float(loss.detach()):.4f} "
                    f"rec={parts['rec']:.4f} st={parts['st']:.4f} edge={parts['edge']:.4f}"
                )
            if max_steps is not None and global_step >= int(max_steps):
                break

        train_loss = running / max(count, 1)
        val = validate(model, val_loader, device)
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_psnr"].append(val["psnr"])
        history["val_ssim"].append(val["ssim"])
        (log_dir / "train_history.json").write_text(json.dumps(history, indent=2))

        blob = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "cfg": cfg}
        torch.save(blob, latest)
        torch.save(blob, ckpt_dir / f"epoch{epoch:03d}.pt")
        best = ckpt_dir / "best.pt"
        sum_path = log_dir / "train_summary.json"
        prev = -1.0
        if sum_path.is_file():
            prev = float(json.loads(sum_path.read_text()).get("val_psnr", -1))
        if val["psnr"] >= prev or not best.is_file():
            torch.save(blob, best)

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_psnr": val["psnr"],
            "val_ssim": val["ssim"],
            "protocol": cfg["train"].get("protocol", "subset"),
            "n_params_m": n_params,
            "seconds": time.time() - t0,
        }
        sum_path.write_text(json.dumps(summary, indent=2))
        print(
            f"epoch={epoch} loss={train_loss:.4f} val_psnr={val['psnr']:.3f} "
            f"val_ssim={val['ssim']:.4f} time={summary['seconds']:.1f}s"
        )
        if max_steps is not None and global_step >= int(max_steps):
            print(f"reached max_steps={max_steps}")
            break


if __name__ == "__main__":
    main()
