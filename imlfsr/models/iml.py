"""Implicit Mutual Learning dual-branch FSR network (Fig. 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BasicBlock, DFRM, ExchangeBlock, HFNet


@dataclass
class IMLConfig:
    n_feats: int = 48
    n_dfrm: int = 2
    scale: int = 8


class EncoderStage(nn.Module):
    def __init__(self, channels: int, down: bool = True):
        super().__init__()
        self.block = BasicBlock(channels)
        self.down = nn.Conv2d(channels, channels, 4, 2, 1) if down else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x)
        return self.down(h), h


class DecoderStage(nn.Module):
    def __init__(self, channels: int, up: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, 4, 2, 1) if up else nn.Identity()
        self.block = BasicBlock(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.fuse(torch.cat([x, skip], dim=1)))


class DualBranch(nn.Module):
    """Four-stage encoder–decoder with DFRM bottlenecks and IEB hooks."""

    def __init__(self, in_ch: int, channels: int, n_dfrm: int):
        super().__init__()
        self.head = nn.Conv2d(in_ch, channels, 3, 1, 1)
        self.enc = nn.ModuleList([EncoderStage(channels, down=(i < 3)) for i in range(4)])
        self.dfrms = nn.Sequential(*[DFRM(channels) for _ in range(n_dfrm)])
        self.dec = nn.ModuleList([DecoderStage(channels, up=(i < 3)) for i in range(4)])

    def encode(self, x: torch.Tensor):
        x = self.head(x)
        skips = []
        for stage in self.enc:
            x, skip = stage(x)
            skips.append(skip)
        x = self.dfrms(x)
        return x, skips

    def decode(self, x: torch.Tensor, skips):
        for stage, skip in zip(self.dec, reversed(skips)):
            x = stage(x, skip)
        return x


class IMLFaceSR(nn.Module):
    """
    Texture + structure dual-branch network with FIEB/SIEB and HFNet.

    Returns dict with keys: sr, sr_mid, sr_struct.
    """

    def __init__(self, cfg: IMLConfig | dict | None = None):
        super().__init__()
        if cfg is None:
            cfg = IMLConfig()
        elif isinstance(cfg, dict):
            cfg = IMLConfig(
                n_feats=int(cfg.get("n_feats", 48)),
                n_dfrm=int(cfg.get("n_dfrm", 2)),
                scale=int(cfg.get("scale", 8)),
            )
        self.cfg = cfg
        c = cfg.n_feats
        self.texture = DualBranch(3, c, cfg.n_dfrm)
        self.structure = DualBranch(3, c, cfg.n_dfrm)
        self.fieb = ExchangeBlock(c)  # texture → structure (encoder)
        self.sieb = ExchangeBlock(c)  # structure → texture (decoder)
        self.hfnet = HFNet(c)

    def forward(self, lr: torch.Tensor, st_lr: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Bicubic upsample to HR canvas as in Algorithm 1
        hr_h = lr.shape[-2] * self.cfg.scale
        hr_w = lr.shape[-1] * self.cfg.scale
        lr_up = F.interpolate(lr, size=(hr_h, hr_w), mode="bicubic", align_corners=False)
        st_up = F.interpolate(st_lr, size=(hr_h, hr_w), mode="bicubic", align_corners=False)

        f_te, skips_te = self.texture.encode(lr_up)
        f_st, skips_st = self.structure.encode(st_up)
        # FIEB at bottleneck: enrich structure with texture
        f_st = self.fieb(f_te, f_st)
        y_te = self.texture.decode(f_te, skips_te)
        y_st = self.structure.decode(f_st, skips_st)
        # SIEB after decode: enrich texture with structure
        y_te = self.sieb(y_st, y_te)

        sr, sr_mid, sr_struct = self.hfnet(y_te, y_st, lr_up)
        return {"sr": sr, "sr_mid": sr_mid, "sr_struct": sr_struct}


def build_model(cfg: dict) -> IMLFaceSR:
    return IMLFaceSR(cfg.get("model", cfg))
