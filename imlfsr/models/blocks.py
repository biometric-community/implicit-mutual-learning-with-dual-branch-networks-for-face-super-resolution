"""Building blocks: BasicBlock, DFRM (DSRB/DRB/TCA), FIEB/SIEB, SAM/HFNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class BasicBlock(nn.Module):
    """Eq. (1): f = frb + Conv(Fca(Fsa(frb)))."""

    def __init__(self, channels: int):
        super().__init__()
        self.rb = ResidualBlock(channels)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(channels)
        self.out = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frb = self.rb(x)
        return frb + self.out(self.ca(self.sa(frb)))


class DSRB(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DRB(nn.Module):
    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class TCA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.sa(self.ca(x)))


class DFRM(nn.Module):
    """Deep Feature Reinforcement Module: DSRB → DRB → TCA."""

    def __init__(self, channels: int, n_dsrb: int = 2, n_drb: int = 2):
        super().__init__()
        self.dsrb = nn.Sequential(*[DSRB(channels) for _ in range(n_dsrb)])
        self.drb = nn.Sequential(*[DRB(channels, dilation=2) for _ in range(n_drb)])
        self.tca = TCA(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tca(self.drb(self.dsrb(x)))


class ExchangeBlock(nn.Module):
    """Shared FIEB/SIEB: concat → 1×1/3×3 → CA/SA → gated multi-branch FFN."""

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.b3 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.b5 = nn.Conv2d(channels * 2, channels, 5, 1, 2)
        self.gate = nn.Conv2d(channels * 2, channels * 2, 1)
        self.out = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Enrich ``dst`` using ``src`` context; return updated dst."""
        x = self.fuse(torch.cat([src, dst], dim=1))
        x = self.sa(self.ca(x))
        exp = F.gelu(self.expand(x))
        y = torch.cat([self.b3(exp), self.b5(exp)], dim=1)
        y = y * torch.sigmoid(self.gate(y))  # simple gate (SG)
        return dst + self.out(y)


class SAM(nn.Module):
    """Supervised Attention Module (HFNet) — merges branch feat with LR guidance."""

    def __init__(self, channels: int):
        super().__init__()
        self.feat = nn.Conv2d(channels, channels, 3, 1, 1)
        self.img = nn.Conv2d(3, channels, 3, 1, 1)
        self.att = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.out = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, feat: torch.Tensor, lr_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.feat(feat) + self.img(lr_up)
        a = self.att(f)
        img = torch.sigmoid(self.out(f * a))
        return img, a


class HFNet(nn.Module):
    """Hybrid Fusion Network: SAM on each branch + adaptive merge."""

    def __init__(self, channels: int):
        super().__init__()
        self.sam_te = SAM(channels)
        self.sam_st = SAM(channels)
        self.merge = nn.Sequential(
            nn.Conv2d(6, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.struct_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, f_te: torch.Tensor, f_st: torch.Tensor, lr_up: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_m, _ = self.sam_te(f_te, lr_up)
        y_st_img, _ = self.sam_st(f_st, lr_up)
        y_sr = self.merge(torch.cat([y_m, y_st_img], dim=1))
        y_st = self.struct_head(f_st)
        return y_sr, y_m, y_st
