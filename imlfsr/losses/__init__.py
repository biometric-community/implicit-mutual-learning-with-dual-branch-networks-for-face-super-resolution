"""Joint loss Eq. (5–8): Lrec + Lst + Ledge (Laplacian)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """Depthwise Laplacian operator ∇ on RGB."""
    kernel = x.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


class IMLLoss(nn.Module):
    def __init__(self, w_rec: float = 1.0, w_st: float = 1.0, w_edge: float = 0.1, delta: float = 0.001):
        super().__init__()
        self.w_rec = float(w_rec)
        self.w_st = float(w_st)
        self.w_edge = float(w_edge)
        self.delta = float(delta)

    def forward(self, out: dict, hr: torch.Tensor, st_hr: torch.Tensor):
        sr, mid, st = out["sr"], out["sr_mid"], out["sr_struct"]
        # Eq. 6
        l_rec = F.mse_loss(sr, hr) + F.mse_loss(mid, hr)
        # Eq. 7
        l_st = F.mse_loss(st, st_hr)
        # Eq. 8: sqrt(||Δgt-Δsr||^2 + δ^2)
        d = self.delta
        l_edge = torch.sqrt((laplacian(hr) - laplacian(sr)).pow(2).mean() + d**2) + torch.sqrt(
            (laplacian(hr) - laplacian(mid)).pow(2).mean() + d**2
        )
        total = self.w_rec * l_rec + self.w_st * l_st + self.w_edge * l_edge
        return total, {
            "rec": float(l_rec.detach()),
            "st": float(l_st.detach()),
            "edge": float(l_edge.detach()),
        }
