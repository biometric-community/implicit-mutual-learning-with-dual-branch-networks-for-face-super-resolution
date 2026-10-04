"""Relative Total Variation (RTV) structure extraction — Xu et al., TOG 2012 [40].

Lightweight iterative RTV used as the paper's structure-map prior.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _gaussian_blur_np(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (reflect pad)."""
    if sigma <= 0:
        return img
    radius = max(int(3 * sigma), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    pad = radius
    out = img
    # horizontal
    tmp = np.pad(out, ((0, 0), (pad, pad), (0, 0)) if out.ndim == 3 else ((0, 0), (pad, pad)), mode="reflect")
    if out.ndim == 3:
        acc = np.zeros_like(out, dtype=np.float64)
        for i, w in enumerate(k):
            acc += w * tmp[:, i : i + out.shape[1], :]
        out = acc
        tmp = np.pad(out, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
        acc = np.zeros_like(out, dtype=np.float64)
        for i, w in enumerate(k):
            acc += w * tmp[i : i + out.shape[0], :, :]
        return acc.astype(np.float32)
    acc = np.zeros_like(out, dtype=np.float64)
    for i, w in enumerate(k):
        acc += w * tmp[:, i : i + out.shape[1]]
    out = acc
    tmp = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
    acc = np.zeros_like(out, dtype=np.float64)
    for i, w in enumerate(k):
        acc += w * tmp[i : i + out.shape[0], :]
    return acc.astype(np.float32)


def rtv_structure(
    img: np.ndarray,
    *,
    lambda_: float = 0.015,
    sigma: float = 3.0,
    sharpness: float = 0.02,
    max_iter: int = 4,
) -> np.ndarray:
    """
    Approximate RTV structure map.

    Parameters match common RTV demos; paper cites Xu et al. without hyperparameters.
    ``img``: float32 HxWxC in [0,1] or HxW.
    """
    u = img.astype(np.float64)
    eps = 1e-4
    for _ in range(max_iter):
        # windowed total variation proxies via blurred gradients
        if u.ndim == 3:
            gx = np.diff(u, axis=1, append=u[:, -1:, :])
            gy = np.diff(u, axis=0, append=u[-1:, :, :])
        else:
            gx = np.diff(u, axis=1, append=u[:, -1:])
            gy = np.diff(u, axis=0, append=u[-1:, :])
        abs_gx = np.abs(gx)
        abs_gy = np.abs(gy)
        lx = _gaussian_blur_np(abs_gx, sigma) + eps
        ly = _gaussian_blur_np(abs_gy, sigma) + eps
        wx = _gaussian_blur_np(abs_gx / lx, sigma) + eps
        wy = _gaussian_blur_np(abs_gy / ly, sigma) + eps
        # anisotropic diffusion-like update toward piecewise-smooth structure
        if u.ndim == 3:
            ux = np.diff(u, axis=1, prepend=u[:, :1, :])
            uy = np.diff(u, axis=0, prepend=u[:1, :, :])
        else:
            ux = np.diff(u, axis=1, prepend=u[:, :1])
            uy = np.diff(u, axis=0, prepend=u[:1, :])
        num = img.astype(np.float64) + lambda_ * (
            _gaussian_blur_np(wx * ux, sharpness) + _gaussian_blur_np(wy * uy, sharpness)
        )
        den = 1.0 + lambda_ * (_gaussian_blur_np(wx, sharpness) + _gaussian_blur_np(wy, sharpness))
        u = num / (den + eps)
        u = np.clip(u, 0.0, 1.0)
    return u.astype(np.float32)


def pil_to_structure(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return rtv_structure(arr)
