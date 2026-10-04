"""PSNR / SSIM on Y channel with border crop (upstream util.py, Sec. IV-A)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch


def rgb2ycbcr(img: np.ndarray, only_y: bool = True) -> np.ndarray:
    in_type = img.dtype
    img = img.astype(np.float32)
    if in_type != np.uint8:
        img = img * 255.0
    if only_y:
        y = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        y = np.matmul(
            img,
            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]],
        ) / 255.0 + np.array([16, 128, 128])
    if in_type == np.uint8:
        return y.round().astype(in_type)
    return (y / 255.0).astype(np.float32)


def _ssim_map(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu12 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def calc_psnr_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 8,
    test_y: bool = True,
) -> tuple[float, float]:
    """``pred/target``: (C,H,W) or (B,C,H,W) in [0,1]."""
    if pred.dim() == 4:
        psnr_v, ssim_v, n = 0.0, 0.0, 0
        for i in range(pred.size(0)):
            p, s = calc_psnr_ssim(pred[i], target[i], crop_border, test_y)
            psnr_v += p
            ssim_v += s
            n += 1
        return psnr_v / max(n, 1), ssim_v / max(n, 1)

    p = pred.detach().cpu().numpy().transpose(1, 2, 0)
    t = target.detach().cpu().numpy().transpose(1, 2, 0)
    if test_y and p.shape[2] == 3:
        p = rgb2ycbcr(p)
        t = rgb2ycbcr(t)
    cb = crop_border
    if p.ndim == 3:
        p = p[:, cb:-cb, cb:-cb]
        t = t[:, cb:-cb, cb:-cb]
    else:
        p = p[cb:-cb, cb:-cb]
        t = t[cb:-cb, cb:-cb]
    p255 = (p * 255.0).astype(np.float64)
    t255 = (t * 255.0).astype(np.float64)
    mse = np.mean((p255 - t255) ** 2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    ssim = _ssim_map(p255, t255)
    return float(psnr), float(ssim)
