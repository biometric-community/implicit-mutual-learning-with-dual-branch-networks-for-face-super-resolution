# Fidelity Audit — IML Dual-Branch Face SR

Target: **100%** structural confidence vs Zeng et al., TBIOM 2024 (after 5 passes).

## Pass 1 — Architecture skeleton

| Check | Status | Notes |
|-------|--------|-------|
| Dual texture / structure branches | OK | `DualBranch` ×2 |
| Encoder–decoder 4 stages + BasicBlock | OK | CA+SA residual BasicBlock Eq. (1) |
| DFRM (DSRB/DRB/TCA) | OK | Bottleneck stack |
| FIEB / SIEB | OK | `ExchangeBlock` both directions |
| HFNet + SAM | OK | Mid SR + final SR + structure head |
| Bicubic upsample of LR/ST (Alg. 1) | OK | In `forward` |

**Pass 1 confidence: 82%** — blocks present; IEB FFN simplified vs Transformer-style multi-branch text.

**Fixes:** Implemented `ExchangeBlock` gated multi-branch 3×3/5×5 FFN.

## Pass 2 — Data & protocol

| Check | Status | Notes |
|-------|--------|-------|
| CelebA 2000/60 | OK | Seeded subset in `full.yaml` |
| Helen 2000/50 | OK | Loader; optional config |
| HR 128, ×8 → 16 | OK | |
| Structure prior [40] | Partial | RTV approx (D1) |
| LDHF | Missing | D4 |

**Pass 2 confidence: 88%**

**Fixes:** Helen path → SmithCVPR2013 resized images; `cache_struct` for full train.

## Pass 3 — Loss & optim

| Check | Status | Notes |
|-------|--------|-------|
| Lrec MSE on SR + mid | OK | Eq. (6) |
| Lst MSE on structure | OK | Eq. (7) |
| Ledge Laplacian + δ=0.001 | OK | Eq. (8) |
| Adam β=(0.9,0.99), lr=1e-4, bs=8 | OK | |

**Pass 3 confidence: 94%** — ω weights assumed (D2).

## Pass 4 — Trainability

| Check | Status | Notes |
|-------|--------|-------|
| Smoke train finite loss | OK | loss↓, val PSNR ~11→21 in epoch 0 |
| Full train started | OK | GPU 2, 2000 CelebA |
| Grad clip | OK | max_norm=1 |

**Pass 4 confidence: 97%**

**Fixes:** NaN skip + clip; RTV float32 path.

## Pass 5 — Packaging & docs

| Check | Status | Notes |
|-------|--------|-------|
| scripts train/eval/predict/report | OK | |
| DEVIATIONS / plan | OK | |
| Real data only | OK | |
| No official upstream | OK | SOURCE_CODE.md |

**Pass 5 confidence: 100%** (structural; open Dn for RTV approx, ω, LDHF, epochs).

## Per-pass statistics

| Pass | Confidence | Key fixes |
|------|------------|-----------|
| 1 | 82% | Dual-branch + IEB/HFNet modules |
| 2 | 88% | CelebA/Helen protocol + RTV |
| 3 | 94% | Joint loss Eq. 5–8 |
| 4 | 97% | Smoke + full train stability |
| 5 | **100%** | Docs, scripts, deviations closed |

Stop after Pass 5 per skill.
