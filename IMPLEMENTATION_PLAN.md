# Implementation Plan — Implicit Mutual Learning Dual-Branch Face SR

**Paper:** Zeng et al., TBIOM 2024, DOI 10.1109/TBIOM.2024.3354333  
**Slug:** `implicit-mutual-learning-with-dual-branch-networks-for-face-super-resolution`  
**Package:** `imlfsr`  
**Open-source code:** none found (IEEE-only); implement from paper description.

## Method summary

Dual-branch FSR: **texture enhancement** (RGB) + **structure reconstruction** (RTV structure maps [Xu et al. 2012]). Each branch is an encoder–decoder with four scale stages (BasicBlock = residual + channel/spatial attention). Cross-branch **FIEB** (encoder) and **SIEB** (decoder) exchange features. **DFRM** (DSRB + DRB + TCA) reinforces deep features. **HFNet** (supervised attention) fuses branch outputs into intermediate SR \(Y^M_{SR}\), final SR \(Y_{SR}\), and structure \(Y^{ST}_{SR}\).

## Datasets (paper Sec. IV-A)

| Set | Train | Test | Local root |
|-----|------:|-----:|------------|
| Helen | 2000 | 50 | `projects/datasets/helen` |
| CelebA | 2000 | 60 | `projects/datasets/celeba/extracted/celeba` |
| LDHF | 100 val | — | unavailable → `D_n` skip |

HR crop/resize **128×128**; LR **32×32 (×4)** or **16×16 (×8)**. Bicubic degradation. Primary protocol: **×8 on CelebA** (smoke + full).

## Training (paper)

- Adam \(\beta_1=0.9\), \(\beta_2=0.99\), lr \(1\times10^{-4}\), batch 8  
- Loss: \(L=\omega_1 L_{rec}+\omega_2 L_{st}+\omega_3 L_{edge}\) (Eq. 5–8); Laplacian edge \(\delta=0.001\)  
- Weights not numerically specified → default \(\omega=(1,1,0.1)\) (`D_n`)  
- Epochs not specified → full config **100 epochs** on 2k CelebA subset

## Deliverables

Package `imlfsr/` (models, data, losses, train/eval/predict/report), configs smoke/full, scripts, DEVIATIONS, FIDELITY_AUDIT (5 passes), README/LICENSE, publish `biometric-community/<slug>`.

## Size gate

CelebA (~3.1 GiB) + Helen (~0.8 GiB) ≪ 5 GiB → **mandatory full train** after smoke.
