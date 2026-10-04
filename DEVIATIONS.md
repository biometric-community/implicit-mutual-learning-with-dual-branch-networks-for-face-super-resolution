# Deviations — IML Dual-Branch Face SR

| ID | Paper | Ours | Justification |
|----|-------|------|---------------|
| D1 | Structure maps via Relative Total Variation (Xu et al. 2012) | Lightweight iterative RTV approximation (`imlfsr/data/rtv.py`) | Official RTV solver not released with the paper; approx. preserves piecewise-smooth structure prior |
| D2 | Loss weights ω₁,ω₂,ω₃ unspecified numerically | Default `(1, 1, 0.1)` | Paper Eq. (5) only names the weights; edge term down-weighted to avoid dominating L₂ |
| D3 | Epoch count not stated | Full train **100 epochs** on 2k CelebA | Matches paper Adam/lr/batch; epoch budget chosen for stable PSNR plateau on small subset |
| D4 | LDHF real-world validation (100 images) | Not run | Dataset not available under `projects/datasets/` |
| D5 | Full DFRM/FIEB/SIEB/HFNet as drawn | Faithful PyTorch reconstruction of named blocks (BasicBlock, DSRB, DRB, TCA, ExchangeBlock, SAM/HFNet) | No official code; topology follows Sec. III & Figs. 2–7 |
| D6 | Helen + CelebA both reported | Full train default = CelebA 2k/60; Helen loader supported via `data.dataset=helen` | Size gate + primary ×8 CelebA protocol |
| D7 | Channel width unspecified | `n_feats=48`, `n_dfrm=2` (~2.4M params) | Ablation Fig. 9 discusses size; mid-size default for 3090-scale training |
