# Implicit Mutual Learning With Dual-Branch Networks for Face Super-Resolution

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

PyTorch rebuild of Zeng et al. (TBIOM 2024): dual-branch texture/structure face SR with DFRM, FIEB/SIEB, and HFNet.

**Repository:** https://github.com/biometric-community/implicit-mutual-learning-with-dual-branch-networks-for-face-super-resolution

No official upstream code was released; this tree follows the paper description.

## Setup

```bash
bash scripts/setup_env.sh
```

Point configs at real CelebA (and optionally Helen) under `projects/datasets/`.

## Train / eval / report

```bash
bash scripts/train.sh                 # smoke
bash scripts/train_full.sh            # CelebA 2k/60 ×8 (size gate <5 GiB)
bash scripts/eval.sh && bash scripts/predict.sh && bash scripts/report.sh
```

See [REPORT.md](REPORT.md), [DEVIATIONS.md](DEVIATIONS.md), [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Citation

```bibtex
@ARTICLE{10409565,
  author={Zeng, Kangli and Wang, Zhongyuan and Lu, Tao and Chen, Jianyu and He, Zheng and Han, Zhen},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  title={Implicit Mutual Learning With Dual-Branch Networks for Face Super-Resolution},
  year={2024},
  volume={6},
  number={2},
  pages={182-194},
  doi={10.1109/TBIOM.2024.3354333}
}
```
