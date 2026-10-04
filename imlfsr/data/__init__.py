"""CelebA / Helen face SR loaders with RTV structure maps (Sec. IV-A)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .rtv import rtv_structure


def resolve_data_root(root: str | Path) -> Path:
    p = Path(root)
    if p.is_absolute() and p.exists():
        return p
    proj = Path(__file__).resolve().parents[2]
    cand = (proj / p).resolve()
    if cand.exists():
        return cand
    repo = proj
    while repo != repo.parent:
        if (repo / "docs" / "PAPERS.md").exists():
            break
        repo = repo.parent
    stripped = Path(*[x for x in p.parts if x != ".."])
    alt = (repo / "projects" / stripped).resolve()
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Dataset root not found: {root}")


def _list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]
    return files


class FaceSRStructureDataset(Dataset):
    """HR 128×128, bicubic LR, RTV structure for LR/HR."""

    def __init__(
        self,
        paths: List[Path],
        img_size: int = 128,
        scale: int = 8,
        cache_struct: bool = False,
    ):
        self.paths = paths
        self.img_size = img_size
        self.scale = scale
        self.lr_size = img_size // scale
        self._struct_cache: Dict[str, np.ndarray] = {}
        self.cache_struct = cache_struct

    def __len__(self) -> int:
        return len(self.paths)

    def _load_hr(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img.resize((self.img_size, self.img_size), Image.BICUBIC)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        hr_pil = self._load_hr(path)
        hr = np.asarray(hr_pil, dtype=np.float32) / 255.0
        key = str(path)
        if self.cache_struct and key in self._struct_cache:
            st_hr = self._struct_cache[key]
        else:
            st_hr = rtv_structure(hr)
            if self.cache_struct:
                self._struct_cache[key] = st_hr

        hr_t = torch.from_numpy(hr).permute(2, 0, 1)
        st_hr_t = torch.from_numpy(st_hr).permute(2, 0, 1)
        lr_t = F.interpolate(
            hr_t.unsqueeze(0), size=(self.lr_size, self.lr_size), mode="bicubic", align_corners=False
        ).squeeze(0)
        st_lr_t = F.interpolate(
            st_hr_t.unsqueeze(0), size=(self.lr_size, self.lr_size), mode="bicubic", align_corners=False
        ).squeeze(0)
        return {
            "lr": lr_t.clamp(0, 1),
            "hr": hr_t.clamp(0, 1),
            "st_lr": st_lr_t.clamp(0, 1),
            "st_hr": st_hr_t.clamp(0, 1),
            "name": path.name,
        }


def _celeba_paths(root: Path, split: str, max_samples: Optional[int], seed: int) -> List[Path]:
    img_dir = root / "img_align_celeba"
    part = root / "list_eval_partition.txt"
    if not img_dir.is_dir() or not part.is_file():
        raise FileNotFoundError(f"CelebA incomplete under {root}")
    split_id = {"train": 0, "val": 1, "test": 2}[split]
    names = []
    with open(part) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[1]) == split_id:
                names.append(parts[0])
    names = sorted(names)
    # Paper: 2000 train / 60 test random subset — use seeded sample when max_samples set
    if max_samples is not None and max_samples > 0:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(names), size=min(max_samples, len(names)), replace=False)
        names = [names[i] for i in sorted(idx.tolist())]
    return [img_dir / n for n in names]


def _helen_paths(root: Path, split: str, max_samples: Optional[int], seed: int) -> List[Path]:
    # Prefer SmithCVPR2013 resized images (local layout)
    for cand in [
        "extracted/SmithCVPR2013_dataset_resized/images",
        "SmithCVPR2013_dataset_resized/images",
        "images",
        "cropped_train",
        "train",
    ]:
        d = root / cand
        if d.is_dir() and any(d.iterdir()):
            break
    else:
        d = root
    files = _list_images(d)
    if not files:
        # try one level deeper
        files = _list_images(root)
    if not files:
        raise FileNotFoundError(f"No Helen images under {root}")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]
    # Paper: 2000 train / 50 test
    n_test = 50
    if split == "test":
        files = files[:n_test]
    elif split == "val":
        files = files[n_test : n_test + 50]
    else:
        files = files[n_test + 50 :]
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]
    return files


def build_dataset(cfg: dict, split: str) -> FaceSRStructureDataset:
    data = cfg["data"]
    name = data.get("dataset", "celeba").lower()
    root = resolve_data_root(data["root"])
    max_key = {"train": "max_train_samples", "val": "max_val_samples", "test": "max_test_samples"}[split]
    max_samples = data.get(max_key)
    seed = int(cfg.get("train", {}).get("seed", 42))
    if name == "celeba":
        # map val → official val partition; paper uses train/test only
        paths = _celeba_paths(root, split if split != "val" else "val", max_samples, seed)
    elif name == "helen":
        paths = _helen_paths(root, split, max_samples, seed)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return FaceSRStructureDataset(
        paths,
        img_size=int(cfg["model"].get("hr_size", 128)),
        scale=int(cfg["model"].get("scale", 8)),
        cache_struct=bool(data.get("cache_struct", False)),
    )


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    ds = build_dataset(cfg, split)
    bs = int(cfg["train"]["batch_size"] if split == "train" else cfg.get("eval", {}).get("batch_size", 8))
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == "train"),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=(split == "train"),
    )
