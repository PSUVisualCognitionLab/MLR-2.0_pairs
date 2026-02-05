from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_memmap(path: Path) -> np.memmap:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2 and arr.shape[1] == 784:
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (28, 28):
        return arr
    raise ValueError(f"Unexpected shape in {path}: {arr.shape}")


def _ensure_28x28_uint8(a: np.ndarray) -> np.ndarray:
    return a.reshape(28, 28) if a.ndim == 1 else a

class QuickDrawPairDatasetFast(Dataset):

    def __init__(self, root, class_pairs, classes_dirname="bitmaps", items_per_pair=70000, seed=123):
        self.root = Path(root)
        self.dir = self.root / classes_dirname if classes_dirname else self.root
        self.class_pairs = list(class_pairs)
        self.items_per_pair = int(items_per_pair)

        # Stable label space = sorted unique class names across all pairs
        self.classes = sorted({c for p in class_pairs for c in p})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Load each needed class memmapped
        self._arrays = {}
        for c in self.classes:
            p = self.dir / f"full_numpy_bitmap_{c}.npy"
            if not p.exists():
                alt = self.dir / f"{c}.npy"
                p = alt if alt.exists() else p
            if not p.exists():
                raise FileNotFoundError(p)
            self._arrays[c] = _load_memmap(p)

        # for deterministic per-epoch randomness
        self._base_seed = int(seed)
        self._epoch = 0

    def set_epoch(self, e: int):
        self._epoch = int(e)

    def _rng(self, index: int):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        s = (self._base_seed + 1009 * self._epoch + 7919 * wid + 97 * index) & 0xFFFFFFFF
        return np.random.default_rng(s)

    def __len__(self):
        return len(self.class_pairs) * self.items_per_pair

    def __getitem__(self, index: int):
        pair_idx = index % len(self.class_pairs)
        l_name, r_name = self.class_pairs[pair_idx]
        la, ra = self._arrays[l_name], self._arrays[r_name]

        rng = self._rng(index)
        li = rng.integers(0, la.shape[0])
        ri = rng.integers(0, ra.shape[0])

        l = _ensure_28x28_uint8(la[li])
        r = _ensure_28x28_uint8(ra[ri])

        l = torch.from_numpy(l).unsqueeze(0)  # (1,28,28) uint8
        r = torch.from_numpy(r).unsqueeze(0)  # (1,28,28) uint8

        y = torch.tensor([self.class_to_idx[l_name], self.class_to_idx[r_name]], dtype=torch.long)
        return l, r, y, pair_idx
