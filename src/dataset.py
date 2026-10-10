"""PyTorch datasets for ACDC: random 2D patches for training."""
import json
import os

import numpy as np
import torch
from scipy.ndimage import rotate
from torch.utils.data import Dataset

DATA_DIR = "data/processed"
SPLIT_PATH = "configs/split.json"


def load_split(name, data_dir=DATA_DIR, split_path=SPLIT_PATH):
    """Return {pid: dict of arrays} for every patient of a split ('train', 'val', 'test')."""
    with open(split_path) as f:
        pids = json.load(f)[name]
    patients = {}
    for pid in pids:
        with np.load(os.path.join(data_dir, f"{pid}.npz")) as d:
            patients[pid] = {k: d[k] for k in d.files}
    return patients


class PatchDataset(Dataset):
    """Random 2D patches drawn from all slices (ED + ES) of a set of patients."""

    def __init__(self, patients, patch_size=128, fg_prob=2 / 3,
                 samples_per_epoch=2000, augment=True):
        self.patch_size = patch_size
        self.fg_prob = fg_prob
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        # Larger patch before rotation, so corners are still filled after rotating
        self.extract_size = int(np.ceil(patch_size * np.sqrt(2))) if augment else patch_size
        self._rng = None

        self.slices = []  # (image, mask, coordinates of cardiac pixels)
        for p in patients.values():
            for phase in ("ed", "es"):
                for img, gt in zip(p[f"img_{phase}"], p[f"gt_{phase}"]):
                    self.slices.append((img, gt, np.argwhere(gt > 0)))

    def __len__(self):
        return self.samples_per_epoch

    def _get_rng(self):
        # One generator per DataLoader worker (torch gives each worker its own seed)
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._rng

    def _pick_center(self, rng, shape, fg):
        """Center on a random cardiac pixel with prob fg_prob, else anywhere."""
        if len(fg) > 0 and rng.random() < self.fg_prob:
            cy, cx = fg[rng.integers(len(fg))]
        else:
            cy, cx = rng.integers(shape[0]), rng.integers(shape[1])
        # Clamp so the patch stays inside the slice
        half = self.extract_size // 2
        cy = int(np.clip(cy, half, shape[0] - self.extract_size + half))
        cx = int(np.clip(cx, half, shape[1] - self.extract_size + half))
        return cy - half, cx - half  # top-left corner

    def _augment(self, rng, img, gt):
        # Small random rotation, then center crop back to patch_size
        angle = rng.uniform(-20, 20)
        img = rotate(img, angle, reshape=False, order=1, mode="nearest")
        gt = rotate(gt, angle, reshape=False, order=0, mode="nearest")
        off = (self.extract_size - self.patch_size) // 2
        img = img[off:off + self.patch_size, off:off + self.patch_size]
        gt = gt[off:off + self.patch_size, off:off + self.patch_size]

        # Random 90-degree rotations and flips (orientation varies across scanners)
        k = rng.integers(4)
        img, gt = np.rot90(img, k), np.rot90(gt, k)
        if rng.random() < 0.5:
            img, gt = img[::-1], gt[::-1]

        # Intensity: contrast/brightness jitter and occasional Gaussian noise
        img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        if rng.random() < 0.3:
            img = img + rng.normal(0, rng.uniform(0, 0.1), img.shape)
        return img, gt

    def __getitem__(self, idx):
        rng = self._get_rng()
        img, gt, fg = self.slices[rng.integers(len(self.slices))]
        y0, x0 = self._pick_center(rng, gt.shape, fg)
        s = self.extract_size
        img = img[y0:y0 + s, x0:x0 + s].astype(np.float32)
        gt = gt[y0:y0 + s, x0:x0 + s]
        if self.augment:
            img, gt = self._augment(rng, img, gt)
        # (1, H, W) float image and (H, W) integer mask, contiguous for torch
        img = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)[None])
        gt = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.int64))
        return img, gt