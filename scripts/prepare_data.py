"""Preprocess ACDC: normalize, resample in-plane to 1.25 mm, center crop/pad
to 256x256, save one .npz per patient, and write the train/val/test split."""
import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
SPLIT_PATH = "configs/split.json"
TARGET_SPACING = 1.25  # mm, in-plane
TARGET_SIZE = 256      # px -> 320 mm field of view (validated by check_crop.py)
SEED = 974


def read_cfg(path):
    """Parse an ACDC Info.cfg file into a dict."""
    cfg = {}
    with open(path) as f:
        for line in f:
            key, value = line.split(":")
            cfg[key.strip()] = value.strip()
    return cfg


def normalize(vol):
    """Clip extreme intensities, then z-score the whole volume."""
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip(vol, lo, hi)
    return (vol - vol.mean()) / (vol.std() + 1e-8)


def resample_inplane(vol, spacing, order):
    """Resample an (X, Y, Z) volume so its in-plane spacing becomes TARGET_SPACING."""
    factors = (spacing[0] / TARGET_SPACING, spacing[1] / TARGET_SPACING, 1.0)
    return zoom(vol, factors, order=order)


def center_crop_or_pad(vol, size, pad_value):
    """Center crop or pad the first two axes of an (X, Y, Z) volume to size x size."""
    out = np.full((size, size, vol.shape[2]), pad_value, dtype=vol.dtype)
    src, dst = [], []
    for n in vol.shape[:2]:
        if n >= size:  # too big: take the central part
            start = (n - size) // 2
            src.append(slice(start, start + size))
            dst.append(slice(0, size))
        else:          # too small: place it in the center of the canvas
            start = (size - n) // 2
            src.append(slice(0, n))
            dst.append(slice(start, start + n))
    out[dst[0], dst[1]] = vol[src[0], src[1]]
    return out


def process_frame(pdir, pid, frame):
    """Load one frame (image + mask) and return preprocessed (Z, H, W) arrays."""
    img_nii = nib.load(os.path.join(pdir, f"{pid}_frame{frame:02d}.nii.gz"))
    gt_nii = nib.load(os.path.join(pdir, f"{pid}_frame{frame:02d}_gt.nii.gz"))
    spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])

    img = img_nii.get_fdata(dtype=np.float32)
    gt = np.asarray(gt_nii.dataobj).astype(np.uint8)
    assert img.shape == gt.shape, f"{pid} frame {frame}: image/mask shape mismatch"

    img = normalize(img)
    img = resample_inplane(img, spacing, order=1)  # linear interpolation
    gt = resample_inplane(gt, spacing, order=0)    # nearest neighbor: keeps labels valid

    img = center_crop_or_pad(img, TARGET_SIZE, pad_value=img.min())
    gt = center_crop_or_pad(gt, TARGET_SIZE, pad_value=0)

    # (X, Y, Z) -> (Z, X, Y): index 0 = slice, convenient for 2D training
    return img.transpose(2, 0, 1).astype(np.float16), gt.transpose(2, 0, 1), spacing


def make_split(meta, val_per_group=2):
    """Stratified train/val split on 'training' (by pathology group); 'testing' = test."""
    rng = np.random.default_rng(SEED)
    train, val = [], []
    for group in sorted({m["group"] for m in meta.values()}):
        pids = sorted(p for p, m in meta.items()
                      if m["subset"] == "training" and m["group"] == group)
        rng.shuffle(pids)
        val += pids[:val_per_group]
        train += pids[val_per_group:]
    test = sorted(p for p, m in meta.items() if m["subset"] == "testing")
    return {"train": sorted(train), "val": sorted(val), "test": test}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    meta = {}

    for pdir in sorted(glob.glob(os.path.join(RAW_DIR, "*", "patient*"))):
        pid = os.path.basename(pdir)
        subset = os.path.basename(os.path.dirname(pdir))  # "training" or "testing"
        cfg = read_cfg(os.path.join(pdir, "Info.cfg"))

        arrays = {}
        for phase in ("ED", "ES"):
            img, gt, spacing = process_frame(pdir, pid, int(cfg[phase]))
            arrays[f"img_{phase.lower()}"] = img
            arrays[f"gt_{phase.lower()}"] = gt

        np.savez_compressed(
            os.path.join(OUT_DIR, f"{pid}.npz"),
            **arrays,
            spacing=np.array([TARGET_SPACING, TARGET_SPACING, spacing[2]], dtype=np.float32),
            orig_spacing=np.array(spacing, dtype=np.float32),
        )
        meta[pid] = {"subset": subset, "group": cfg["Group"], "n_slices": int(img.shape[0])}
        print(f"{pid} ({subset}, {cfg['Group']}): {img.shape[0]} slices")

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    split = make_split(meta)
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)
    print({k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()