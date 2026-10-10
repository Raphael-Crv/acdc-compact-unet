"""Check how much of the ground-truth mask a centered 160 mm crop would cut off."""
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd

CROP_MM = 160  # 128 px * 1.25 mm


def read_cfg(path):
    """Parse an ACDC Info.cfg file into a dict."""
    cfg = {}
    with open(path) as f:
        for line in f:
            key, value = line.split(":")
            cfg[key.strip()] = value.strip()
    return cfg


rows = []
for pdir in sorted(glob.glob("data/raw/*/patient*")):
    pid = os.path.basename(pdir)
    cfg = read_cfg(os.path.join(pdir, "Info.cfg"))

    for phase in ["ED", "ES"]:
        frame = int(cfg[phase])
        nii = nib.load(os.path.join(pdir, f"{pid}_frame{frame:02d}_gt.nii.gz"))
        heart = np.asarray(nii.dataobj) > 0  # True for any cardiac label
        sx, sy = nii.header.get_zooms()[:2]
        nx, ny = heart.shape[:2]

        # Crop window in original pixels, centered on the image
        hx, hy = CROP_MM / 2 / sx, CROP_MM / 2 / sy
        x0, x1 = max(int(round(nx / 2 - hx)), 0), int(round(nx / 2 + hx))
        y0, y1 = max(int(round(ny / 2 - hy)), 0), int(round(ny / 2 + hy))

        total = heart.sum()
        inside = heart[x0:x1, y0:y1, :].sum()

        # Distance between heart centroid and image center, in mm
        xs, ys, _ = np.nonzero(heart)
        dx = (xs.mean() - nx / 2) * sx
        dy = (ys.mean() - ny / 2) * sy

        rows.append({
            "patient": pid,
            "phase": phase,
            "group": cfg.get("Group", "?"),
            "lost_pct": 100 * (1 - inside / total),
            "offset_mm": np.hypot(dx, dy),
        })

df = pd.DataFrame(rows)
print(df["lost_pct"].describe(), "\n")
print("Worst cases:")
print(df.sort_values("lost_pct", ascending=False).head(10).to_string(index=False))
print(f"\nMax centroid offset: {df['offset_mm'].max():.1f} mm")

os.makedirs("results", exist_ok=True)
df.to_csv("results/crop_check_160.csv", index=False)