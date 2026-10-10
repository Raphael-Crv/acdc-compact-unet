import sys, time
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from src.dataset import load_split, PatchDataset

ds = PatchDataset(load_split("train"))
print(f"{len(ds.slices)} slices, {len(ds)} patches per epoch")

t = time.time()
samples = [ds[i] for i in range(100)]
print(f"{(time.time() - t) * 10:.1f} ms per patch")
img, gt = samples[0]
print(img.shape, img.dtype, gt.shape, gt.dtype, gt.unique())
print(f"patches with heart: {np.mean([(g > 0).any().item() for _, g in samples]):.0%}")

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, (img, gt) in zip(axes.flat, samples):
    ax.imshow(img[0], cmap="gray")
    ax.imshow(np.ma.masked_equal(gt.numpy(), 0), cmap="jet", alpha=0.4, vmin=0, vmax=3)
    ax.axis("off")
plt.tight_layout()
plt.show()