import sys, time
import torch

sys.path.insert(0, ".")
from src.model import UNet, count_params

for width in (8, 16, 32, 64):
    model = UNet(base_channels=width).eval()
    with torch.no_grad():
        for size in (128, 256):  # training patch, then full slice
            x = torch.randn(1, 1, size, size)
            t = time.time()
            y = model(x)
            print(f"width {width:2d} | {count_params(model) / 1e6:5.2f} M params | "
                  f"in {tuple(x.shape)} -> out {tuple(y.shape)} | {(time.time() - t) * 1000:.0f} ms on CPU")