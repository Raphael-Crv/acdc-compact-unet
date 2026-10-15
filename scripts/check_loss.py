import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.dataset import load_split, PatchDataset
from src.losses import DiceCELoss

ds = PatchDataset(load_split("val"), augment=False)
gt = torch.stack([ds[i][1] for i in range(16)])  # a batch of 16 real masks
print("pixel share per class:", [f"{(gt == c).float().mean():.1%}" for c in range(4)])

loss = DiceCELoss()
onehot = F.one_hot(gt, 4).permute(0, 3, 1, 2).float()
cases = {
    "perfect": onehot * 5,                       # confident and correct everywhere
    "random": torch.randn(16, 4, 128, 128),      # untrained model
    "all background": F.one_hot(torch.zeros_like(gt), 4).permute(0, 3, 1, 2).float() * 5,
}
for name, logits in cases.items():
    ce = loss.ce(logits, gt).item()
    dice_loss = 1 - loss.soft_dice(logits, gt).item()
    print(f"{name:15s} | CE {ce:.3f} | Dice loss {dice_loss:.3f} | total {loss(logits, gt).item():.3f}")