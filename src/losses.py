"""Segmentation loss: soft Dice + cross-entropy."""
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Sum of cross-entropy and soft Dice loss (Dice on foreground classes only)."""

    def __init__(self, n_classes=4, smooth=1e-5, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def soft_dice(self, logits, target):
        """Mean soft Dice over foreground classes, pooled over the whole batch."""
        probs = logits.float().softmax(dim=1)  # (B, C, H, W), float32 for stability
        onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)  # sum over batch and pixels, keep one value per class
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return dice[1:].mean()  # skip background (class 0)

    def forward(self, logits, target):
        ce_loss = self.ce(logits.float(), target)
        dice_loss = 1 - self.soft_dice(logits, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss