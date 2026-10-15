"""Inference and evaluation helpers shared by training and evaluation."""
import numpy as np
import torch


@torch.no_grad()
def predict_volume(model, vol, device, batch_size=16):
    """Segment a (Z, H, W) volume slice by slice; return a (Z, H, W) uint8 label map."""
    model.eval()
    x = torch.from_numpy(vol.astype(np.float32))[:, None]  # (Z, 1, H, W)
    preds = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(x[i:i + batch_size].to(device))
        preds.append(logits.argmax(dim=1).cpu())  # most likely class per pixel
    return torch.cat(preds).numpy().astype(np.uint8)


def dice_per_class(pred, gt, classes=(1, 2, 3)):
    """Hard 3D Dice for each class; 1.0 if the class is absent from both pred and gt."""
    scores = []
    for c in classes:
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        scores.append(1.0 if denom == 0 else 2 * np.logical_and(p, g).sum() / denom)
    return np.array(scores)