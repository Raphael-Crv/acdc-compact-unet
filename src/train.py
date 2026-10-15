"""Train a U-Net of a given width on ACDC patches; validate on full val volumes."""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PatchDataset, load_split
from src.inference import dice_per_class, predict_volume
from src.losses import DiceCELoss
from src.model import UNet, count_params
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default="runs")
    return p.parse_args()


def validate(model, patients, device):
    """Mean hard Dice per class (RV, MYO, LV) over all val volumes (ED + ES)."""
    scores = []
    for p in patients.values():
        for phase in ("ed", "es"):
            pred = predict_volume(model, p[f"img_{phase}"], device)
            scores.append(dice_per_class(pred, p[f"gt_{phase}"]))
    return np.mean(scores, axis=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run_dir = os.path.join(args.out_dir, f"w{args.width}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Data
    train_ds = PatchDataset(load_split("train", args.data_dir),
                            samples_per_epoch=args.samples_per_epoch)
    val_patients = load_split("val", args.data_dir)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=use_amp, persistent_workers=args.num_workers > 0, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),  # reproducible worker seeds
    )

    # Model, loss, optimizer
    model = UNet(base_channels=args.width).to(device)
    loss_fn = DiceCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device {device} | width {args.width} | {count_params(model) / 1e6:.2f} M params")

    log_path = os.path.join(run_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "dice_rv", "dice_myo", "dice_lv",
                                "dice_mean", "lr", "elapsed_s"])

    best_dice, best_epoch = -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Training: one pass over samples_per_epoch random patches
        model.train()
        running = 0.0
        for img, gt in loader:
            img = img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(img), gt)
            scaler.scale(loss).backward()  # compute gradients
            scaler.step(opt)               # update weights
            scaler.update()
            running += loss.item()
        lr = sched.get_last_lr()[0]
        sched.step()

        # Validation on full volumes, keep the best checkpoint
        dice = validate(model, val_patients, device)
        elapsed = time.time() - t0
        if dice.mean() > best_dice:
            best_dice, best_epoch = float(dice.mean()), epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

        train_loss = running / len(loader)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", *(f"{d:.4f}" for d in dice),
                                    f"{dice.mean():.4f}", f"{lr:.2e}", f"{elapsed:.1f}"])
        print(f"epoch {epoch:3d} | loss {train_loss:.3f} | val Dice RV {dice[0]:.3f} "
              f"MYO {dice[1]:.3f} LV {dice[2]:.3f} | mean {dice.mean():.3f} | {elapsed:.0f}s")

    summary = {
        "width": args.width, "seed": args.seed, "params": count_params(model),
        "best_val_dice": best_dice, "best_epoch": best_epoch,
        "train_time_s": round(time.time() - t0, 1), "device": str(device),
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"best val Dice {best_dice:.3f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()