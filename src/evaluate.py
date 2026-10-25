"""Evaluate every trained run on the test set: per-volume Dice, volumes, inference time."""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

from src.dataset import load_split
from src.inference import dice_per_class, keep_largest_component, predict_volume
from src.model import UNet

CLASSES = {1: "rv", 2: "myo", 3: "lv"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/test_per_volume.csv")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    patients = load_split(args.split, args.data_dir)
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)

    fields = (["run", "width", "seed", "patient", "group", "phase", "post"]
              + [f"dice_{n}" for n in CLASSES.values()]
              + [f"vol_pred_{n}" for n in CLASSES.values()]
              + [f"vol_gt_{n}" for n in CLASSES.values()]
              + ["ms_per_slice"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "w*_s*"))):
            with open(os.path.join(run_dir, "summary.json")) as s:
                summary = json.load(s)
            model = UNet(base_channels=summary["width"]).to(device)
            model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"),
                                             map_location=device))
            model.eval()

            # Warm-up so the first timed volume is not penalized by initialization
            warmup_vol = next(iter(patients.values()))["img_ed"]
            for _ in range(3):
                predict_volume(model, warmup_vol, device)

            for pid, p in patients.items():
                voxel_ml = float(np.prod(p["spacing"])) / 1000  # mm³ -> mL
                for phase in ("ed", "es"):
                    img, gt = p[f"img_{phase}"], p[f"gt_{phase}"]
                    t = time.perf_counter()
                    pred = predict_volume(model, img, device)  # ends with .cpu(), so GPU is synced
                    ms_per_slice = (time.perf_counter() - t) * 1000 / len(img)

                    for post, mask in (("raw", pred), ("lcc", keep_largest_component(pred))):
                        dice = dice_per_class(mask, gt, classes=tuple(CLASSES))
                        row = {"run": os.path.basename(run_dir), "width": summary["width"],
                               "seed": summary["seed"], "patient": pid,
                               "group": meta[pid]["group"], "phase": phase, "post": post,
                               "ms_per_slice": f"{ms_per_slice:.2f}"}
                        for (c, name), d in zip(CLASSES.items(), dice):
                            row[f"dice_{name}"] = f"{d:.4f}"
                            row[f"vol_pred_{name}"] = f"{(mask == c).sum() * voxel_ml:.2f}"
                            row[f"vol_gt_{name}"] = f"{(gt == c).sum() * voxel_ml:.2f}"
                        writer.writerow(row)
            print(f"{os.path.basename(run_dir)} done")


if __name__ == "__main__":
    main()