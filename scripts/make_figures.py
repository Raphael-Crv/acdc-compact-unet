"""Figures for the README: accuracy vs cost, learning curves, qualitative examples."""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, ".")
from src.dataset import load_split
from src.inference import keep_largest_component, predict_volume
from src.model import UNet

OUT = "results/figures"
NNUNET_DICE = 0.9295  # published nnU-Net ensemble, ACDC test set (Isensee et al., 2021)
CMAP = ListedColormap(["#1f77b4", "#ffd92f", "#d62728"])  # RV, MYO, LV
LEGEND = [Patch(color=c, label=n) for c, n in zip(CMAP.colors, ["RV", "MYO", "LV"])]


def accuracy_vs_cost():
    """Mean test Dice (95% CI) vs parameter count and vs CPU inference time."""
    t = pd.read_csv("results/summary_test.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for post, fmt, label in (("raw", "--o", "no post-processing"),
                             ("lcc", "-o", "largest connected component")):
        s = t[t["post"] == post].sort_values("width")
        yerr = [s["dice_mean"] - s["dice_mean_lo"], s["dice_mean_hi"] - s["dice_mean"]]
        for ax, x in zip(axes, (s["params"] / 1e6, s["ms_per_slice_cpu"])):
            ax.errorbar(x, s["dice_mean"], yerr=yerr, fmt=fmt, capsize=4, label=label)
            if post == "lcc":
                for xi, yi, w in zip(x, s["dice_mean"], s["width"]):
                    ax.annotate(f"w{w}", (xi, yi), textcoords="offset points",
                                xytext=(0, 10), ha="center", fontsize=9)
    for ax, xlabel in zip(axes, ("Parameters (millions, log scale)",
                                 "CPU inference time per slice (ms, log scale)")):
        ax.axhline(NNUNET_DICE, color="gray", linestyle=":", label="nnU-Net ensemble (published)")
        ax.set_xscale("log")
        ticks = [0.5, 1, 2, 4, 8, 16, 32] if "Parameters" in xlabel else [25, 50, 100, 200, 400]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean Dice on ACDC test set (95% CI)")
    axes[0].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "accuracy_vs_cost.png"), dpi=150)
    plt.close()


def learning_curves():
    """Validation Dice per epoch: mean over seeds, shaded min-max range."""
    logs = []
    for f in glob.glob("runs/w*_s*/log.csv"):
        d = pd.read_csv(f)
        d["width"] = int(os.path.basename(os.path.dirname(f)).split("_")[0][1:])
        logs.append(d)
    curves = pd.concat(logs).groupby(["width", "epoch"])["dice_mean"].agg(["mean", "min", "max"])
    plt.figure(figsize=(8, 4.5))
    for width, c in curves.groupby(level="width"):
        c = c.droplevel("width")
        plt.plot(c.index, c["mean"], label=f"width {width}")
        plt.fill_between(c.index, c["min"], c["max"], alpha=0.2)
    plt.ylim(0.80, 0.93)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice (mean over 3 seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "learning_curves.png"), dpi=150)
    plt.close()


def overlay(ax, img, mask, title):
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_equal(mask, 0), cmap=CMAP, vmin=1, vmax=3,
              alpha=0.45, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def qualitative(run="w16_s0", width=16):
    """Worst, median and best test volumes: expert vs model on 3 slices each."""
    df = pd.read_csv("results/test_per_volume.csv")
    sel = df[(df["run"] == run) & (df["post"] == "lcc")].copy()
    sel["dice_mean"] = sel[["dice_rv", "dice_myo", "dice_lv"]].mean(axis=1)
    sel = sel.sort_values("dice_mean").reset_index(drop=True)
    cases = {"worst": sel.iloc[0], "median": sel.iloc[len(sel) // 2], "best": sel.iloc[-1]}

    device = torch.device("cpu")
    model = UNet(base_channels=width)
    model.load_state_dict(torch.load(f"runs/{run}/best.pt", map_location=device))
    patients = load_split("test")

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    for row, (name, c) in zip(axes, cases.items()):
        p = patients[c["patient"]]
        img = p[f"img_{c['phase']}"].astype(np.float32)
        gt = p[f"gt_{c['phase']}"]
        pred = keep_largest_component(predict_volume(model, p[f"img_{c['phase']}"], device))

        # First, middle and last slices containing LV cavity in the expert mask
        zs = np.where((gt == 3).any(axis=(1, 2)))[0]
        slices = [zs[0], zs[len(zs) // 2], zs[-1]]
        # Crop around the heart for readability
        ys, xs = np.where((gt > 0).any(axis=0))
        crop = (slice(max(ys.min() - 20, 0), ys.max() + 20),
                slice(max(xs.min() - 20, 0), xs.max() + 20))

        for k, z in enumerate(slices):
            overlay(row[2 * k], img[z][crop], gt[z][crop], f"slice {z}: expert")
            overlay(row[2 * k + 1], img[z][crop], pred[z][crop], f"slice {z}: model")
        row[0].text(-0.1, 0.5, f"{name}\n{c['patient']} {c['phase'].upper()}\n"
                    f"({c['group']})\nDice {c['dice_mean']:.3f}",
                    transform=row[0].transAxes, ha="right", va="center", fontsize=10)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3)
    plt.subplots_adjust(left=0.12, bottom=0.06, wspace=0.05, hspace=0.25)
    plt.savefig(os.path.join(OUT, f"qualitative_{run}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    accuracy_vs_cost()
    learning_curves()
    qualitative()
    print(f"Figures saved to {OUT}/")