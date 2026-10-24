"""Ejection fraction analysis: predicted LVEF vs reference LVEF on the test set."""
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/testing"
N_BOOT = 2000
EF_THRESHOLD = 40  # %, reduced-EF cut-off used in ACDC group definitions
FIG_WIDTHS = (16, 64)  # widths shown on the Bland-Altman figure


def read_cfg(path):
    """Parse an ACDC Info.cfg file into a dict."""
    cfg = {}
    with open(path) as f:
        for line in f:
            key, value = line.split(":")
            cfg[key.strip()] = value.strip()
    return cfg


def reference_volumes(pid):
    """LV cavity volumes (mL) at ED and ES from the raw expert masks, original resolution."""
    pdir = os.path.join(RAW_DIR, pid)
    cfg = read_cfg(os.path.join(pdir, "Info.cfg"))
    vols = {"patient": pid}
    for phase in ("ED", "ES"):
        nii = nib.load(os.path.join(pdir, f"{pid}_frame{int(cfg[phase]):02d}_gt.nii.gz"))
        voxel_ml = float(np.prod(nii.header.get_zooms()[:3])) / 1000
        vols[f"{phase.lower()}v_ref"] = (np.asarray(nii.dataobj) == 3).sum() * voxel_ml
    return vols


def ef(edv, esv):
    return 100 * (edv - esv) / edv


# Predicted LV volumes (post-processed masks), one row per run x patient
df = pd.read_csv("results/test_per_volume.csv")
df = df[df["post"] == "lcc"]
key = ["width", "seed", "patient"]
ed = df[df["phase"] == "ed"].set_index(key)
es = df[df["phase"] == "es"].set_index(key)
pred = pd.DataFrame({
    "group": ed["group"],
    "edv_pred": ed["vol_pred_lv"], "esv_pred": es["vol_pred_lv"],
    "edv_resampled": ed["vol_gt_lv"], "esv_resampled": es["vol_gt_lv"],
}).reset_index()
pred["ef_pred"] = ef(pred["edv_pred"], pred["esv_pred"])

# Reference volumes and EF from the raw masks
patients = sorted(pred["patient"].unique())
ref = pd.DataFrame([reference_volumes(p) for p in patients])
ref["ef_ref"] = ef(ref["edv_ref"], ref["esv_ref"])
pred = pred.merge(ref, on="patient")

# Sanity check: EF computed on resampled expert masks vs raw expert masks
ef_resampled = ef(pred["edv_resampled"], pred["esv_resampled"])
print(f"Resampling effect on reference EF: "
      f"mean |diff| = {(ef_resampled - pred['ef_ref']).abs().mean():.2f} points\n")

rng = np.random.default_rng(0)
boot_idx = rng.integers(0, len(patients), size=(N_BOOT, len(patients)))

rows, per_width = [], {}
for width, g in pred.groupby("width"):
    g = g.assign(abs_err=(g["ef_pred"] - g["ef_ref"]).abs(),
                 edv_err=(g["edv_pred"] - g["edv_ref"]).abs(),
                 esv_err=(g["esv_pred"] - g["esv_ref"]).abs())
    # One value per patient, averaged over the 3 seeds
    p = g.groupby("patient").agg(
        group=("group", "first"), ef_ref=("ef_ref", "first"), ef_pred=("ef_pred", "mean"),
        abs_err=("abs_err", "mean"), edv_err=("edv_err", "mean"), esv_err=("esv_err", "mean"),
    ).loc[patients]
    per_width[width] = p

    mae_boot = p["abs_err"].to_numpy()[boot_idx].mean(axis=1)
    diff = p["ef_pred"] - p["ef_ref"]
    bias, sd = diff.mean(), diff.std()
    misclassified = ((p["ef_pred"] < EF_THRESHOLD) != (p["ef_ref"] < EF_THRESHOLD)).sum()
    rows.append({
        "width": width,
        "ef_mae": p["abs_err"].mean(),
        "ef_mae_lo": np.percentile(mae_boot, 2.5), "ef_mae_hi": np.percentile(mae_boot, 97.5),
        "bias": bias, "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd,
        "corr": np.corrcoef(p["ef_pred"], p["ef_ref"])[0, 1],
        "edv_mae_ml": p["edv_err"].mean(), "esv_mae_ml": p["esv_err"].mean(),
        "misclassified_40": int(misclassified),
    })

table = pd.DataFrame(rows)
os.makedirs("results/figures", exist_ok=True)
table.to_csv("results/ef_summary.csv", index=False)
for _, r in table.iterrows():
    print(f"w{int(r.width):<3}| EF MAE {r.ef_mae:4.2f} [{r.ef_mae_lo:4.2f}, {r.ef_mae_hi:4.2f}] pts | "
          f"bias {r.bias:+5.2f} | LoA [{r.loa_lo:+5.2f}, {r.loa_hi:+5.2f}] | r = {r['corr']:.3f} | "
          f"EDV err {r.edv_mae_ml:4.1f} mL | ESV err {r.esv_mae_ml:4.1f} mL | "
          f"wrong side of {EF_THRESHOLD}%: {r.misclassified_40}/{len(patients)}")

# Bland-Altman plots
fig, axes = plt.subplots(1, len(FIG_WIDTHS), figsize=(6 * len(FIG_WIDTHS), 5), sharey=True)
for ax, width in zip(np.atleast_1d(axes), FIG_WIDTHS):
    p = per_width[width]
    mean = (p["ef_pred"] + p["ef_ref"]) / 2
    diff = p["ef_pred"] - p["ef_ref"]
    r = table.set_index("width").loc[width]
    for grp, sub in p.groupby("group"):
        ax.scatter(mean[sub.index], diff[sub.index], label=grp, s=30)
    ax.axhline(r["bias"], color="black", label=f"bias {r['bias']:+.2f}")
    for lim in (r["loa_lo"], r["loa_hi"]):
        ax.axhline(lim, color="gray", linestyle="--")
    ax.set_title(f"U-Net width {width} ({EF_THRESHOLD}% threshold errors: {int(r['misclassified_40'])})")
    ax.set_xlabel("Mean of predicted and reference LVEF (%)")
ax0 = np.atleast_1d(axes)[0]
ax0.set_ylabel("Predicted − reference LVEF (points)")
ax0.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/ef_bland_altman.png", dpi=150)
print("\nFigure saved to results/figures/ef_bland_altman.png")