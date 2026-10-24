"""Aggregate test results: Dice with patient-level bootstrap CIs, seed variability, cost."""
import glob
import json

import numpy as np
import pandas as pd

N_BOOT = 2000
REF_WIDTH = 64  # "standard" U-Net used as reference
STRUCTS = ["rv", "myo", "lv"]
COLS = [f"dice_{s}" for s in STRUCTS] + ["dice_mean"]

df = pd.read_csv("results/test_per_volume.csv")
df["dice_mean"] = df[[f"dice_{s}" for s in STRUCTS]].mean(axis=1)

# One score per patient: mean over ED/ES and over the 3 seeds of each width
per_patient = df.groupby(["post", "width", "patient"])[COLS].mean()
patients = sorted(df["patient"].unique())

# Same bootstrap resamples for every model -> paired comparisons
rng = np.random.default_rng(0)
boot_idx = rng.integers(0, len(patients), size=(N_BOOT, len(patients)))


def bootstrap_ci(values):
    """95% CI of the mean of a per-patient array, by resampling patients."""
    means = values[boot_idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


rows = []
for (post, width), g in per_patient.groupby(level=["post", "width"]):
    g = g.droplevel(["post", "width"]).loc[patients]
    row = {"post": post, "width": width}
    for c in COLS:
        v = g[c].to_numpy()
        row[c] = v.mean()
        row[f"{c}_lo"], row[f"{c}_hi"] = bootstrap_ci(v)

    # Paired difference with the reference model, patient by patient
    ref = per_patient.loc[(post, REF_WIDTH)].loc[patients, "dice_mean"].to_numpy()
    diff = g["dice_mean"].to_numpy() - ref
    row["diff_vs_ref"] = diff.mean()
    row["diff_lo"], row["diff_hi"] = bootstrap_ci(diff)

    # Training variability: std of the 3 runs' mean Dice
    sel = df[(df["post"] == post) & (df["width"] == width)]
    row["seed_std"] = sel.groupby("seed")["dice_mean"].mean().std()
    rows.append(row)
table = pd.DataFrame(rows)

# Cost: parameters, training time (GPU), inference time (CPU)
summaries = pd.DataFrame([json.load(open(f)) for f in glob.glob("runs/*/summary.json")])
cost = summaries.groupby("width").agg(params=("params", "first"),
                                      train_min=("train_time_s", lambda s: s.mean() / 60))
cost["ms_per_slice_cpu"] = df[df["post"] == "raw"].groupby("width")["ms_per_slice"].mean()
table = table.merge(cost, left_on="width", right_index=True)
table.to_csv("results/summary_test.csv", index=False)

for post in ("raw", "lcc"):
    print(f"\n=== post-processing: {post} ===")
    for _, r in table[table["post"] == post].sort_values("width").iterrows():
        print(f"w{int(r.width):<3}{r.params / 1e6:6.2f} M | train {r.train_min:4.1f} min | "
              f"{r.ms_per_slice_cpu:6.1f} ms/slice | "
              f"Dice {r.dice_mean:.3f} [{r.dice_mean_lo:.3f}, {r.dice_mean_hi:.3f}] | "
              f"RV {r.dice_rv:.3f} MYO {r.dice_myo:.3f} LV {r.dice_lv:.3f} | "
              f"vs w{REF_WIDTH} {r.diff_vs_ref:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}] | "
              f"seed sd {r.seed_std:.3f}")