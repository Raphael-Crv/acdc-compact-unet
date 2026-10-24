import glob, json
import pandas as pd

rows = [json.load(open(f)) for f in glob.glob("runs/*/summary.json")]
df = pd.DataFrame(rows)[["width", "seed", "params", "best_val_dice", "best_epoch", "train_time_s"]]
print(df.sort_values(["width", "seed"]).to_string(index=False))
print(df.groupby("width")[["best_val_dice", "train_time_s"]].agg(["mean", "std"]).round(3))