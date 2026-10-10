import glob, nibabel as nib, pandas as pd

rows = []
for f in sorted(glob.glob("data/raw/*/patient*/patient*_frame01.nii.gz")):
    img = nib.load(f)
    rows.append({"file": f, "shape": img.shape, "spacing": img.header.get_zooms()})
df = pd.DataFrame(rows)
print(df["shape"].value_counts().head(10))
print(df["spacing"].value_counts().head(10))