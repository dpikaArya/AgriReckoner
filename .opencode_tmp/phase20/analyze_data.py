"""Deeper analysis of yield observations, spatial/temporal fields."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import p20_common as C
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

u21 = pd.read_parquet(C.PROJECT_ROOT / "outputs" / "UAMS_v2.1.parquet")
print("UAMS_v2.1 rows:", len(u21))
print("\n== Variable value counts (top 40) ==")
vc = u21["Variable"].value_counts()
print(vc.head(40))

yield_vars = [c for c in u21["Variable"].unique() if any(
    k in str(c).lower() for k in ["yield", "biomass", "height", "protein", "spad"])]
print("\n== Yield-ish variables present ==")
for v in sorted(yield_vars):
    sub = u21[u21["Variable"] == v]
    print(f"  {v:<30} rows={len(sub):<6} units={sorted(sub['NormalizedUnit'].dropna().unique())[:6]}")

print("\n== Spatial columns availability in UAMS_v2.1 ==")
for col in ["Latitude", "Longitude", "Location", "Country", "State", "Site", "Institution", "Season"]:
    if col in u21.columns:
        nonnull = u21[col].notna().sum()
        print(f"  {col:<15} nonnull={nonnull}  sample={list(u21[col].dropna().unique()[:5])}")
    else:
        print(f"  {col:<15} NOT A COLUMN")

print("\n== External observations staging ==")
ext = pd.read_parquet(C.PROJECT_ROOT / "outputs" / "phase18" / "staging" / "external_observations.parquet")
print("rows:", len(ext), "cols:", list(ext.columns))
print("\n== Source value counts ==")
print(ext["Source"].value_counts(dropna=False).head(30))
print("\n== Variable value counts ==")
print(ext["Variable"].value_counts().head(40))
print("\n== Spatial completeness ==")
print("  with lat:", ext["Location_Lat"].notna().sum(), " with lon:", ext["Location_Lon"].notna().sum())
print("  with year:", ext["Year"].notna().sum(), " with crop:", ext["Crop"].notna().sum())
print("\n== Mapping/validation status ==")
print(ext["Validation_Status"].value_counts(dropna=False))
print(ext["Is_Duplicate"].value_counts(dropna=False))
print("\n== Unique (lat,lon) pairs ==")
ll = ext.dropna(subset=["Location_Lat", "Location_Lon"])
print(ll.groupby([ll["Location_Lat"].round(2), ll["Location_Lon"].round(2)]).size().sort_values(ascending=False).head(20))
