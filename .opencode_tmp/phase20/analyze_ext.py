"""Analyze external rows in UAMS_v2.1 and how they link to phase18 staging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import p20_common as C
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

u21 = pd.read_parquet(C.PROJECT_ROOT / "outputs" / "UAMS_v2.1.parquet")
u21["PaperID_s"] = u21["PaperID"].astype(str).str.strip()
ext = u21[u21["PaperID_s"].eq("")].copy()
print("external rows:", len(ext))

print("\n== Sample external yield row ==")
print(ext[ext["Variable"] == "Yield_per_Hectare"].head(5).T.to_string())

print("\n== external rows: non-empty column check (metadata) ==")
meta_cols = ["DOI", "OriginalPaperID", "SourceSystem", "ExperimentID", "TreatmentID",
             "SourceTable", "Caption", "ConfidenceScore", "ReliabilityScore"]
for c in meta_cols:
    if c in ext.columns:
        n = ext[c].notna().sum()
        print(f"  {c:<18} nonnull={n}  uniq={ext[c].nunique()}  sample={list(ext[c].dropna().unique()[:4])}")

# staging linkage
stg = pd.read_parquet(C.PROJECT_ROOT / "outputs" / "phase18" / "staging" / "external_observations.parquet")
print("\n== staging: Yield rows sample ==")
sy = stg[stg["Variable"] == "yield_per_hectare"]
print(sy.head(3).T.to_string())
print("\n  staging yield rows:", len(sy), " countries:", sy["Country"].nunique())
print("\n  staging rows with Country non-null:", stg["Country"].notna().sum())

# Check if external rows have a way to know country: look at Yield value magnitude vs staging
print("\n== compare UAMS_v2.1 external yield values to staging ==")
u_ext_y = ext[ext["Variable"] == "Yield_per_Hectare"]["NormalizedValue"].astype(float)
s_y = stg[stg["Variable"] == "yield_per_hectare"]["Value"].astype(float)
print("  uams_v2.1 external yield: n=", len(u_ext_y), " range=", u_ext_y.min(), u_ext_y.max())
print("  staging yield:            n=", len(s_y), " range=", s_y.min(), s_y.max())

# Weather/soil external rows: what identifies their location in uams_v2.1?
print("\n== weather/soil external rows in uams_v2.1 ==")
wsext = ext[ext["Variable"].isin(["Average_Temperature", "Soil_pH", "Organic_Carbon", "Rainfall", "Nitrogen"])]
print(wsext[["ObservationID", "Variable", "NormalizedValue", "OriginalUnit", "ConfidenceScore"]].to_string())
