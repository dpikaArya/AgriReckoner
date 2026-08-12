"""Analyze yield observations and their spatial/temporal metadata."""
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

lit = u21[u21["PaperID_s"].ne("")]
ext = u21[u21["PaperID_s"].eq("")]
print("literature rows:", len(lit), " external rows:", len(ext))

# variables used as spatial metadata
meta_vars = ["Latitude", "Longitude", "Location", "Country", "State", "Site", "Institution",
             "Season", "Sowing_Date", "Month", "Year", "Altitude"]
print("\n== spatial metadata variables in literature rows ==")
for v in meta_vars:
    sub = lit[lit["Variable"] == v]
    print(f"  {v:<14} rows={len(sub)}  sample={list(sub['NormalizedValue'].dropna().unique()[:6])}")

# yield in literature rows
lit_y = lit[lit["Variable"] == "Yield_per_Hectare"]
print("\n== Literature Yield_per_Hectare rows ==")
print("  rows:", len(lit_y), " papers:", lit_y["PaperID"].nunique(), " experiments:", lit_y["ExperimentID"].nunique(),
      " treatments:", lit_y["TreatmentID"].nunique())
print("  years:", sorted(lit_y["Year"].dropna().unique()))
print("  crops:", lit_y["Crop"].value_counts().head(15).to_dict())

# external yield rows
ext_y = ext[ext["Variable"] == "Yield_per_Hectare"]
print("\n== External Yield_per_Hectare rows ==")
print("  rows:", len(ext_y), " countries:", ext_y["Crop"].nunique())

# For literature yield rows: do they have any coords anywhere in the paper?
print("\n== spatial coords presence per literature yield paper ==")
lit_papers = set(lit_y["PaperID"])
coords = lit[lit["Variable"].isin(["Latitude", "Longitude"])]
coord_papers = set(coords["PaperID"])
print("  papers with yield:", len(lit_papers), " papers with lat/lon var:", len(coord_papers))
print("  overlap:", len(lit_papers & coord_papers))

# check variable list of spatial vars in ext rows
print("\n== external rows variable types ==")
print(ext["Variable"].value_counts().head(25).to_dict())

# Examine a specific literature yield row's experiment variables
print("\n== sample literature yield paper experiments ==")
for pid in list(lit_papers)[:3]:
    sub = lit[lit["PaperID"] == pid]
    print("  PAPER:", pid, " rows:", len(sub))
    print("   vars:", sub["Variable"].value_counts().head(20).to_dict())
