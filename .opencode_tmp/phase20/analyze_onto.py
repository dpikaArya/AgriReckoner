"""Inspect ontology aliases, phase17 mapping, rejected staging rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import p20_common as C
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

p17 = C.PROJECT_ROOT / "outputs" / "phase17"
print("== phase17 ontology_aliases.parquet ==")
oa = pd.read_parquet(p17 / "ontology_aliases.parquet")
print("shape:", oa.shape, "cols:", list(oa.columns))
print(oa.head(5).to_string())

print("\n== phase17 source_uams_mapping.parquet ==")
sm = pd.read_parquet(p17 / "source_uams_mapping.parquet")
print("shape:", sm.shape, "cols:", list(sm.columns))
print(sm.head(3).to_string())

print("\n== phase17 resolution_records.parquet ==")
rr = pd.read_parquet(p17 / "resolution_records.parquet")
print("shape:", rr.shape, "cols:", list(rr.columns))
print(rr.head(3).to_string())

print("\n== rejected staging rows (UAMS_Column empty) ==")
stg = pd.read_parquet(C.PROJECT_ROOT / "outputs" / "phase18" / "staging" / "external_observations.parquet")
rej = stg[stg["UAMS_Column"].isna() | (stg["UAMS_Column"] == "")]
print("rejected rows:", len(rej))
print("Variable counts:", rej["Variable"].value_counts().to_dict())
print("Dataset_ID sample:", rej["Dataset_ID"].dropna().unique()[:10])
print("\nRow sample:")
print(rej.head(3).T.to_string())
