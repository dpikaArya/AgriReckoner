"""Phase 20 read-only inspection of key project inputs (no writes outside reports)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import p20_common as C
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

print("PYTHON:", sys.version)
print("PROJECT_ROOT:", C.PROJECT_ROOT)


def peek(name, df, n=3):
    print("\n" + "=" * 100)
    print(name, "shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(n).to_string())


def peek_parquet(rel):
    p = C.PROJECT_ROOT / rel
    if not p.exists():
        print("\nMISSING:", rel)
        return
    df = pd.read_parquet(p)
    print("\n" + "=" * 100)
    print(rel, "shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(2).to_string())


for rel in [
    "outputs/UAMS_v2.parquet",
    "outputs/UAMS_v2.1.parquet",
    "outputs/phase18/staging/external_observations.parquet",
    "outputs/phase19/observation_missingness.parquet",
    "outputs/phase19/feature_information_gain.parquet",
    "outputs/phase19/leakage_audit.parquet",
    "outputs/phase19/model_readiness.json",
    "outputs/phase19/model_targets.json",
    "outputs/phase19/predictor_sets.json",
    "outputs/phase19/version_manifest.json",
]:
    if str(rel).endswith(".json"):
        import json

        p = C.PROJECT_ROOT / rel
        if p.exists():
            print("\n==== " + rel + " ====")
            print(json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=1)[:4000])
        else:
            print("\nMISSING:", rel)
    else:
        peek_parquet(rel)
