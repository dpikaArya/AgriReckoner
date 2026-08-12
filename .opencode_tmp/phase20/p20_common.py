"""Phase 20 shared configuration, project-root enforcement, path safety,
checkpointing, hashing, and I/O helpers.

Phase 20: Observation Linkage, Predictor Completion, Information Gain
Enrichment, and Leakage Safe Model Ready Dataset Construction.

Project-root safety: every path handled by Phase 20 must resolve beneath the
single absolute PROJECT_ROOT. The module refuses to construct or access paths
outside it and never scans other drives.

PROJECT_ROOT is pinned to the confirmed project path (task title):
    F:\\Agentic AI Frameworks\\Agriculture AI Framework\\Agriculture Intelligence Framework3
On machines where that pinned path is absent (e.g. CI runners), the repository
root is discovered from this module's location (<root>/.opencode_tmp/phase20/).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PINNED_ROOT = Path(
    r"F:\Agentic AI Frameworks\Agriculture AI Framework\Agriculture Intelligence Framework3"
).resolve()


def _discover_project_root() -> Path:
    """Locate the repository root from this module's location.

    The repository layout places this module under <root>/.opencode_tmp/phase20/,
    so walking up to the first ancestor that contains config/phase20.yaml yields
    the root on any machine (e.g. CI runners without the pinned F: drive).
    """
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "config" / "phase20.yaml").is_file():
            return candidate
    return start.parent.parent


PROJECT_ROOT = PINNED_ROOT if PINNED_ROOT.exists() else _discover_project_root()

assert PROJECT_ROOT.is_dir(), f"PROJECT_ROOT not a directory: {PROJECT_ROOT}"
if PROJECT_ROOT == PINNED_ROOT:
    assert PROJECT_ROOT.drive.upper() == "F:", f"PROJECT_ROOT not on F: drive: {PROJECT_ROOT}"
    assert PROJECT_ROOT.name == "Agriculture Intelligence Framework3", (
        f"PROJECT_ROOT name mismatch: {PROJECT_ROOT.name}"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "outputs" / "phase20"
REPORTS = PROJECT_ROOT / "reports" / "phase20"
TMP = PROJECT_ROOT / ".opencode_tmp" / "phase20"
LOGS = OUT / "logs"
CHECKPOINT_FILE = PROJECT_ROOT / ".checkpoints" / "phase20_checkpoints.json"
RAG_OUT = OUT / "rag_sync"
ONTO_EXT = TMP / "phase20_ontology_extensions.yaml"
PATH_SAFETY_REPORT = REPORTS / "path_safety_report.json"

for _d in (OUT, REPORTS, TMP, LOGS, RAG_OUT):
    _d.mkdir(parents=True, exist_ok=True)

CONFIG = PROJECT_ROOT / "config" / "phase20.yaml"

PHASE20_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# default configuration (merged with config/phase20.yaml `phase20:` section)
# ---------------------------------------------------------------------------

DEFAULTS: dict = {
    "inputs": {},
    "dry_run": True,
    "ontology": {
        "alias_min_confidence": 0.7,
        "max_aliases_per_variable": 100,
    },
    "spatial": {
        "exact_match_radius_km": 5.0,
        "station_match_radius_km": 50.0,
        "district_match_radius_km": 200.0,
    },
    "temporal": {
        "year_range": (1970, 2100),
        "default_growing_days": 120,
    },
    "readiness_classes": {
        "MODEL_READY_min": 0.80,
        "NEAR_READY_min": 0.60,
        "PARTIAL_min": 0.30,
    },
    "predictor_tiers": {
        "TIER_A": [
            "Yield",
            "Crop",
            "Location",
            "Year",
            "Season",
            "Rainfall",
            "Temperature",
            "Soil_pH",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
        ],
        "TIER_B": [
            "Solar_Radiation",
            "Organic_Carbon",
            "Soil_Moisture",
            "Texture",
            "Humidity",
            "Cultivar",
            "Plant_Population",
        ],
        "TIER_C": [
            "Soil_EC",
            "Irrigation",
            "Fertilizer_Type",
            "N_Dose",
            "P_Dose",
            "K_Dose",
            "Plant_Height_cm",
            "Days_to_Maturity",
        ],
    },
    "required_predictors": [
        "Yield",
        "Crop",
        "Location",
        "Year",
        "Season",
        "Rainfall",
        "Temperature",
        "Soil_pH",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
    ],
    "critical_predictors": [
        "Crop",
        "Location",
        "Year",
        "Season",
        "Rainfall",
        "Temperature",
        "Soil_pH",
    ],
    "splitting": {
        "primary": "study_grouped",
        "secondary": "location_grouped",
        "temporal": "temporal_validation",
        "min_temporal_years": 6,
        "n_splits": 5,
        "crop_stratified": True,
    },
    "information_gain": {
        "near_ready_boost": 1.0,
        "completion_weight": 0.3,
        "crop_coverage_weight": 0.15,
        "study_diversity_weight": 0.1,
        "source_weight": 0.1,
        "reliability_weight": 0.1,
        "spatial_weight": 0.05,
        "temporal_weight": 0.05,
        "feasibility_weight": 0.1,
        "cost_weight": 0.05,
    },
    "minimum_requirements": {
        "min_independent_studies": 5,
        "min_independent_locations": 5,
        "min_observations": 50,
        "min_predictor_completeness": 0.60,
        "min_crops": 3,
        "min_years": 3,
        "min_complete_observations_per_crop": 10,
        "status_when_not_met": "NOT_READY",
    },
    "gates": {
        "protected_unchanged": True,
        "no_uams_modification": True,
        "no_fabricated_external_data": True,
        "external_provenance_required": True,
        "spatial_confidence_required": True,
        "temporal_confidence_required": True,
        "derived_predictor_provenance_required": True,
        "no_target_leakage": True,
        "no_study_leakage": True,
        "location_leakage_safe": True,
        "ontology_recovery_auditable": True,
        "model_readiness_independent": True,
        "all_outputs_inside_root": True,
        "idempotent": True,
    },
    "rag": {
        "chunk_types_to_add": [
            "model_ready_observation",
            "linkage_metric",
            "phase20_metric",
        ],
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_chunks": 200,
    },
}


def _merge(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: Path | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    p = Path(path) if path else CONFIG
    if p.exists():
        try:
            user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            user = {}
        _merge(cfg, user.get("phase20") or {})
    return cfg


# ---------------------------------------------------------------------------
# path safety
# ---------------------------------------------------------------------------


def safe_resolve(path: Path | str) -> Path:
    """Resolve a path and guarantee it stays inside PROJECT_ROOT.

    Raises RuntimeError when the resolved path escapes PROJECT_ROOT.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if p != PROJECT_ROOT and not p.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"path escapes PROJECT_ROOT: {p}")
    return p


def is_inside_root(path: Path | str) -> bool:
    try:
        safe_resolve(path)
        return True
    except RuntimeError:
        return False


def audit_paths(config: dict) -> list[dict]:
    """Resolve every configured input/output path and report safety verdicts."""
    rows: list[dict] = []

    def _walk(node, prefix):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _walk(v, key)
            elif isinstance(v, (str, Path)):
                rows.append(_record_path(key, v))

    _walk(config, "config")
    for label, p in [
        ("outputs.phase20", OUT),
        ("reports.phase20", REPORTS),
        ("tmp.phase20", TMP),
        ("logs.phase20", LOGS),
        ("rag_sync.phase20", RAG_OUT),
        ("checkpoints.phase20", CHECKPOINT_FILE),
    ]:
        rows.append(_record_path(label, p))

    report = []
    for r in rows:
        report.append(
            {
                "configured_path": str(r["configured"]),
                "resolved_path": str(r["resolved"]),
                "inside_project_root": r["ok"],
                "access_decision": "ALLOW" if r["ok"] else "REJECT",
                "reason": r["reason"],
            }
        )
    return report


def _record_path(label: str, value: object) -> dict:
    try:
        if isinstance(value, Path):
            resolved = value.resolve()
        else:
            resolved = Path(str(value)).resolve()
        ok = resolved == PROJECT_ROOT or resolved.is_relative_to(PROJECT_ROOT)
        reason = "resolved within project root" if ok else "resolved OUTSIDE project root"
    except Exception as exc:  # pragma: no cover
        resolved = Path("UNRESOLVED")
        ok = False
        reason = f"path resolution failed: {exc}"
    return {"configured": value, "resolved": resolved, "ok": ok, "reason": reason}


# ---------------------------------------------------------------------------
# checkpointing (idempotent, resumable)
# ---------------------------------------------------------------------------


def load_checkpoints() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_checkpoints(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def mark_done(stage: str, payload=None) -> dict:
    state = load_checkpoints()
    state[stage] = {"completed_at": now_iso(), "status": "done"}
    if payload is not None:
        state[stage]["payload"] = payload
    save_checkpoints(state)
    return state


def is_done(stage: str) -> bool:
    return load_checkpoints().get(stage, {}).get("status") == "done"


def require_prior(stage: str) -> None:
    if not is_done(stage):
        raise RuntimeError(f"stage '{stage}' not completed")


# ---------------------------------------------------------------------------
# scalar + file helpers
# ---------------------------------------------------------------------------


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, (np.floating,)) and np.isnan(x):
        return ""
    return str(x).strip()


def clean(x) -> str:
    return s(x).lower()


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def now_full_iso() -> str:
    return datetime.now().isoformat()


def write_json(obj, path) -> Path:
    p = safe_resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def read_json(path, default=None):
    p = safe_resolve(path)
    if not p.exists():
        return default if default is not None else {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}


def to_parquet(df: pd.DataFrame, path) -> Path:
    p = safe_resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def to_excel(df: pd.DataFrame, path, sheet="Sheet1") -> Path:
    p = safe_resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet, index=False)
    return p


def write_excel_multi(sheets: dict, path) -> Path:
    p = safe_resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=str(name)[:31], index=False)
    return p


def log_msg(msg) -> str:
    stamp = now_iso()
    logf = LOGS / "phase20.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {msg}\n")
    return msg


def sha256_file(path) -> str:
    p = safe_resolve(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum_df(df: pd.DataFrame) -> str:
    """Deterministic content checksum of a DataFrame (schema + sorted rows)."""
    buf = df.copy()
    for c in buf.select_dtypes(include="object").columns:
        buf[c] = buf[c].fillna("").astype(str)
    buf = buf.fillna("")
    buf = buf.sort_values(by=buf.columns.tolist()).reset_index(drop=True)
    text = buf.to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_fingerprint(*parts) -> str:
    key = "|".join(str(x).strip().lower() for x in parts)
    key = re.sub(r"[^a-z0-9|]", "", key)
    return key if key else ""


# ---------------------------------------------------------------------------
# protected artifact registry + checksums
# ---------------------------------------------------------------------------

PROTECTED_INPUTS: list[str] = [
    "outputs/UAMS_v2.parquet",
    "outputs/UAMS_v2.1.parquet",
    "outputs/phase12/UAMS_v2_Migration_Report.parquet",
    "outputs/phase13",
    "outputs/phase14/harmonized_evidence.parquet",
    "outputs/phase14/meta_analysis_results.parquet",
    "outputs/phase14/effect_size_database.parquet",
    "outputs/phase14/ready_reckoner_knowledge_base.parquet",
    "outputs/phase14/knowledge_graph.json",
    "outputs/phase14_5a",
    "outputs/phase14_5b",
    "outputs/phase16",
    "outputs/phase16_1",
    "outputs/phase18",
    "outputs/phase19",
]


def protected_checksums(only_existing: bool = True) -> dict:
    out: dict = {}
    for rel in PROTECTED_INPUTS:
        p = PROJECT_ROOT / rel
        if p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file():
                    out[str(fp.relative_to(PROJECT_ROOT))] = sha256_file(fp)
        elif p.exists():
            out[rel] = sha256_file(p)
        elif not only_existing:
            out[rel] = None
    return out


def load_input(name: str, cfg: dict | None = None):
    """Resolve a configured input name and load its content.

    `name` may be dotted within config['inputs'].
    """
    cfg = cfg or load_config()
    key = str(name)
    if key.startswith("inputs."):
        key = key[len("inputs.") :]
    cur = cfg["inputs"]
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"config has no inputs.{key}")
        cur = cur[part]
    if isinstance(cur, dict):
        raise KeyError(f"inputs.{key} is a group, not a path")
    p = safe_resolve(str(cur))
    if not p.exists():
        raise FileNotFoundError(f"missing input '{name}': {p}")
    return _read_any(p)


def _read_any(p: Path):
    if p.suffix == ".json":
        return read_json(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError(f"unsupported input type: {p.suffix} for {p}")


def p_prev(module: str, package: str = "phase19"):
    """Import a helper module from an earlier phase's .opencode_tmp dir."""
    p = PROJECT_ROOT / ".opencode_tmp" / package
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return importlib.import_module(module)
