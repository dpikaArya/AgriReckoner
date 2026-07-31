"""
Universal Agricultural Schema Generator (Merge Agent)
======================================================
Generates the Universal Agricultural Metadata Schema (UAMS v2.0)
by merging master datasets and enriching with external data sources
(NASA POWER, SoilGrids, etc.) via spatial/crop key-based joins.

Usage:
    python universal_schema_generator.py
    python universal_schema_generator.py --use-external-data
"""

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "master_datasets"
EXTERNAL_DATA_DIR = BASE_DIR / "external_data"
DATASET_REGISTRY_PATH = BASE_DIR / "database" / "dataset_registry.sqlite"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_master_datasets(data_dir: Path) -> pd.DataFrame:
    """Load all .xlsx master datasets into a single DataFrame."""
    all_frames = []
    if data_dir.exists():
        for f in sorted(data_dir.glob("*.xlsx")):
            try:
                df = pd.read_excel(f)
                df["_source_file"] = f.name
                all_frames.append(df)
                log.info("  Loaded %s: %d rows x %d cols", f.name, len(df), len(df.columns))
            except Exception as e:
                log.warning("  Failed %s: %s", f.name, e)
    combined = (
        pd.concat(all_frames, ignore_index=True, sort=False) if all_frames else pd.DataFrame()
    )
    log.info("Combined master: %d rows x %d cols", len(combined), len(combined.columns))
    return combined


def map_columns_to_uams(df: pd.DataFrame) -> pd.DataFrame:
    """Map dataset columns to UAMS canonical names via VARIANT_MAP."""
    from agri_ai_agent.external_data.column_mapper import map_column

    uams_rename = {}
    unmapped = []
    for col in df.columns:
        mapped = map_column("_master", col)
        if mapped and mapped != col:
            uams_rename[col] = mapped
        elif not mapped:
            unmapped.append(col)
    if uams_rename:
        df = df.rename(columns=uams_rename)
        log.info("Mapped %d columns to UAMS names", len(uams_rename))
    if unmapped:
        log.info("Unmapped master columns (%d): %s", len(unmapped), unmapped[:10])
    return df


def enrich_with_external_data(df: pd.DataFrame) -> pd.DataFrame:
    """Download external data from healthy sources and enrich via key-based joins."""
    from agri_ai_agent.config.schema import UAMS_COLUMNS
    from agri_ai_agent.external_data.connector_manager import ConnectorManager
    from agri_ai_agent.external_data.data_enricher import enrich_master
    from agri_ai_agent.external_data.registry_db import DatasetRegistry

    registry = DatasetRegistry(DATASET_REGISTRY_PATH)
    manager = ConnectorManager(
        registry=registry,
        download_dir=EXTERNAL_DATA_DIR,
        max_workers=2,
    )

    available = manager.list_sources()
    log.info("External sources: %s", available)

    health_results = manager.health_checks()
    healthy = sorted([s for s, h in health_results.items() if h.is_healthy])
    unhealthy = sorted([s for s, h in health_results.items() if not h.is_healthy])
    log.info("Healthy: %s", healthy)
    log.info("Unhealthy: %s", unhealthy)

    if not healthy:
        log.warning("No healthy external sources — skipping enrichment")
        registry.close()
        return df

    max_discovery = int(os.environ.get("AGRI_MAX_DISCOVERY_PER_SOURCE", "15"))
    quick_sources = []
    for s in healthy:
        h = health_results.get(s)
        if h and h.discovered_count <= max_discovery:
            quick_sources.append(s)
        else:
            log.info(
                "Skipping %s (%d datasets > %d max)",
                s,
                h.discovered_count if h else -1,
                max_discovery,
            )

    if not quick_sources:
        log.warning("No quick-sync sources — skipping enrichment")
        registry.close()
        return df

    log.info("Downloading from %s ...", quick_sources)
    packages_by_source, _ = manager.run_all(sources=quick_sources)
    total_valid = sum(1 for pkgs in packages_by_source.values() for p in pkgs if p.is_valid)
    log.info("Downloaded: %d valid packages", total_valid)

    if packages_by_source:
        result = enrich_master(df, packages_by_source)
        uams_present = [c for c in UAMS_COLUMNS if c in result.columns]
        log.info(
            "Enriched: %d rows x %d cols | UAMS matched: %d/%d",
            len(result),
            len(result.columns),
            len(uams_present),
            len(UAMS_COLUMNS),
        )
        new_cols = [c for c in result.columns if c not in df.columns]
        if new_cols:
            log.info("New external columns (%d): %s", len(new_cols), new_cols)
    else:
        result = df

    registry.close()
    return result


def run_pipeline(input_path=None, use_external_data=False):
    """
    Run the Universal Schema generation pipeline.

    Parameters
    ----------
    input_path : str or Path, optional
        If None, auto-detect .xlsx files in data/master_datasets/.
    use_external_data : bool
        If True, run external data source connectors and enrich
        the master dataset via spatial/crop key-based joins.

    Returns
    -------
    pd.DataFrame : the merged, mapped, and enriched schema.
    """
    if use_external_data:
        return _run_enrichment_pipeline(DATA_DIR)
    return _run_legacy_pipeline(input_path)


def _run_legacy_pipeline(input_path=None):
    """Original single-file ADES orchestrator pathway."""
    from agri_ai_agent.config.settings import AgriAISettings as ADESSettings
    from agri_ai_agent.orchestrator import Orchestrator
    from agri_ai_agent.utils.logging_utils import setup_logging

    settings = ADESSettings()
    setup_logging(level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

    if input_path is None:
        xlsx_files = sorted(settings.DATA_DIR.glob("*.xlsx"))
        if xlsx_files:
            input_path = str(xlsx_files[0])
        else:
            log.error("No .xlsx files found in %s", DATA_DIR)
            sys.exit(1)

    orch = Orchestrator(settings=settings)
    df = orch.run(filepath=str(input_path))
    return df


def _run_enrichment_pipeline(data_dir: Path) -> pd.DataFrame:
    """Load all master datasets, map to UAMS, enrich with external data."""
    log.info("=" * 60)
    log.info("UAMS ENRICHMENT PIPELINE")
    log.info("=" * 60)

    t0 = time.perf_counter()

    # Step 1: Load master datasets
    log.info("")
    log.info("--- Step 1: Load Master Datasets ---")
    df = load_master_datasets(data_dir)
    if df.empty:
        log.error("No master datasets found — aborting")
        sys.exit(1)

    # Step 2: Map columns to UAMS canonical names
    log.info("")
    log.info("--- Step 2: Map Columns to UAMS ---")
    df = map_columns_to_uams(df)

    # Step 3: Enrich with external data
    log.info("")
    log.info("--- Step 3: External Data Enrichment ---")
    df = enrich_with_external_data(df)

    elapsed = time.perf_counter() - t0
    log.info("")
    log.info("Pipeline complete in %.1fs: %d rows x %d cols", elapsed, len(df), len(df.columns))
    return df


def save_schema(df: pd.DataFrame, output_path=None):
    """Save the Universal Schema DataFrame to CSV and XLSX."""
    if output_path is None:
        output_path = OUTPUT_DIR / "Universal_Agricultural_Schema"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    xlsx_path = output_path.with_suffix(".xlsx")
    df.to_csv(csv_path, index=False)
    log.info("Schema saved to %s (%d cols, %d rows)", csv_path, len(df.columns), len(df))
    try:
        df.to_excel(xlsx_path, index=False)
        log.info("Schema saved to %s", xlsx_path)
    except Exception as e:
        log.warning("Could not save XLSX: %s", e)
    return csv_path, xlsx_path


if __name__ == "__main__":
    try:
        use_ext = "--use-external-data" in sys.argv
        path = None
        for arg in sys.argv[1:]:
            if arg != "--use-external-data" and arg.startswith("--"):
                continue
            elif arg != "--use-external-data" and not arg.startswith("-"):
                path = arg
                break
        df = run_pipeline(path, use_external_data=use_ext)
        save_schema(df)
        print("\nUniversal Schema generated successfully.")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  External data: {'enabled [OK]' if use_ext else 'disabled'}")
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        sys.exit(1)
