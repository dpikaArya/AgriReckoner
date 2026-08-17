import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Enrichment")

from agri_ai_agent.config.schema import UAMS_COLUMNS  # noqa: E402
from agri_ai_agent.external_data.column_mapper import map_column  # noqa: E402
from agri_ai_agent.external_data.connector_manager import ConnectorManager  # noqa: E402
from agri_ai_agent.external_data.data_enricher import enrich_master  # noqa: E402
from agri_ai_agent.external_data.registry_db import DatasetRegistry  # noqa: E402


def load_master_datasets(data_dir: Path) -> pd.DataFrame:
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


def main():
    settings = type(
        "Settings",
        (),
        {
            "DATA_DIR": BASE_DIR / "data" / "master_datasets",
            "EXTERNAL_DATA_DIR": BASE_DIR / "external_data",
            "DATASET_REGISTRY_PATH": BASE_DIR / "database" / "dataset_registry.sqlite",
            "OUTPUT_DIR": BASE_DIR / "outputs",
        },
    )()
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load master datasets
    log.info("=" * 60)
    log.info("STEP 1: Load Master Datasets")
    log.info("=" * 60)
    master_df = load_master_datasets(settings.DATA_DIR)
    if master_df.empty:
        log.error("No master datasets found — aborting")
        sys.exit(1)
    log.info("Master has columns: %s", list(master_df.columns)[:20])

    # Map master dataset columns to UAMS canonical names where possible
    uams_rename = {}
    unmapped_master = []
    for col in master_df.columns:
        mapped = map_column("_master", col)
        if mapped and mapped != col:
            uams_rename[col] = mapped
        elif not mapped:
            unmapped_master.append(col)
    if uams_rename:
        master_df = master_df.rename(columns=uams_rename)
        log.info("Mapped %d master columns to UAMS names: %s", len(uams_rename), uams_rename)
    if unmapped_master:
        log.info("Unmapped master columns (%d): %s", len(unmapped_master), unmapped_master[:10])

    # Step 2: Initialize ConnectorManager
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: Discover & Health-Check External Sources")
    log.info("=" * 60)
    registry = DatasetRegistry(settings.DATASET_REGISTRY_PATH)
    manager = ConnectorManager(
        registry=registry,
        download_dir=settings.EXTERNAL_DATA_DIR,
        max_workers=2,
    )
    available = manager.list_sources()
    log.info("Available sources: %s", available)

    health_results = manager.health_checks()
    healthy = sorted([s for s, h in health_results.items() if h.is_healthy])
    unhealthy = sorted([s for s, h in health_results.items() if not h.is_healthy])
    log.info("Healthy: %s", healthy)
    log.info("Unhealthy: %s", unhealthy)
    for s, h in health_results.items():
        log.info(
            "  %s: healthy=%s latency=%.1fms discovered=%d error=%s",
            s,
            h.is_healthy,
            h.latency_ms,
            h.discovered_count,
            h.error or "none",
        )

    if not healthy:
        log.error("No healthy sources — aborting")
        registry.close()
        sys.exit(1)

    # Step 3: Download data from healthy sources (filter by discovery limit)
    log.info("")
    log.info("=" * 60)
    log.info("STEP 3: Download External Data")
    log.info("=" * 60)
    max_discovery = int(os.environ.get("AGRI_MAX_DISCOVERY_PER_SOURCE", "10"))
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
        log.warning("No quick-sync sources after filtering")
        registry.close()
        sys.exit(1)

    t0 = time.perf_counter()
    packages_by_source, run_logs = manager.run_all(sources=quick_sources)
    elapsed = time.perf_counter() - t0
    log.info("Downloaded in %.1fs", elapsed)

    total_downloaded = sum(len(v) for v in packages_by_source.values())
    total_valid = sum(1 for pkgs in packages_by_source.values() for p in pkgs if p.is_valid)
    log.info("Packages: %d downloaded, %d valid", total_downloaded, total_valid)
    for src, pkgs in packages_by_source.items():
        log.info("  %s: %d package(s)", src, len(pkgs))
        for p in pkgs:
            log.info(
                "    - %s (valid=%s, rows=%d, cols=%d)",
                p.resource_id,
                p.is_valid,
                p.row_count,
                p.column_count,
            )

    # Step 4: Enrich master datasets via spatial/crop join
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4: Enrich Master Dataset (Spatial/Crop Joins)")
    log.info("=" * 60)
    if packages_by_source:
        enriched_df = enrich_master(master_df, packages_by_source)
        log.info(
            "Enriched: %d rows x %d columns (was %d cols)",
            len(enriched_df),
            len(enriched_df.columns),
            len(master_df.columns),
        )

        # Show new columns
        new_cols = [c for c in enriched_df.columns if c not in master_df.columns]
        if new_cols:
            log.info("New columns from enrichment (%d): %s", len(new_cols), new_cols)
        else:
            log.info("No new columns added — columns matched existing UAMS columns")
    else:
        log.warning("No packages downloaded — skipping enrichment")
        enriched_df = master_df

    # Step 5: Save enriched schema
    log.info("")
    log.info("=" * 60)
    log.info("STEP 5: Save Enriched Universal Schema")
    log.info("=" * 60)
    csv_path = settings.OUTPUT_DIR / "Universal_Agricultural_Schema.csv"
    xlsx_path = settings.OUTPUT_DIR / "Universal_Agricultural_Schema.xlsx"
    enriched_df.to_csv(csv_path, index=False)
    log.info(
        "Saved CSV: %s (%d cols, %d rows)", csv_path, len(enriched_df.columns), len(enriched_df)
    )
    try:
        enriched_df.to_excel(xlsx_path, index=False)
        log.info("Saved XLSX: %s", xlsx_path)
    except Exception as e:
        log.warning("XLSX save failed: %s", e)

    # Summary
    uams_cols_present = [c for c in UAMS_COLUMNS if c in enriched_df.columns]
    log.info("")
    log.info("=" * 60)
    log.info("ENRICHMENT COMPLETE")
    log.info("=" * 60)
    log.info("  Master rows: %d", len(master_df))
    log.info("  Master columns: %d", len(master_df.columns))
    log.info("  Enriched rows: %d", len(enriched_df))
    log.info("  Enriched columns: %d", len(enriched_df.columns))
    log.info("  UAMS columns matched: %d / %d", len(uams_cols_present), len(UAMS_COLUMNS))
    log.info("  New columns from external data: %d", len(new_cols) if packages_by_source else 0)
    log.info("  External packages: %d downloaded, %d valid", total_downloaded, total_valid)
    log.info("  Sources used: %s", list(packages_by_source.keys()))

    registry.close()


if __name__ == "__main__":
    main()
