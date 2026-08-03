"""End-to-end literature acquisition pipeline.

Stages:

    1. harvest     — incremental sync across literature connectors
    2. agri_sync   — incremental sync across agricultural data connectors
    3. classify    — enrich every record with design/variable/crop/original-study
    4. expand      — references, citations, related articles, PDF locations
    5. deduplicate — merge records into canonical papers (DOI + title)
    6. validate    — live DOI validation against Crossref/DataCite
    7. score       — quality score per canonical paper
    8. store       — write BibliographyDB (papers/authors/sources/refs/citations/pdfs)
    9. verify      — HEAD-verify PDF locations, record failures
   10. graph       — citation / author / experiment graphs
   11. export      — parquet/csv/xlsx/docx/html deliverables + UAMS schema update
   12. retrain     — trigger incremental retraining when thresholds are met

The pipeline never fails as a whole: connector-level errors are collected in
``PipelineReport.errors`` and the remaining stages continue.
"""

from __future__ import annotations

import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agri_ai_agent.config.schema import UAMS_COLUMNS
from agri_ai_agent.literature.agri.manager import (
    AgriculturalDataManager,
)
from agri_ai_agent.literature.classify import enrich_record
from agri_ai_agent.literature.config import LiteratureConfig
from agri_ai_agent.literature.connector import (
    LiteratureConnector,
    normalize_doi,
)
from agri_ai_agent.literature.dashboard import build_health_dashboard
from agri_ai_agent.literature.dedupe import (
    compute_quality_score,
    deduplicate_records,
    validate_doi_live,
)
from agri_ai_agent.literature.graphs import (
    build_author_graph,
    build_citation_graph,
    build_experiment_graph,
)
from agri_ai_agent.literature.models import (
    AgriculturalRecord,
    LiteratureRecord,
    SyncResult,
)
from agri_ai_agent.literature.registry import discover_literature_connectors
from agri_ai_agent.literature.state import ConnectorStateStore
from agri_ai_agent.literature.storage import (
    BibliographyDB,
    records_to_dataframe,
    write_csv,
    write_docx_report,
    write_excel,
    write_html_report,
    write_parquet,
    write_pdf_report,
)

logger = logging.getLogger(__name__)

# Agricultural variables -> UAMS column names for schema output mapping.
AGRI_VARIABLE_TO_UAMS: dict[str, str] = {
    "2m_temperature": "Average_Temperature",
    "temperature_2m": "Average_Temperature",
    "mean_temperature": "Average_Temperature",
    "total_precipitation": "Rainfall",
    "precipitation": "Rainfall",
    "rainfall": "Rainfall",
    "solar_radiation": "Solar_Radiation",
    "wind_speed": "Wind_Speed",
    "relative_humidity": "Humidity",
    "soil_moisture": "Soil_Moisture",
    "soil_temperature": "Average_Temperature",
    "evapotranspiration": "ET0",
    "ndvi": "NDVI",
    "evi": "EVI",
    "yield_per_hectare": "Yield_per_Hectare",
}

_VERIFY_PDF_CAP_DEFAULT = 200


@dataclass
class PipelineReport:
    """Aggregate result of a pipeline run."""

    run_id: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    duration_sec: float = 0.0

    sync_results: list[SyncResult] = field(default_factory=list)
    agri_sync_results: list[SyncResult] = field(default_factory=list)

    total_fetched: int = 0
    new_records: int = 0
    skipped_records: int = 0
    total_papers: int = 0
    verified_papers: int = 0
    duplicates: int = 0
    validated_dois: int = 0
    failed_pdfs: int = 0
    agri_records: int = 0

    outputs: dict[str, Path] = field(default_factory=dict)
    retrain_triggered: bool = False
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Run {self.run_id} finished in {self.duration_sec:.1f}s",
            f"  literature synced : {self.total_fetched} fetched, {self.new_records} new, "
            f"{self.skipped_records} cached",
            f"  papers             : {self.total_papers} (verified original: {self.verified_papers})",
            f"  duplicates merged  : {self.duplicates}",
            f"  DOIs validated     : {self.validated_dois}",
            f"  failed PDFs        : {self.failed_pdfs}",
            f"  agri records       : {self.agri_records}",
            f"  outputs written    : {len(self.outputs)}",
        ]
        if self.errors:
            lines.append(f"  errors             : {len(self.errors)}")
            for err in self.errors[:8]:
                lines.append(f"    - {err}")
        return "\n".join(lines)


def _paper_id(record: LiteratureRecord) -> str:
    import hashlib

    key = normalize_doi(record.canonical_doi or record.doi) or (
        (record.canonical_title or record.title or record.source_id).strip().lower()
    )
    return "PAPER_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class LiteraturePipeline:
    def __init__(
        self,
        config: LiteratureConfig | None = None,
        state: ConnectorStateStore | None = None,
        bibliography: BibliographyDB | None = None,
        connector_classes: list[type[LiteratureConnector]] | None = None,
        agri_connector_classes: list[type[Any]] | None = None,
    ):
        self.config = config or LiteratureConfig.from_env()
        self.state = state or ConnectorStateStore(self.config.state_db)
        self.bibliography = bibliography or BibliographyDB(self.config.bibliography_db)
        self._connector_classes = list(connector_classes or [])
        self._agri_connector_classes = list(agri_connector_classes or [])

        # merge external schema mapping over the built-in default
        self.variable_to_uams = dict(AGRI_VARIABLE_TO_UAMS)
        self.variable_to_uams.update(self.config.variable_to_uams or {})

        self.connectors: list[LiteratureConnector] = []
        self._lit_by_source: dict[str, LiteratureConnector] = {}
        self._build_literature_connectors()

        self.agri_manager = AgriculturalDataManager(self.config, self.state)
        self.agri_manager.build_connectors(
            self._agri_connector_classes or None
        )

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _build_literature_connectors(self) -> None:
        for cls in discover_literature_connectors(self._connector_classes):
            try:
                connector = cls(self.config, self.state)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to instantiate connector %s", cls)
                continue
            self.connectors.append(connector)
            self._lit_by_source[connector.source_name] = connector
            logger.info(
                "Connector %s: %s",
                connector.source_name,
                "enabled" if connector.enabled else "disabled",
            )

    @property
    def enabled_connectors(self) -> list[LiteratureConnector]:
        return [c for c in self.connectors if c.enabled]

    # ------------------------------------------------------------------ #
    # orchestration
    # ------------------------------------------------------------------ #
    def run(
        self,
        terms: tuple[str, ...] | None = None,
        run_id: str | None = None,
        *,
        full: bool = False,
        max_verify_pdfs: int | None = None,
        skip_agri: bool = False,
    ) -> PipelineReport:
        report = PipelineReport(run_id=run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}")
        started = time.monotonic()

        if terms:
            self.config.search_terms = terms

        # 1) harvest literature
        for connector in self.enabled_connectors:
            result = connector.incremental_sync(terms=terms)
            result.completed_at = datetime.now(timezone.utc).isoformat()
            report.sync_results.append(result)
            self.bibliography.log_crawl(result)
            report.total_fetched += result.fetched_records
            report.new_records += result.new_records
            report.skipped_records += result.skipped_records
            report.errors.extend(f"{connector.source_name}: {e}" for e in result.errors)

        # 2) harvest agricultural data
        if not skip_agri:
            for result in self.agri_manager.run_all():
                result.completed_at = datetime.now(timezone.utc).isoformat()
                report.agri_sync_results.append(result)
                report.agri_records += result.fetched_records

        # 3) load + classify
        records = self._load_literature_records()
        for record in records:
            enrich_record(record)

        # 4) expand (references/citations/related/pdfs)
        self._expand(records, full=full, report=report)

        # 5) deduplicate
        canonical, source_to_paper = deduplicate_records(records)
        for paper in canonical:
            paper.paper_id = _paper_id(paper)
            paper.canonical_title = paper.title
            paper.canonical_doi = normalize_doi(paper.doi) or paper.canonical_doi
        for (source, source_id), pid in list(source_to_paper.items()):
            source_to_paper[(source, source_id)] = (
                self._resolve_paper_id(canonical, pid)
            )
        for record in records:
            record.paper_id = source_to_paper.get((record.source, record.source_id))
        report.duplicates = sum(1 for r in records if r.duplicate_of)
        report.total_papers = len(canonical)

        # 6) validate DOIs (live) + recompute quality
        validated = 0
        for paper in canonical:
            if self.config.validate_dois_live and paper.doi:
                paper.valid_doi = validate_doi_live(
                    paper.doi,
                    source=self.config.doi_validation_source,
                    config=self.config,
                )
                validated += 1
            elif paper.doi:
                paper.valid_doi = normalize_doi(paper.doi) is not None
            paper.quality_score = compute_quality_score(paper, self.config.quality_weights)
        report.validated_dois = validated

        # 7) store bibliography
        for paper in canonical:
            self._store_paper(paper)
        for paper in canonical:
            self._store_relations(paper)

        # 8) verify PDFs
        report.failed_pdfs = self._verify_pdfs(canonical, cap=max_verify_pdfs)

        # 9) graphs
        citation_df = build_citation_graph(canonical)
        author_df = build_author_graph(canonical)
        experiment_df = build_experiment_graph(canonical)

        verified = [
            p for p in canonical
            if p.is_original_study and (p.quality_score or 0) >= self.config.quality_min_score
        ]
        report.verified_papers = len(verified)

        # 10) outputs
        agri_records = self.agri_manager.collect_records()
        report.agri_records = report.agri_records or len(agri_records)
        report.outputs = self._write_outputs(
            report=report,
            records=records,
            canonical=canonical,
            verified=verified,
            source_to_paper=source_to_paper,
            citation_df=citation_df,
            author_df=author_df,
            experiment_df=experiment_df,
            agri_records=agri_records,
        )

        # 11) retrain gate
        report.retrain_triggered = self._maybe_trigger_retrain(report)

        report.completed_at = datetime.now(timezone.utc).isoformat()
        report.duration_sec = round(time.monotonic() - started, 2)
        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
    def _load_literature_records(self) -> list[LiteratureRecord]:
        records: list[LiteratureRecord] = []
        for source in self.state.iter_sources():
            if source not in self._lit_by_source:
                continue
            for _source_id, payload in self.state.iter_cache(source):
                try:
                    records.append(LiteratureRecord.from_dict(payload))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("unparsable cached record %s/%s: %s", source, _source_id, exc)
        return records

    def _expand(
        self,
        records: list[LiteratureRecord],
        *,
        full: bool,
        report: PipelineReport,
    ) -> None:
        for record in records:
            connector = self._lit_by_source.get(record.source)
            if connector is None or not connector.enabled:
                continue
            key = f"expanded::{record.source_id}"
            if not full and self.state.get_cursor(record.source, key):
                continue
            try:
                if self.config.fetch_references and not record.references:
                    record.references = self._clean_ids(connector.fetch_references(record))
                if self.config.fetch_citations and not record.related_ids:
                    record.related_ids = self._clean_ids(connector.fetch_citations(record))
                if self.config.fetch_related:
                    related = self._clean_ids(connector.fetch_related_articles(record))
                    record.related_ids = list(dict.fromkeys(record.related_ids + related))
                if not record.pdf_locations:
                    record.pdf_locations = connector.fetch_pdf_location(record) or []
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"expand {record.source}/{record.source_id}: {exc}")
            self.state.set_cursor(record.source, key, True)

    @staticmethod
    def _clean_ids(ids: list[str] | None) -> list[str]:
        return list(dict.fromkeys(i for i in (ids or []) if i and str(i).strip()))

    def _resolve_paper_id(self, canonical: list[LiteratureRecord], pid: str) -> str:
        for paper in canonical:
            if paper.source_id == pid or paper.paper_id == pid:
                return paper.paper_id or paper.source_id
        return pid

    def _store_paper(self, paper: LiteratureRecord) -> None:
        try:
            self.bibliography.upsert_paper(paper)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to store paper %s", paper.paper_id)
        for pdf in paper.pdf_locations:
            try:
                self.bibliography.add_pdf(
                    paper.paper_id,
                    pdf.url,
                    pdf.source_kind,
                    pdf.license,
                    pdf.content_type,
                    pdf.verified,
                    pdf.source_repository,
                )
            except Exception:  # noqa: BLE001
                continue

    def _store_relations(self, paper: LiteratureRecord) -> None:
        try:
            if paper.references:
                self.bibliography.add_references(paper.paper_id, paper.references, paper.source)
            if paper.related_ids:
                self.bibliography.add_citations(paper.paper_id, paper.related_ids, paper.source)
        except Exception as exc:  # noqa: BLE001
            logger.debug("relation store failed for %s: %s", paper.paper_id, exc)

    def _verify_pdfs(
        self,
        papers: list[LiteratureRecord],
        cap: int | None = None,
    ) -> int:
        if not self.config.verify_pdfs:
            return 0
        cap = cap or _VERIFY_PDF_CAP_DEFAULT
        seen: set[str] = set()
        failures = 0
        for paper in papers:
            if failures >= cap:
                break
            for loc in paper.pdf_locations:
                if failures >= cap:
                    break
                if loc.url in seen:
                    continue
                seen.add(loc.url)
                connector = self._lit_by_source.get(paper.source)
                client = connector.http if connector else None
                if client is None:
                    continue
                status, content_type = client.head(loc.url)
                if status and 200 <= status < 400:
                    loc.verified = True
                    loc.content_type = content_type or loc.content_type
                else:
                    loc.verified = False
                    failures += 1
                    self._record_failed_pdf(paper, loc, status)
        return failures

    def _record_failed_pdf(self, paper: LiteratureRecord, loc: Any, status: int) -> None:
        try:
            self.state.set_cursor(
                paper.source,
                f"pdf_fail::{paper.source_id}::{loc.url[:120]}",
                {"status": status, "url": loc.url, "paper_id": paper.paper_id},
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # outputs
    # ------------------------------------------------------------------ #
    def _write_outputs(
        self,
        *,
        report: PipelineReport,
        records: list[LiteratureRecord],
        canonical: list[LiteratureRecord],
        verified: list[LiteratureRecord],
        source_to_paper: dict[tuple[str, str], str],
        citation_df: pd.DataFrame,
        author_df: pd.DataFrame,
        experiment_df: pd.DataFrame,
        agri_records: list[AgriculturalRecord],
    ) -> dict[str, Path]:
        out = self.config.output_dir
        outputs: dict[str, Path] = {}

        # --- metadata parquet: every original source record ------------- #
        meta_df = records_to_dataframe(records)
        if not meta_df.empty:
            meta_df = meta_df.drop(columns=["raw"], errors="ignore")
        outputs["literature_metadata.parquet"] = write_parquet(meta_df, out / "literature_metadata.parquet")

        # --- verified papers ------------------------------------------- #
        verified_df = records_to_dataframe(verified)
        if not verified_df.empty:
            keep = [
                c for c in (
                    "paper_id", "canonical_doi", "title", "year", "journal", "authors",
                    "country", "crop_terms", "experimental_design", "study_variables",
                    "quality_score", "valid_doi", "citations_count", "pdf_locations",
                ) if c in verified_df.columns
            ]
            verified_df = verified_df[keep]
        outputs["verified_papers.csv"] = write_csv(verified_df, out / "verified_papers.csv")

        # --- duplicates ------------------------------------------------- #
        dup_rows = [
            {
                "canonical_paper_id": source_to_paper.get((r.source, r.source_id), ""),
                "source": r.source,
                "source_id": r.source_id,
                "duplicate_of": r.duplicate_of,
                "doi": r.doi,
                "title": r.title,
            }
            for r in records if r.duplicate_of
        ]
        outputs["duplicate_records.csv"] = write_csv(
            pd.DataFrame(dup_rows), out / "duplicate_records.csv"
        )

        # --- failed downloads ------------------------------------------ #
        failed_rows: list[dict[str, Any]] = []
        for paper in canonical:
            for loc in paper.pdf_locations:
                if not loc.verified:
                    failed_rows.append(
                        {
                            "paper_id": paper.paper_id,
                            "doi": paper.canonical_doi or paper.doi,
                            "title": paper.title,
                            "url": loc.url,
                            "source_kind": loc.source_kind,
                            "source": paper.source,
                        }
                    )
        outputs["failed_downloads.csv"] = write_csv(
            pd.DataFrame(failed_rows), out / "failed_downloads.csv"
        )

        # --- graphs ----------------------------------------------------- #
        outputs["citation_graph.parquet"] = write_parquet(citation_df, out / "citation_graph.parquet")
        outputs["author_graph.parquet"] = write_parquet(author_df, out / "author_graph.parquet")
        outputs["experiment_graph.parquet"] = write_parquet(experiment_df, out / "experiment_graph.parquet")

        # --- training dataset ------------------------------------------ #
        train_df = self._training_dataset(verified)
        outputs["training_dataset.parquet"] = write_parquet(train_df, out / "training_dataset.parquet")

        # --- UAMS schema update ---------------------------------------- #
        uams_df = self._updated_schema(verified, agri_records)
        outputs["updated_schema.csv"] = write_csv(uams_df, out / "updated_schema.csv")
        outputs["universal_schema.csv"] = write_csv(
            uams_df, out / "universal_schema.csv"
        )

        # --- quality scores -------------------------------------------- #
        outputs["quality_scores.csv"] = write_csv(
            self._quality_scores(records), out / "quality_scores.csv"
        )

        # --- ready reckoner / model performance ------------------------ #
        outputs["ready_reckoner.xlsx"] = write_excel(
            self._ready_reckoner_sheets(verified, records, agri_records),
            out / "ready_reckoner.xlsx",
        )
        outputs["model_performance.xlsx"] = write_excel(
            self._model_performance_sheets(report, canonical),
            out / "model_performance.xlsx",
        )

        # --- framework reports ----------------------------------------- #
        html_sections = self._report_sections(report, canonical, verified, citation_df, author_df, experiment_df)
        outputs["framework_report.html"] = write_html_report(
            out / "framework_report.html", "AAIF Literature Intelligence Framework Report", html_sections
        )
        outputs["framework_report.docx"] = write_docx_report(
            out / "framework_report.docx", "AAIF Literature Intelligence Framework Report", html_sections
        )
        outputs["framework_report.pdf"] = write_pdf_report(
            out / "framework_report.pdf", "AAIF Literature Intelligence Framework Report", html_sections
        )

        # --- framework health dashboard -------------------------------- #
        outputs["framework_health_dashboard.html"] = build_health_dashboard(
            report=report,
            pipeline=self,
            output_dir=out,
        )
        return outputs

    def _training_dataset(self, verified: list[LiteratureRecord]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for paper in verified:
            rows.append(
                {
                    "paper_id": paper.paper_id,
                    "doi": paper.canonical_doi,
                    "title": paper.title,
                    "year": paper.year,
                    "journal": paper.journal,
                    "authors": "; ".join(a.full_name for a in paper.authors),
                    "country": paper.country,
                    "crop": ";".join(paper.crop_terms),
                    "experimental_design": ";".join(paper.experimental_design),
                    "study_variables": ";".join(paper.study_variables),
                    "quality_score": paper.quality_score,
                    "citations_count": paper.citations_count,
                    "source": paper.source,
                    "source_id": paper.source_id,
                }
            )
        return pd.DataFrame(rows)

    def _updated_schema(
        self,
        verified: list[LiteratureRecord],
        agri_records: list[AgriculturalRecord],
    ) -> pd.DataFrame:
        schema_path = Path("outputs/Universal_Agricultural_Schema.csv")
        if schema_path.exists():
            df = pd.read_csv(schema_path)
        else:
            df = pd.DataFrame(columns=list(UAMS_COLUMNS))
        for paper in verified:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            self._uams_row(paper)
                        ]
                    ),
                ],
                ignore_index=True,
            )
        for rec in agri_records:
            df = pd.concat([df, pd.DataFrame([self._uams_agri_row(rec)])], ignore_index=True)
        return df

    def _uams_row(self, paper: LiteratureRecord) -> dict[str, Any]:
        row = {col: None for col in UAMS_COLUMNS}
        row["Paper_ID"] = paper.paper_id
        row["DOI"] = paper.canonical_doi
        row["Journal"] = paper.journal
        row["Year"] = paper.year
        row["Authors"] = "; ".join(a.full_name for a in paper.authors)
        row["Country"] = paper.country
        row["Title"] = paper.title
        row["Crop"] = ";".join(paper.crop_terms) or None
        row["Design"] = ";".join(paper.experimental_design) or None
        row["Yield_per_Hectare"] = None
        return row

    def _uams_agri_row(self, rec: AgriculturalRecord) -> dict[str, Any]:
        row = {col: None for col in UAMS_COLUMNS}
        row["Year"] = rec.year
        row["Country"] = rec.country
        row["Crop"] = rec.crop
        row["Latitude"] = rec.location_lat
        row["Longitude"] = rec.location_lon
        target = self.variable_to_uams.get(rec.variable)
        if target in UAMS_COLUMNS:
            row[target] = rec.value
        return row

    def _ready_reckoner_sheets(
        self,
        verified: list[LiteratureRecord],
        records: list[LiteratureRecord],
        agri_records: list[AgriculturalRecord],
    ) -> dict[str, pd.DataFrame]:
        papers_df = self._training_dataset(verified)

        crop_design_rows = [
            {"crop": crop, "experimental_design": design}
            for p in verified for crop in (p.crop_terms or ["(unspecified)"])
            for design in (p.experimental_design or ["(unspecified)"])
        ]
        crop_design = pd.DataFrame(crop_design_rows).groupby(
            ["crop", "experimental_design"], dropna=False
        ).size().reset_index(name="papers")

        var_rows = [
            {"crop": crop, "study_variable": var}
            for p in verified for crop in (p.crop_terms or ["(unspecified)"])
            for var in (p.study_variables or [])
        ]
        crop_var = pd.DataFrame(var_rows).groupby(
            ["crop", "study_variable"], dropna=False
        ).size().reset_index(name="papers")

        agri_df = pd.DataFrame([r.to_dict() for r in agri_records])
        if not agri_df.empty:
            agri_df = agri_df.drop(columns=["extra"], errors="ignore")

        return {
            "Verified_Papers": papers_df,
            "Crop_x_Design": crop_design,
            "Crop_x_Variable": crop_var,
            "Agri_Datasets": agri_df,
            "Sources_Count": pd.DataFrame(
                [
                    {
                        "source": c.source_name,
                        "enabled": c.enabled,
                        "records": sum(
                            1 for r in records if r.source == c.source_name
                        ),
                    }
                    for c in self.connectors
                ]
            ),
        }

    def _model_performance_sheets(
        self,
        report: PipelineReport,
        canonical: list[LiteratureRecord],
    ) -> dict[str, pd.DataFrame]:
        connector_rows = [
            {
                "connector": r.connector,
                "fetched": r.fetched_records,
                "new": r.new_records,
                "updated": r.updated_records,
                "skipped": r.skipped_records,
                "errors": "; ".join(r.errors),
                "cursor": r.cursor,
            }
            for r in report.sync_results + report.agri_sync_results
        ]
        dup = sum(1 for p in canonical if p.duplicate_of)
        doi_valid = sum(1 for p in canonical if p.valid_doi)
        return {
            "Connector_Summary": pd.DataFrame(connector_rows),
            "Deduplication": pd.DataFrame(
                [
                    {"papers": len(canonical), "duplicates_merged": dup},
                    {"run_wide": True},
                ]
            ),
            "DOI_Validation": pd.DataFrame(
                [
                    {
                        "papers_with_doi": sum(1 for p in canonical if p.canonical_doi),
                        "validated": doi_valid,
                        "invalid_or_unchecked": sum(
                            1 for p in canonical if p.canonical_doi and not p.valid_doi
                        ),
                    }
                ]
            ),
            "Quality_Distribution": self._quality_distribution(canonical),
            "Verified_By_Year": pd.DataFrame(
                [
                    {"year": y, "papers": n}
                    for y, n in pd.Series(
                        [p.year for p in canonical if p.is_original_study and p.year]
                    ).value_counts().sort_index().items()
                ]
            ),
            "Graph_Stats": pd.DataFrame(
                [
                    {"metric": "citation_edges", "value": report.outputs.get("citation_graph.parquet").stat().st_size
                     if report.outputs.get("citation_graph.parquet") else 0},
                ]
            ),
        }

    def _quality_distribution(self, canonical: list[LiteratureRecord]) -> pd.DataFrame:
        bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        counts = []
        for lo, hi in bins:
            n = sum(1 for p in canonical if p.quality_score is not None and lo <= p.quality_score < hi)
            counts.append({"score_bin": f"{lo:.1f}-{hi:.1f}", "papers": n})
        return pd.DataFrame(counts)

    def _quality_scores(self, records: list[LiteratureRecord]) -> pd.DataFrame:
        """Per-record quality breakdown used by the quality-scores deliverable."""
        rows: list[dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "paper_id": record.paper_id,
                    "source": record.source,
                    "source_id": record.source_id,
                    "doi": record.canonical_doi or record.doi,
                    "title": record.title,
                    "year": record.year,
                    "is_original_study": record.is_original_study,
                    "quality_score": record.quality_score,
                    "has_doi": bool(record.doi),
                    "has_abstract": bool(record.abstract and len(record.abstract) > 100),
                    "has_pdf": bool(record.pdf_locations),
                    "has_references": bool(record.references),
                    "design_signal": bool(record.experimental_design),
                    "valid_doi": bool(record.valid_doi),
                    "duplicate_of": record.duplicate_of,
                }
            )
        return pd.DataFrame(rows)

    def _report_sections(
        self,
        report: PipelineReport,
        canonical: list[LiteratureRecord],
        verified: list[LiteratureRecord],
        citation_df: pd.DataFrame,
        author_df: pd.DataFrame,
        experiment_df: pd.DataFrame,
    ) -> list[tuple[str, Any]]:
        conn_health = [
            {"source": c.source_name, "enabled": c.enabled}
            for c in self.connectors
        ]
        return [
            ("Executive Summary", {
                "run_id": report.run_id,
                "completed_at": report.completed_at,
                "duration_sec": report.duration_sec,
                "papers": report.total_papers,
                "verified_papers": report.verified_papers,
                "duplicates_merged": report.duplicates,
                "new_records": report.new_records,
                "agri_records": report.agri_records,
                "outputs": len(report.outputs),
            }),
            ("Connector Health", conn_health),
            ("Verified Papers", self._training_dataset(verified)),
            ("Citation Graph (sample)", citation_df.head(200) if not citation_df.empty else pd.DataFrame()),
            ("Author Graph (top 50 by shared papers)", author_df.head(50) if not author_df.empty else pd.DataFrame()),
            ("Experiment Graph (sample)", experiment_df.head(200) if not experiment_df.empty else pd.DataFrame()),
            ("Errors", report.errors or ["(none)"]),
        ]

    # ------------------------------------------------------------------ #
    # retrain gate
    # ------------------------------------------------------------------ #
    def _maybe_trigger_retrain(self, report: PipelineReport) -> bool:
        thresholds_met = (
            report.verified_papers >= self.config.retrain_min_verified_papers
            and report.new_records >= self.config.retrain_min_new_papers
        )
        if not thresholds_met:
            return False
        logger.warning(
            "Retrain thresholds met (verified=%d >= %d, new=%d >= %d)",
            report.verified_papers,
            self.config.retrain_min_verified_papers,
            report.new_records,
            self.config.retrain_min_new_papers,
        )
        if not self.config.auto_retrain:
            return True
        command = self.config.retrain_command
        if not command:
            logger.warning(
                "auto_retrain enabled but no AGRI_LIT_RETRAIN_COMMAND configured; "
                "training not started."
            )
            return True
        try:
            logger.info("Launching retrain: %s", command)
            subprocess.run(
                command.split(),
                cwd=str(Path.cwd()),
                check=False,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"retrain command failed: {exc}")
            return False
