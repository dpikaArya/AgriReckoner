"""
LiteratureAgent — Phase 0 of the AAIF orchestrator.

Wraps the Literature Intelligence Module pipeline as a native orchestrator
step.  The step:

* runs the literature pipeline (discovery + agri sync + validation + PDFs
  + schema update) against the externalized ``config/`` policy files;
* merges the resulting Universal Agricultural Schema (UAMS) rows into the
  current dataframe, so downstream phases train on both user datasets and
  literature-sourced observations;
* never terminates the pipeline on connector failures (errors are captured
  in the agent contract and the input dataframe is preserved).

The phase can be disabled wholesale via ``AGRI_INCLUDE_LITERATURE_PHASE=0``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LiteratureAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "LiteratureAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not self.settings.INCLUDE_LITERATURE_PHASE:
            self.log.info("Literature phase disabled via settings; passing through.")
            return df

        from agri_ai_agent.literature.config import LiteratureConfig
        from agri_ai_agent.literature.pipeline import LiteraturePipeline

        self.log.info("Running Literature Intelligence Module (Phase 0)...")
        try:
            config = LiteratureConfig.from_env(root=self.settings.BASE_DIR)
            if config.search_terms is None or not config.search_terms:
                config.search_terms = _default_terms()
            skip_agri = _as_bool(self.settings.LITERATURE_SKIP_AGRI_ON_RUN)
            pipeline = LiteraturePipeline(config=config)
            report = pipeline.run(
                terms=config.search_terms,
                skip_agri=skip_agri,
            )
            self.log.info("Literature pipeline complete:\n%s", report.summary())

            uams = self._load_uams_rows(config.output_dir)
            self._last_report = report

            if df is not None and not df.empty:
                if uams is not None and not uams.empty:
                    combined = pd.concat([df, uams], ignore_index=True, sort=False)
                    self.log.info(
                        "Merged %d literature UAMS rows into %d-row dataframe",
                        len(uams),
                        len(df),
                    )
                    return combined
                return df
            return uams if uams is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001 - connector isolation
            self.log.error("Literature phase failed (%s); passing input through.", exc)
            if self.contract is not None:
                self.contract.errors.append(f"literature_phase: {exc}")
            return df

    # ------------------------------------------------------------------ #
    def _load_uams_rows(self, output_dir: Path) -> pd.DataFrame | None:
        candidates = [
            Path(output_dir) / "universal_schema.csv",
            Path(output_dir) / "updated_schema.csv",
        ]
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        return df
                except Exception as exc:  # noqa: BLE001
                    self.log.debug("Could not read %s: %s", path, exc)
        return None

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        output = super()._build_output(df, **kwargs)
        report = getattr(self, "_last_report", None)
        if report is not None:
            output.update(
                {
                    "literature_run_id": report.run_id,
                    "papers": report.total_papers,
                    "verified_original_papers": report.verified_papers,
                    "duplicates_merged": report.duplicates,
                    "validated_dois": report.validated_dois,
                    "new_records": report.new_records,
                    "agri_records": report.agri_records,
                    "failed_pdfs": report.failed_pdfs,
                    "outputs": len(report.outputs),
                    "errors": list(report.errors[:20]),
                    "retrain_triggered": report.retrain_triggered,
                }
            )
        return output


def _as_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _default_terms() -> tuple[str, ...]:
    """Default boolean query set (field-experiment / agronomy coverage)."""
    return (
        "field experiment AND (crop OR wheat OR maize OR rice) AND yield",
        "agronomy AND (nitrogen OR fertilizer) AND (field trial) AND yield",
        "(soil AND irrigation) AND (randomized complete block) AND yield",
        "crop modeling AND (variety OR cultivar) AND (field) AND yield",
    )
