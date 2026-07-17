"""
Evidence Fusion Agent
Merges extraction results from multiple readers, resolves conflicts,
assigns confidence scores, and preserves full provenance metadata.
Replaces the legacy aaif/extraction/validation_agent.py for the pipeline.
"""

import os
import re
import json
from collections import defaultdict
from typing import Any, Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.config.settings import AgriAISettings


READER_CONFIDENCE_WEIGHTS = {
    "pdfminer": 0.70,
    "camelot": 0.85,
    "pdfplumber": 0.80,
    "poppler": 0.65,
    "ocr": 0.50,
    "semantic": 0.60,
    "table_detector": 0.85,
    "master_dataset": 0.90,
}

UNIT_CONVERSIONS = {
    "t_per_ha": {"to": "kg_per_ha", "factor": 1000},
    "q_per_ha": {"to": "kg_per_ha", "factor": 100},
    "kg_per_acre": {"to": "kg_per_ha", "factor": 2.471},
    "g_per_plot": {"to": "kg_per_ha", "factor": 0.001},
}

YIELD_COLS = [
    "Yield_per_Plot", "Yield_per_Hectare", "Yield_per_Acre",
    "Biomass_Yield", "Harvest_Index",
]

TID_CLEAN = re.compile(r"[†‡*]")


class EvidenceFusionAgent(BaseAgent):
    def __init__(self, settings: Optional[AgriAISettings] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.fusion_report: dict[str, Any] = {}

    @property
    def agent_name(self) -> str:
        return "EvidenceFusionAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stage_results = kwargs.get("stage_results", [])
        source_file = kwargs.get("source_file", "")

        if not stage_results and df is not None and not df.empty:
            return df

        if not stage_results:
            if source_file:
                fallback = self._fallback_metadata(source_file)
                result_df = pd.DataFrame(fallback)
            else:
                result_df = df if df is not None else pd.DataFrame()
            self._enrich_output(result_df)
            return result_df

        all_evidence = self._collect_evidence(stage_results)
        if not all_evidence:
            if source_file:
                fallback = self._fallback_metadata(source_file)
                result_df = pd.DataFrame(fallback)
            else:
                result_df = pd.DataFrame()
            self._enrich_output(result_df)
            return result_df

        fused = self._fuse_evidence(all_evidence)
        fused = self._normalize_units(fused)
        fused = self._deduplicate(fused)
        fused = self._validate_crop_consistency(fused)
        fused = self._validate_treatment_ids(fused)

        result_df = pd.DataFrame(fused)
        self._enrich_output(result_df)
        self.fusion_report = self._build_report(stage_results, len(fused))
        return result_df

    def run(
        self,
        df: pd.DataFrame,
        contract: Optional[AgentContract] = None,
        **kwargs,
    ) -> AgentContract:
        contract = contract or AgentContract(agent_name=self.agent_name)
        contract.status = "running"

        try:
            result_df = self.process(df, **kwargs)
            self.dataframe = result_df
            contract.status = "success"
            contract.output_data = self._build_output(result_df, **kwargs)
            contract.output_data["fusion_report"] = self.fusion_report
        except Exception as e:
            contract.status = "failed"
            contract.errors.append(str(e))
            self.log.error("[%s] Fusion failed: %s", self.agent_name, e)

        return contract

    def _collect_evidence(self, stage_results: list[dict]) -> list[dict]:
        all_evidence = []
        for sr in stage_results:
            if not sr.get("success") or not sr.get("rows"):
                continue
            reader = sr.get("reader", "unknown")
            base_conf = READER_CONFIDENCE_WEIGHTS.get(reader, 0.50)
            sr_conf = sr.get("confidence", 0.5)

            for row in sr["rows"]:
                evidence_row = dict(row)
                evidence_row["_reader"] = reader
                evidence_row["_reader_confidence"] = base_conf
                evidence_row["_stage_confidence"] = sr_conf
                evidence_row["_fused_confidence"] = round(
                    (base_conf * 0.6 + sr_conf * 0.4), 4
                )
                evidence_row["_source_file"] = row.get("Source_File", "")
                all_evidence.append(evidence_row)
        return all_evidence

    def _fuse_evidence(self, all_evidence: list[dict]) -> list[dict]:
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for ev in all_evidence:
            key = (
                ev.get("_source_file", ""),
                ev.get("Source_File", ""),
                ev.get("Treatment", ""),
            )
            grouped[key].append(ev)

        fused_rows = []
        for (src_file, _, treatment), evidence_list in grouped.items():
            merged: dict[str, Any] = {
                "Source_File": src_file,
                "Treatment": TID_CLEAN.sub("", treatment).strip() if treatment else "",
            }
            provenance: dict[str, dict] = {}

            all_columns = set()
            for ev in evidence_list:
                all_columns.update(ev.keys())

            skip_cols = {
                "_reader", "_reader_confidence", "_stage_confidence",
                "_fused_confidence", "_source_file", "Source_File", "Treatment",
            }

            for col in all_columns - skip_cols:
                candidates = []
                for ev in evidence_list:
                    val = ev.get(col)
                    if pd.notna(val) and val is not None and val != "":
                        candidates.append(
                            {
                                "value": val,
                                "confidence": ev.get("_fused_confidence", 0.5),
                                "reader": ev.get("_reader", "unknown"),
                                "source_file": ev.get("_source_file", ""),
                            }
                        )

                if not candidates:
                    continue

                if len(candidates) == 1:
                    best = candidates[0]
                else:
                    best = max(candidates, key=lambda c: c["confidence"])

                merged[col] = best["value"]
                provenance[col] = {
                    "value": best["value"],
                    "confidence": best["confidence"],
                    "source_reader": best["reader"],
                    "source_file": best["source_file"],
                    "total_observations": len(candidates),
                    "conflict_resolved": len(candidates) > 1,
                }

                if len(candidates) > 1:
                    all_vals = [c["value"] for c in candidates]
                    if all(isinstance(v, (int, float)) for v in all_vals):
                        provenance[col]["values_tried"] = all_vals
                        provenance[col]["mean"] = round(
                            sum(all_vals) / len(all_vals), 4
                        )

            merged["_provenance"] = provenance
            merged["_evidence_count"] = len(evidence_list)
            merged["_readers_used"] = list(
                set(ev.get("_reader", "unknown") for ev in evidence_list)
            )

            confs = [
                ev.get("_fused_confidence", 0.5) for ev in evidence_list
            ]
            merged["_fused_confidence"] = round(sum(confs) / len(confs), 4)

            fused_rows.append(merged)

        return fused_rows

    def _normalize_units(self, rows: list[dict]) -> list[dict]:
        for row in rows:
            for yc in YIELD_COLS:
                v = row.get(yc)
                if v is None or not isinstance(v, (int, float)):
                    continue
                prov = row.get("_provenance", {}).get(yc, {})
                if yc == "Yield_per_Hectare" and v > 50000:
                    row[yc] = round(v / 1000, 2)
                    prov["unit_converted"] = True
                    prov["conversion_note"] = "Divided by 1000 (assumed g/m2 to kg/ha)"
                elif yc == "Yield_per_Plot" and v < 1:
                    row[yc] = round(v * 1000, 2)
                    prov["unit_converted"] = True
                    prov["conversion_note"] = "Multiplied by 1000 (assumed t to kg)"
                if prov:
                    row["_provenance"][yc] = prov
        return rows

    def _deduplicate(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return rows
        seen: dict[tuple, dict] = {}
        for r in rows:
            key = (r.get("Source_File", ""), r.get("Treatment", ""))
            if key not in seen:
                seen[key] = r
            else:
                existing = seen[key]
                n_new = sum(
                    1 for k, v in r.items()
                    if v is not None
                    and k not in ("Treatment", "Source_File", "_reader", "_confidence",
                                  "_provenance", "_evidence_count", "_readers_used",
                                  "_fused_confidence")
                )
                n_old = sum(
                    1 for k, v in existing.items()
                    if v is not None
                    and k not in ("Treatment", "Source_File", "_reader", "_confidence",
                                  "_provenance", "_evidence_count", "_readers_used",
                                  "_fused_confidence")
                )
                if n_new > n_old:
                    seen[key] = r
        return list(seen.values())

    def _validate_crop_consistency(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return rows
        paper_crops: dict[str, set] = defaultdict(set)
        for r in rows:
            src = r.get("Source_File", "")
            crop = r.get("Crop", "")
            if src and crop:
                paper_crops[src].add(crop)

        for src, crops in paper_crops.items():
            if len(crops) <= 1:
                continue
            counts = {}
            for r in rows:
                if r.get("Source_File") == src and r.get("Crop"):
                    counts[r["Crop"]] = counts.get(r["Crop"], 0) + 1
            majority = max(counts, key=counts.get)
            for r in rows:
                if r.get("Source_File") == src and r.get("Crop") != majority:
                    old_crop = r["Crop"]
                    r["Crop"] = majority
                    prov = r.get("_provenance", {})
                    prov["Crop"] = {
                        "value": majority,
                        "confidence": 0.95,
                        "source_reader": "evidence_fusion",
                        "crop_corrected_from": old_crop,
                    }
                    r["_provenance"] = prov
        return rows

    def _validate_treatment_ids(self, rows: list[dict]) -> list[dict]:
        for r in rows:
            tid = r.get("Treatment", "")
            if tid and len(tid) > 50:
                r["Treatment"] = tid[:50]
                r.setdefault("_warnings", []).append(
                    f"Treatment ID truncated from {len(tid)} to 50 chars"
                )
        return rows

    def _fallback_metadata(self, source_file: str) -> list[dict]:
        pdf_name = os.path.basename(source_file)
        return [
            {
                "Source_File": pdf_name,
                "Treatment": "UNKNOWN",
                "_fused_confidence": 0.1,
                "_evidence_count": 0,
                "_readers_used": [],
                "_provenance": {},
            }
        ]

    def _enrich_output(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        for col in [
            "_provenance", "_evidence_count", "_readers_used", "_fused_confidence",
        ]:
            if col not in df.columns:
                df[col] = None

    def _build_report(self, stage_results: list[dict], n_fused: int) -> dict:
        readers_summary = {}
        for sr in stage_results:
            reader = sr.get("reader", "unknown")
            readers_summary[reader] = {
                "success": sr.get("success", False),
                "confidence": sr.get("confidence", 0.0),
                "rows_extracted": len(sr.get("rows", [])),
            }

        total_raw = sum(len(sr.get("rows", [])) for sr in stage_results)
        successful = sum(1 for sr in stage_results if sr.get("success"))

        return {
            "stages_input": len(stage_results),
            "stages_successful": successful,
            "total_raw_rows": total_raw,
            "rows_after_fusion": n_fused,
            "fusion_rate": round(n_fused / max(1, total_raw), 3),
            "readers": readers_summary,
        }

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        base = super()._build_output(df, **kwargs)
        base["fusion_report"] = self.fusion_report
        if df is not None and not df.empty:
            prov_cols = [c for c in df.columns if c.startswith("_")]
            base["provenance_columns"] = prov_cols
            base["rows_with_provenance"] = int(
                df["_provenance"].notna().sum() if "_provenance" in df.columns else 0
            )
        return base
