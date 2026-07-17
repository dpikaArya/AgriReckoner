"""
ProvenanceAgent — Phase 14.

Ensures every extracted value carries full provenance metadata:
source (paper/DOI), confidence, extraction_method, unit, timestamp,
page_number, table_id, and derivation_method for computed columns.
"""

import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract


EXTRACTION_METHODS = {
    "pdfminer": 0.70,
    "camelot": 0.85,
    "pdfplumber": 0.80,
    "poppler": 0.65,
    "ocr": 0.50,
    "semantic": 0.60,
    "table_detector": 0.85,
    "master_dataset": 0.90,
    "manual": 0.95,
    "knowledge_base": 0.80,
    "computed": 0.75,
    "imputed": 0.60,
    "default": 0.50,
}


class ProvenanceAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ProvenanceAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.log.info("=" * 60)
        self.log.info("ProvenanceAgent: Tagging values with lineage metadata")
        self.log.info("=" * 60)

        source_paper = kwargs.get("source_paper", "unknown")
        source_doi = kwargs.get("source_doi", "")
        extraction_method = kwargs.get("extraction_method", "master_dataset")
        page_number = kwargs.get("page_number", None)
        table_id = kwargs.get("table_id", None)

        prov_cols = {
            "Provenance_Source": source_paper,
            "Provenance_DOI": source_doi,
            "Provenance_Method": extraction_method,
            "Provenance_Confidence": EXTRACTION_METHODS.get(extraction_method, 0.50),
            "Provenance_Timestamp": datetime.now().isoformat(),
            "Provenance_Page": page_number,
            "Provenance_Table": table_id,
            "Provenance_Derived": False,
        }

        for col, default_val in prov_cols.items():
            if col not in df.columns:
                df[col] = default_val

        self._tag_derived_columns(df)
        self._tag_missing_values(df)

        report = self._generate_report(df)
        if self.contract is not None:
            self.contract.output_data["provenance_report"] = report

        self.log.info("Provenance coverage: %.1f%% non-null values",
                       self._coverage_pct(df))
        self.log.info("Total tracked values: %d", self._total_values(df))

        return df

    def _tag_derived_columns(self, df: pd.DataFrame):
        derived_keywords = [
            "_log", "_squared", "_sqrt", "_ratio", "_bin", "_category",
            "_x_", "_interaction", "_NPK", "RUE", "WUE", "HI_",
            "NUE", "Shoot_Root", "Root_Shoot", "Root_pct", "Shoot_pct",
        ]
        for col in df.columns:
            if any(kw in col for kw in derived_keywords):
                mask = pd.notna(df[col])
                df.loc[mask, "Provenance_Derived"] = True
                df.loc[mask, "Provenance_Method"] = "computed"

    def _tag_missing_values(self, df: pd.DataFrame):
        missing_cols = []
        for col in df.columns:
            if col.startswith("Provenance_"):
                continue
            if df[col].isna().any():
                missing_cols.append(col)
        if missing_cols and "Provenance_Missing_Fields" not in df.columns:
            df["Provenance_Missing_Fields"] = ""
        for idx in df.index:
            row_missing = []
            for col in df.columns:
                if col.startswith("Provenance_"):
                    continue
                if pd.isna(df.at[idx, col]):
                    row_missing.append(col)
            if row_missing:
                df.at[idx, "Provenance_Missing_Fields"] = ",".join(row_missing)

    def _coverage_pct(self, df: pd.DataFrame) -> float:
        tracked = [c for c in df.columns if not c.startswith("Provenance_")]
        if not tracked:
            return 0.0
        total = 0
        filled = 0
        for col in tracked:
            total += len(df)
            filled += df[col].notna().sum()
        return (filled / total * 100) if total > 0 else 0.0

    def _total_values(self, df: pd.DataFrame) -> int:
        tracked = [c for c in df.columns if not c.startswith("Provenance_")]
        return sum(len(df) for _ in tracked)

    def _generate_report(self, df: pd.DataFrame) -> dict:
        prov_cols = [c for c in df.columns if c.startswith("Provenance_")]
        method_counts = {}
        if "Provenance_Method" in df.columns:
            method_counts = df["Provenance_Method"].value_counts().to_dict()

        avg_confidence = 0.0
        if "Provenance_Confidence" in df.columns:
            vals = pd.to_numeric(df["Provenance_Confidence"], errors="coerce").dropna()
            avg_confidence = float(vals.mean()) if len(vals) > 0 else 0.0

        derived_count = 0
        if "Provenance_Derived" in df.columns:
            derived_count = int(df["Provenance_Derived"].sum())

        return {
            "provenance_columns_added": len(prov_cols),
            "coverage_pct": round(self._coverage_pct(df), 1),
            "total_values_tracked": self._total_values(df),
            "method_distribution": method_counts,
            "average_confidence": round(avg_confidence, 3),
            "derived_values": derived_count,
        }

    def validate_provenance(self, df: pd.DataFrame) -> dict:
        issues = []
        prov_cols = ["Provenance_Source", "Provenance_Method", "Provenance_Confidence"]
        for col in prov_cols:
            if col not in df.columns:
                issues.append(f"Missing provenance column: {col}")

        if "Provenance_Source" in df.columns:
            unknown_mask = df["Provenance_Source"].isin(["unknown", "", None])
            if unknown_mask.any():
                issues.append(f"{unknown_mask.sum()} rows with unknown source")

        if "Provenance_Confidence" in df.columns:
            low_conf = pd.to_numeric(df["Provenance_Confidence"], errors="coerce") < 0.5
            if low_conf.sum() > 0:
                issues.append(f"{low_conf.sum()} rows with confidence < 0.5")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "issue_count": len(issues),
        }

    def provenance_summary_by_paper(self, df: pd.DataFrame) -> dict:
        if "Provenance_Source" not in df.columns:
            return {}
        summary = {}
        for paper, group in df.groupby("Provenance_Source"):
            prov_cols = [c for c in group.columns if not c.startswith("Provenance_")]
            total = sum(len(group) for _ in prov_cols)
            filled = sum(group[c].notna().sum() for c in prov_cols)
            avg_conf = 0.0
            if "Provenance_Confidence" in group.columns:
                vals = pd.to_numeric(group["Provenance_Confidence"], errors="coerce").dropna()
                avg_conf = float(vals.mean()) if len(vals) > 0 else 0.0
            summary[paper] = {
                "rows": len(group),
                "coverage_pct": round(filled / total * 100, 1) if total > 0 else 0.0,
                "avg_confidence": round(avg_conf, 3),
            }
        return summary
