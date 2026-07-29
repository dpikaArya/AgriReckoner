import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.knowledge_graph.graph import KnowledgeGraph
from agri_ai_agent.knowledge_graph.provenance import (
    register_dataset_lineage,
    register_document_lineage,
    update_knowledge_graph,
)


class DatasetNormalizationAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "DatasetNormalizationAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        packages_in: list[DatasetPackage] = kwargs.get("packages") or []
        raw_df = df if df is not None else pd.DataFrame()

        normalized: list[DatasetPackage] = []

        for pkg in packages_in:
            norm = self._normalize_package(pkg)
            normalized.append(norm)

        if raw_df is not None and not raw_df.empty and not normalized:
            pkg = self._package_from_dataframe(raw_df, kwargs)
            normalized.append(pkg)

        if not normalized and raw_df is not None and not raw_df.empty:
            pkg = self._package_from_dataframe(raw_df, kwargs)
            normalized.append(pkg)

        merged = self._merge_normalized(normalized, raw_df)

        self._save_artifact(normalized)
        self._write_registry_export(normalized)

        kg: KnowledgeGraph = kwargs.get("knowledge_graph")
        provenance_dir = Path(kwargs.get("provenance_dir", self.settings.EXTERNAL_DATA_DIR / "provenance"))
        for pkg in normalized:
            try:
                if kg is not None:
                    update_knowledge_graph(kg, pkg)
                lineage = register_dataset_lineage(pkg)
                doc_lineage = register_document_lineage(pkg)
                provenance_dir.mkdir(parents=True, exist_ok=True)
                (provenance_dir / f"lineage_{pkg.dataset_id}.json").write_text(
                    json.dumps({"dataset_lineage": lineage, "document_lineage": doc_lineage},
                               indent=2, default=str),
                    encoding="utf-8",
                )
                self.contract.dataset_id = pkg.dataset_id
                self.contract.provider = pkg.provider or pkg.source
                self.contract.checksum = pkg.checksum
                self.contract.processing_stage = "dataset_normalization"
                self.contract.processing_mode = "normalize"
            except Exception as exc:
                self.log.warning("[%s] lineage registration failed for %s: %s",
                                 self.agent_name, pkg.dataset_id, exc)

        self.log.info(
            "[%s] Normalized %d package(s) into %d rows x %d cols",
            self.agent_name, len(normalized), len(merged), len(merged.columns) if not merged.empty else 0,
        )
        return merged

    def _normalize_package(self, pkg: DatasetPackage) -> DatasetPackage:
        if not pkg.dataset_id:
            pkg.dataset_id = pkg._generate_id()
        if not pkg.provider:
            pkg.provider = pkg.source
        if not pkg.document_type:
            pkg.document_type = self._detect_document_type(pkg)
        if not pkg.checksum:
            pkg.checksum = pkg._compute_checksum()

        df = pkg.to_dataframe()
        if df is not None and not df.empty:
            if not pkg.tables:
                pkg.tables = [df]
            if not pkg.schema:
                pkg.build_schema()
            if not pkg.statistics:
                pkg.build_statistics()
            pkg.row_count = len(df)
            pkg.column_count = len(df.columns)

        if not pkg.metadata:
            pkg.metadata = {
                "name": pkg.name or pkg.resource_id,
                "source": pkg.source or pkg.provider,
                "resource_id": pkg.resource_id,
                "version": pkg.version,
                "source_url": pkg.source_url,
                "downloaded_at": str(pkg.downloaded_at or datetime.now()),
                "description": pkg.description,
            }

        if not pkg.is_valid:
            if df is not None and not df.empty:
                pkg.is_valid = True
            elif pkg.download_path and pkg.download_path.exists():
                pkg.is_valid = True

        return pkg

    def _package_from_dataframe(self, df: pd.DataFrame, kwargs: dict) -> DatasetPackage:
        provider = kwargs.get("provider", "pipeline_input")
        document_type = "tabular"
        dataset_id = hashlib.sha256(f"{provider}/{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
        }

        numeric = df.select_dtypes(include="number")
        statistics = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": len(numeric.columns),
            "numeric_stats": numeric.describe().to_dict() if not numeric.empty else {},
        }

        metadata = {
            "name": kwargs.get("name", "pipeline_input"),
            "provider": provider,
            "rows": len(df),
            "columns": len(df.columns),
            "created_at": datetime.now().isoformat(),
        }

        return DatasetPackage(
            dataset_id=dataset_id,
            provider=provider,
            document_type=document_type,
            metadata=metadata,
            tables=[df],
            data=df,
            schema=schema,
            statistics=statistics,
            row_count=len(df),
            column_count=len(df.columns),
            is_valid=True,
        )

    def _detect_document_type(self, pkg: DatasetPackage) -> str:
        if pkg.tables or pkg.data is not None:
            return "tabular"
        if pkg.documents:
            return "document"
        if pkg.images:
            return "image"
        if pkg.download_path:
            ext = pkg.download_path.suffix.lower()
            if ext in (".csv", ".xlsx", ".xls", ".parquet", ".json"):
                return "tabular"
            if ext == ".pdf":
                return "pdf"
            if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                return "image"
        return "unknown"

    def _merge_normalized(self, packages: list[DatasetPackage], existing_df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        if existing_df is not None and not existing_df.empty:
            frames.append(existing_df)
        for pkg in packages:
            df = pkg.to_dataframe()
            if df is not None and not df.empty:
                for col in ["dataset_id", "provider", "document_type", "normalized_at"]:
                    if col not in df.columns:
                        df[col] = ""
                df["dataset_id"] = pkg.dataset_id
                df["provider"] = pkg.provider or pkg.source
                df["document_type"] = pkg.document_type
                df["normalized_at"] = datetime.now().isoformat()
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True, sort=False)
        return result.loc[:, ~result.columns.duplicated(keep="first")]

    def _save_artifact(self, packages: list[DatasetPackage]):
        rows = [p.to_dict() for p in packages]
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        path = self.save_artifact(df, "normalized_packages.csv", subdir="normalization")
        self.log.info("[%s] Package manifest → %s", self.agent_name, path)

    def _write_registry_export(self, packages: list[DatasetPackage]):
        rows = []
        for pkg in packages:
            rows.append({
                "dataset_id": pkg.dataset_id,
                "provider": pkg.provider or pkg.source,
                "document_type": pkg.document_type,
                "rows": pkg.row_count,
                "columns": pkg.column_count,
                "tables": len(pkg.tables),
                "documents": len(pkg.documents),
                "images": len(pkg.images),
                "is_valid": pkg.is_valid,
                "checksum": pkg.checksum,
                "license": pkg.license,
                "version": pkg.version,
            })
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            path = self.save_artifact(df, "dataset_registry_normalized.csv", subdir="normalization")
            self.log.info("[%s] Registry export → %s", self.agent_name, path)

    def run(self, df: pd.DataFrame, contract: Optional[AgentContract] = None, **kwargs) -> AgentContract:
        self.dataframe = df
        self.contract = contract or AgentContract(agent_name=self.agent_name)
        self.contract.status = "running"
        self.contract.started_at = datetime.now()
        try:
            result_df = self.process(df, **kwargs)
            self.dataframe = result_df
            self.contract.status = "success"
            self.contract.completed_at = datetime.now()
            self.contract.execution_time_sec = (
                self.contract.completed_at - self.contract.started_at
            ).total_seconds()
            self.contract.output_data = self._build_output(result_df, **kwargs)
            self.log.info(
                "[%s] Completed in %.2fs",
                self.agent_name, self.contract.execution_time_sec,
            )
        except Exception as e:
            self.contract.status = "failed"
            self.contract.completed_at = datetime.now()
            self.contract.errors.append(str(e))
            self.log.error("[%s] Failed: %s", self.agent_name, e)
        return self.contract
