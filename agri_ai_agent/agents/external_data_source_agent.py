import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.external_data.connector_manager import (
    ConnectorManager,
)
from agri_ai_agent.external_data.data_enricher import enrich_master
from agri_ai_agent.external_data.registry_db import DatasetRegistry
from agri_ai_agent.knowledge_graph.graph import KnowledgeGraph
from agri_ai_agent.knowledge_graph.provenance import (
    register_dataset_lineage,
    write_provenance_artifact,
    update_knowledge_graph,
)


class ExternalDataSourceAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ExternalDataSourceAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sources = kwargs.get("sources", None)
        download_dir = kwargs.get("download_dir", self.settings.EXTERNAL_DATA_DIR)
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        registry_path = kwargs.get("registry_path", self.settings.DATASET_REGISTRY_PATH)
        registry = DatasetRegistry(registry_path)

        use_integration = kwargs.get("use_integration",
                                     os.getenv("AGRI_USE_DATA_SOURCE_INTEGRATION", "false").lower() == "true")

        if use_integration:
            from src.data_sources.integration import DataSourceIntegration
            integration = DataSourceIntegration(
                registry=registry,
                download_dir=download_dir,
                storage_dir=kwargs.get("storage_dir"),
                max_workers=kwargs.get("max_workers"),
                retry_max_attempts=kwargs.get("retry_max_attempts"),
                retry_base_delay=kwargs.get("retry_base_delay"),
            )
            integration_impl = integration
            available = integration_impl.list_sources()
        else:
            manager = ConnectorManager(
                registry=registry,
                download_dir=download_dir,
                max_workers=kwargs.get("max_workers", 4),
                retry_max_attempts=kwargs.get("retry_max_attempts", 3),
                retry_base_delay=kwargs.get("retry_base_delay", 2.0),
            )
            integration_impl = manager
            available = manager.list_sources()

        self.log.info("Discovered %d external data sources: %s", len(available), available)

        selected = sources if sources is not None else available
        if not selected:
            self.log.warning("No sources selected, skipping external data layer")
            return df if df is not None else pd.DataFrame()

        self.log.info("Running health checks on %d source(s)...", len(selected))
        health_results = integration_impl.health_checks(selected)
        healthy = [s for s, h in health_results.items() if h.is_healthy]
        unhealthy = [s for s, h in health_results.items() if not h.is_healthy]
        if unhealthy:
            self.log.warning("Unhealthy sources (skipped): %s", unhealthy)
        if not healthy:
            self.log.warning("No healthy sources available")
            return df if df is not None else pd.DataFrame()

        self.log.info(
            "Running %d healthy connector(s) concurrently...",
            len(healthy),
        )
        packages_by_source, run_logs = integration_impl.run_all(sources=healthy)

        summary = integration_impl.summarize_runs(run_logs)
        self.log.info(
            "Connector run complete: %d succeeded, %d failed, "
            "%d datasets downloaded, %d valid, %.1fs total",
            summary["succeeded"], summary["failed"],
            summary["total_datasets_downloaded"],
            summary["total_datasets_valid"],
            summary["total_duration_sec"],
        )

        all_packages = []
        results_rows = []

        provenance_dir = Path(kwargs.get("provenance_dir", download_dir / "provenance"))
        kg: Optional[KnowledgeGraph] = kwargs.get("knowledge_graph")

        for src, pkgs in packages_by_source.items():
            for pkg in pkgs:
                all_packages.append(pkg)
                try:
                    write_provenance_artifact(provenance_dir, pkg, connector_name=src)
                    if kg is not None:
                        update_knowledge_graph(kg, pkg, connector_name=src)
                    self.contract.dataset_id = pkg.dataset_id
                    self.contract.provider = pkg.provider or pkg.source
                    self.contract.connector_name = src
                    self.contract.checksum = pkg.checksum
                    self.contract.processing_stage = "external_data"
                    self.contract.processing_mode = "download"
                except Exception as exc:
                    self.log.warning("[%s] provenance tracking failed for %s/%s: %s",
                                     self.agent_name, src, pkg.resource_id, exc)
                results_rows.append({
                    "source": src,
                    "resource_id": pkg.resource_id,
                    "name": pkg.name,
                    "rows": pkg.row_count,
                    "columns": pkg.column_count,
                    "is_valid": pkg.is_valid,
                    "version": pkg.version,
                    "download_path": str(pkg.download_path) if pkg.download_path else "",
                })

        result_df = pd.DataFrame(results_rows) if results_rows else pd.DataFrame()
        summary_path = self.save_artifact(
            result_df, "external_data_sources.csv", subdir="external_data"
        )
        self.log.info("External data source summary written to %s", summary_path)

        if hasattr(integration_impl, "enrich_packages"):
            merged = integration_impl.enrich_packages(packages_by_source, df)
            enrich_mode = "enriched (key-based join)"
        else:
            merged = integration_impl.merge_packages(packages_by_source, df)
            enrich_mode = "merged (row append)"
        self.log.info(
            "External data layer complete: %d packages from %d sources %s into %d columns",
            len(all_packages), len(healthy), enrich_mode, len(merged.columns),
        )
        return merged

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
