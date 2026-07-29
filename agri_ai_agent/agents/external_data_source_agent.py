from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.external_data.registry import ConnectorRegistry
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class ExternalDataSourceAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ExternalDataSourceAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sources = kwargs.get("sources", None)
        download_dir = kwargs.get("download_dir", self.settings.EXTERNAL_DATA_DIR)
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        ConnectorRegistry.discover()
        available = ConnectorRegistry.list_sources()
        self.log.info("Discovered %d external data sources: %s", len(available), available)

        selected = sources if sources else available
        all_packages: list[DatasetPackage] = []
        results = []

        for source_name in selected:
            connector = ConnectorRegistry.instantiate(source_name)
            if connector is None:
                self.log.warning("Connector not found for source: %s", source_name)
                continue

            self.log.info("Connecting to %s...", source_name)
            connected = connector.connect()
            if not connected:
                self.log.warning("Cannot connect to %s, skipping", source_name)
                continue

            self.log.info("Discovering datasets from %s...", source_name)
            datasets = connector.discover()
            self.log.info("Found %d datasets from %s", len(datasets), source_name)

            for ds in datasets:
                resource_id = ds.get("id", "")
                if not resource_id:
                    continue
                self.log.info("Downloading %s from %s...", resource_id, source_name)
                try:
                    package = connector.fetch(resource_id, download_dir / source_name)
                    if package is not None:
                        all_packages.append(package)
                        results.append({
                            "source": source_name,
                            "resource_id": resource_id,
                            "name": package.name,
                            "rows": package.row_count,
                            "columns": package.column_count,
                            "is_valid": package.is_valid,
                            "download_path": str(package.download_path) if package.download_path else "",
                        })
                        self.log.info(
                            "Successfully fetched %s from %s (%d rows, %d cols)",
                            resource_id, source_name, package.row_count, package.column_count,
                        )
                except Exception as e:
                    self.log.error("Error fetching %s from %s: %s", resource_id, source_name, e)

            connector.close()

        result_df = pd.DataFrame(results) if results else pd.DataFrame()
        summary_path = self.save_artifact(
            result_df, "external_data_sources.csv", subdir="external_data"
        )
        self.log.info("External data source summary written to %s", summary_path)

        merged = self._merge_packages(all_packages, df)
        self.log.info(
            "External data layer complete: %d packages from %d sources merged into %d rows",
            len(all_packages), len(selected), len(merged),
        )
        return merged

    def _merge_packages(
        self, packages: list[DatasetPackage], existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        if not packages:
            return existing_df if existing_df is not None else pd.DataFrame()

        frames = [existing_df] if existing_df is not None else []
        for pkg in packages:
            df = pkg.to_dataframe()
            if df is not None and not df.empty:
                for col in ["source", "resource_id", "download_timestamp"]:
                    if col not in df.columns:
                        df[col] = ""
                df["source"] = pkg.source
                df["resource_id"] = pkg.resource_id
                df["download_timestamp"] = datetime.now().isoformat()
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True, sort=False)
        result = result.loc[:, ~result.columns.duplicated(keep="first")]
        return result

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
