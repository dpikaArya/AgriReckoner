import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class DatasetIngestionBridgeAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "DatasetIngestionBridgeAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        packages: list[DatasetPackage] = kwargs.get("packages") or []
        existing = df if df is not None and not df.empty else None

        if not packages and existing is None:
            return pd.DataFrame()

        bridge_rows = []

        for pkg in packages:
            dataset_id = pkg.dataset_id
            provider = pkg.provider or pkg.source
            doc_id = f"{dataset_id}/doc/0"
            paper_id = f"{dataset_id}/paper"

            table = pkg.to_dataframe()
            if table is None or table.empty:
                self.log.warning("[%s] Package %s has no data, skipping", self.agent_name, dataset_id)
                continue

            exp_groups = self._detect_experiment_groups(table)
            experiment_counter = 0

            for exp_key, exp_indices in exp_groups.items():
                experiment_id = f"{dataset_id}/exp/{experiment_counter}"
                experiment_counter += 1
                exp_slice = table.iloc[exp_indices]
                treatment_counter = 0

                for _, row in exp_slice.iterrows():
                    treatment_key = self._treatment_key(row)
                    treatment_id = f"{experiment_id}/trt/{treatment_counter}"
                    treatment_counter += 1
                    obs_id = f"OBS/{dataset_id}/{uuid.uuid4().hex[:12]}"

                    row_dict = row.to_dict()
                    row_dict["Dataset_ID"] = dataset_id
                    row_dict["Document_ID"] = doc_id
                    row_dict["Paper_ID"] = paper_id
                    row_dict["Experiment_ID"] = experiment_id
                    row_dict["Treatment_ID"] = treatment_id
                    row_dict["Observation_ID"] = obs_id
                    row_dict["_provider"] = provider
                    row_dict["_bridge_version"] = pkg.version
                    bridge_rows.append(row_dict)

        bridge_df = pd.DataFrame(bridge_rows) if bridge_rows else pd.DataFrame()
        self.log.info(
            "[%s] Bridged %d package(s) → %d rows with IDs",
            self.agent_name, len(packages), len(bridge_df),
        )

        if existing is not None:
            result = pd.concat([existing, bridge_df], ignore_index=True, sort=False)
        else:
            result = bridge_df

        result = result.loc[:, ~result.columns.duplicated(keep="first")]
        return result

    def _detect_experiment_groups(self, df: pd.DataFrame) -> dict[str, list[int]]:
        partition_cols = [c for c in ["Location", "Site", "State", "Season", "Year", "Design", "Crop"]
                         if c in df.columns]
        if not partition_cols:
            return {"default": list(range(len(df)))}

        grouped = df.groupby(partition_cols, sort=False)
        result: dict[str, list[int]] = {}
        for key, group in grouped:
            if isinstance(key, tuple):
                label = "_".join(str(v) for v in key)
            else:
                label = str(key)
            result[label] = list(group.index)
        return result

    def _treatment_key(self, row: pd.Series) -> str:
        parts = []
        for col in ["Treatment", "Fertilizer_Name", "Dose", "Variety"]:
            val = row.get(col)
            if val is not None and pd.notna(val) and str(val).strip():
                parts.append(f"{col}={val}")
        return ";".join(parts) if parts else "default_trt"

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
                "[%s] Completed in %.2fs", self.agent_name, self.contract.execution_time_sec,
            )
        except Exception as e:
            self.contract.status = "failed"
            self.contract.completed_at = datetime.now()
            self.contract.errors.append(str(e))
            self.log.error("[%s] Failed: %s", self.agent_name, e)
        return self.contract
