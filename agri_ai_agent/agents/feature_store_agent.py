from datetime import datetime
from pathlib import Path

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract

OBSERVATION_ID_COLS = [
    "Dataset_ID",
    "Document_ID",
    "Paper_ID",
    "Experiment_ID",
    "Treatment_ID",
    "Observation_ID",
]


class FeatureStoreAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "FeatureStoreAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df is None or df.empty:
            self.log.warning("[%s] Empty DataFrame, nothing to store", self.agent_name)
            return pd.DataFrame()

        validated = self._filter_validated(df, **kwargs)
        stored_count = self._store_observations(validated, **kwargs)

        self.log.info(
            "[%s] Stored %d validated observation(s); passing %d to Feature Engineering",
            self.agent_name,
            stored_count,
            len(validated),
        )

        return validated

    def _filter_validated(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "_validation_status" in df.columns:
            valid = df[df["_validation_status"] == "ACCEPTED"].copy()
            if len(valid) < len(df):
                rejected = len(df) - len(valid)
                self.log.warning(
                    "[%s] Filtered out %d rejected observation(s)",
                    self.agent_name,
                    rejected,
                )
            return valid

        if "_is_valid" in df.columns:
            valid = df[df["_is_valid"]].copy()
            if len(valid) < len(df):
                self.log.warning(
                    "[%s] Filtered out %d invalid observation(s)",
                    self.agent_name,
                    len(df) - len(valid),
                )
            return valid

        return df.copy()

    def _store_observations(self, df: pd.DataFrame, **kwargs) -> int:
        store_dir = Path(
            kwargs.get("feature_store_dir", self.settings.OUTPUT_DIR / "feature_store")
        )
        store_dir.mkdir(parents=True, exist_ok=True)

        present_id_cols = [c for c in OBSERVATION_ID_COLS if c in df.columns]
        manifest_rows = []

        if present_id_cols:
            grouped = df.groupby(
                [c for c in present_id_cols if c in ["Dataset_ID", "Paper_ID", "Experiment_ID"]],
                sort=False,
            )
            for key, group in grouped:
                if isinstance(key, tuple):
                    str(key[0]) if len(key) > 0 else "unknown"
                    exp_part = "_".join(str(k) for k in key)
                else:
                    str(key)
                    exp_part = str(key)

                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in exp_part)
                path = store_dir / f"observations_{safe_name}.parquet"
                group.to_parquet(path, index=False)

                manifest_rows.append(
                    {
                        "store_path": str(path),
                        "observation_count": len(group),
                        "experiment_key": exp_part,
                    }
                )
        else:
            path = store_dir / "observations.parquet"
            df.to_parquet(path, index=False)
            manifest_rows.append(
                {
                    "store_path": str(path),
                    "observation_count": len(df),
                    "experiment_key": "all",
                }
            )

        if manifest_rows:
            manifest_df = pd.DataFrame(manifest_rows)
            self.save_artifact(manifest_df, "feature_store_manifest.csv", subdir="feature_store")

        return len(df)

    def run(
        self, df: pd.DataFrame, contract: AgentContract | None = None, **kwargs
    ) -> AgentContract:
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
                self.agent_name,
                self.contract.execution_time_sec,
            )
        except Exception as e:
            self.contract.status = "failed"
            self.contract.completed_at = datetime.now()
            self.contract.errors.append(str(e))
            self.log.error("[%s] Failed: %s", self.agent_name, e)
        return self.contract
