import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.contracts.messages import AgentContract


OBS_MANDATORY_IDS = [
    "Dataset_ID", "Document_ID", "Paper_ID",
    "Experiment_ID", "Treatment_ID", "Observation_ID",
]


class ObservationGenerationAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ObservationGenerationAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        df = self._ensure_id_columns(df)
        df = self._normalize_observation_hierarchy(df)
        df = self._deduplicate_treatments(df)

        obs_count = len(df)
        self.log.info(
            "[%s] Generated %d observation(s) across %d paper(s) and %d experiment(s)",
            self.agent_name, obs_count,
            df["Paper_ID"].nunique() if "Paper_ID" in df.columns else 0,
            df["Experiment_ID"].nunique() if "Experiment_ID" in df.columns else 0,
        )

        manifest = df[OBS_MANDATORY_IDS].drop_duplicates() if all(
            c in df.columns for c in OBS_MANDATORY_IDS
        ) else pd.DataFrame()
        if not manifest.empty:
            self.save_artifact(manifest, "observation_manifest.csv", subdir="observation")
        return df

    def _ensure_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in OBS_MANDATORY_IDS:
            if col not in df.columns:
                df[col] = ""
        return df

    def _normalize_observation_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Paper_ID" not in df.columns:
            df["Paper_ID"] = "PAPER/DEFAULT"

        paper_with_no_ds = df["Paper_ID"] == "" 
        if paper_with_no_ds.any():
            df.loc[paper_with_no_ds, "Paper_ID"] = df.loc[paper_with_no_ds].apply(
                lambda _: f"PAPER/{uuid.uuid4().hex[:8]}", axis=1,
            )

        if "Experiment_ID" not in df.columns or df["Experiment_ID"].isna().all() or (df["Experiment_ID"] == "").all():
            exp_cols = [c for c in ["Location", "Site", "State", "Season", "Year", "Design", "Crop"]
                       if c in df.columns]
            if exp_cols:
                df["Experiment_ID"] = df.apply(
                    lambda r: f"{r['Paper_ID']}/exp/"
                              f"{'_'.join(str(r.get(c, '')).strip() for c in exp_cols if pd.notna(r.get(c, '')))}",
                    axis=1,
                )
            else:
                df["Experiment_ID"] = df.apply(
                    lambda r: f"{r['Paper_ID']}/exp/0", axis=1,
                )

        if "Treatment_ID" not in df.columns or df["Treatment_ID"].isna().all() or (df["Treatment_ID"] == "").all():
            trt_cols = [c for c in ["Treatment", "Fertilizer_Name", "Dose", "Variety"]
                       if c in df.columns]
            if trt_cols:
                df["Treatment_ID"] = df.apply(
                    lambda r: f"{r['Experiment_ID']}/trt/"
                              f"{'_'.join(str(r.get(c, '')).strip() for c in trt_cols if pd.notna(r.get(c, '')))}",
                    axis=1,
                )
            else:
                df["Treatment_ID"] = df.apply(
                    lambda r: f"{r['Experiment_ID']}/trt/{uuid.uuid4().hex[:6]}", axis=1,
                )

        if "Dataset_ID" not in df.columns or df["Dataset_ID"].isna().all() or (df["Dataset_ID"] == "").all():
            df["Dataset_ID"] = df["Paper_ID"].apply(lambda p: p.replace("/paper", ""))

        if "Document_ID" not in df.columns or df["Document_ID"].isna().all() or (df["Document_ID"] == "").all():
            df["Document_ID"] = df["Dataset_ID"].apply(lambda d: f"{d}/doc/0")

        return df

    def _deduplicate_treatments(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = [c for c in ["Paper_ID", "Experiment_ID", "Treatment_ID"]
                     if c in df.columns]

        if not group_cols or len(group_cols) < 3:
            if "Observation_ID" not in df.columns or df["Observation_ID"].isna().all() or (df["Observation_ID"] == "").all():
                df["Observation_ID"] = [f"OBS/{uuid.uuid4().hex[:12]}" for _ in range(len(df))]
            return df

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        fused_rows = []
        for _, group in df.groupby(group_cols, sort=False):
            if len(group) == 1:
                row = group.iloc[0].to_dict()
                row["_replicates"] = 1
                fused_rows.append(row)
                continue

            first = group.iloc[0].to_dict()
            for col in numeric_cols:
                vals = group[col].dropna()
                if not vals.empty:
                    first[col] = vals.mean()
            for col in cat_cols:
                if col not in group_cols:
                    first_non_null = group[col].dropna()
                    if not first_non_null.empty:
                        first[col] = first_non_null.iloc[0]
            first["_replicates"] = len(group)
            fused_rows.append(first)

        result = pd.DataFrame(fused_rows) if fused_rows else df
        if "Observation_ID" not in result.columns or result["Observation_ID"].isna().all() or (result["Observation_ID"] == "").all():
            result["Observation_ID"] = [
                f"OBS/{uuid.uuid4().hex[:12]}" for _ in range(len(result))
            ]
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
            if all(c in result_df.columns for c in ["Observation_ID", "Experiment_ID", "Treatment_ID"]):
                self.contract.output_data["observation_count"] = len(result_df)
                self.contract.output_data["experiment_count"] = int(result_df["Experiment_ID"].nunique())
                self.contract.output_data["treatment_count"] = int(result_df["Treatment_ID"].nunique())
            self.log.info(
                "[%s] Completed in %.2fs", self.agent_name, self.contract.execution_time_sec,
            )
        except Exception as e:
            self.contract.status = "failed"
            self.contract.completed_at = datetime.now()
            self.contract.errors.append(str(e))
            self.log.error("[%s] Failed: %s", self.agent_name, e)
        return self.contract
