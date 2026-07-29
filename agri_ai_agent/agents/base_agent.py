"""
Base agent class with common functionality.
Every agent extends BaseAgent and implements run().

Downstream agents consume DatasetPackage as the standard exchange format.
Agents access tabular data via package.to_dataframe() internally.
"""

import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.utils.logging_utils import get_logger


class BaseAgent(ABC):
    def __init__(
        self,
        settings: Optional[AgriAISettings] = None,
        retry_max: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ):
        self.settings = settings or AgriAISettings()
        self.retry_max = retry_max if retry_max is not None else self.settings.AGENT_RETRY_MAX
        self.retry_delay = retry_delay if retry_delay is not None else self.settings.AGENT_RETRY_DELAY_SEC
        self.log = get_logger(self.__class__.__name__)
        self.contract: Optional[AgentContract] = None
        self.dataframe: Optional[pd.DataFrame] = None
        self.dataset_package: Optional[DatasetPackage] = None

    @property
    @abstractmethod
    def agent_name(self) -> str:
        ...

    @abstractmethod
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...

    def run(
        self,
        df: pd.DataFrame,
        contract: Optional[AgentContract] = None,
        **kwargs,
    ) -> AgentContract:
        self.dataframe = df
        self.contract = contract or AgentContract(agent_name=self.agent_name)
        self.contract.status = "running"
        self.contract.started_at = datetime.now()

        for attempt in range(1, self.retry_max + 1):
            try:
                self.contract.retry_count = attempt - 1
                self.log.info("[%s] Starting (attempt %d/%d)", self.agent_name, attempt, self.retry_max)

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
                return self.contract

            except Exception as e:
                self.log.warning(
                    "[%s] Attempt %d failed: %s", self.agent_name, attempt, e
                )
                self.contract.errors.append(f"Attempt {attempt}: {traceback.format_exc()}")

                if attempt < self.retry_max:
                    time.sleep(self.retry_delay)
                else:
                    self.contract.status = "failed"
                    self.contract.completed_at = datetime.now()
                    self.log.error("[%s] All %d attempts failed", self.agent_name, self.retry_max)
                    return self.contract

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "column_count": len(df.columns),
        }

    def _resolve_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate-named columns, keeping the first occurrence of each."""
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        return df

    def _normalize(self, name: str) -> str:
        import re
        n = str(name).lower().strip()
        n = re.sub(r"\s+", "_", n)
        n = n.replace("-", "_").replace(".", "")
        return n

    def save_artifact(self, df: pd.DataFrame, filename: str, subdir: str = "") -> Path:
        out = self.settings.OUTPUT_DIR
        if subdir:
            out = out / subdir
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        if filename.endswith(".csv"):
            df.to_csv(path, index=False, encoding="utf-8-sig")
        elif filename.endswith((".xlsx", ".xls")):
            df.to_excel(path, index=False, engine="openpyxl")
        elif filename.endswith(".json"):
            df.to_json(path, orient="records", indent=2)
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        if self.contract is not None:
            self.contract.artifacts.append(str(path))
        return path

    def save_text_artifact(self, text: str, filename: str, subdir: str = "") -> Path:
        out = self.settings.OUTPUT_DIR
        if subdir:
            out = out / subdir
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        path.write_text(text, encoding="utf-8")
        if self.contract is not None:
            self.contract.artifacts.append(str(path))
        return path
