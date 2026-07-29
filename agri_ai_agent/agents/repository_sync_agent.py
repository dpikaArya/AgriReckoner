from datetime import datetime

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.continuous_learning.repository_registry import RepositoryRegistry
from agri_ai_agent.continuous_learning.version_history import VersionHistory
from agri_ai_agent.continuous_learning.change_detector import ChangeDetector, ChangeSet
from agri_ai_agent.continuous_learning.dependency_graph import DependencyGraph
from agri_ai_agent.continuous_learning.reports import (
    generate_sync_report,
    generate_provenance_report,
)


class RepositorySyncAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "RepositorySyncAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        db_path = kwargs.get("continuous_learning_db", self.settings.CONTINUOUS_LEARNING_DB)
        registry = RepositoryRegistry(db_path)
        history = VersionHistory(db_path)

        auto_register = kwargs.get("auto_register_repos", [])
        for repo_cfg in auto_register:
            registry.register(
                name=repo_cfg.get("name", ""),
                url=repo_cfg.get("url", ""),
                repo_type=repo_cfg.get("type", "generic"),
                connector_name=repo_cfg.get("connector", ""),
                metadata=repo_cfg.get("metadata"),
            )

        detector = ChangeDetector(registry, history)
        changes: ChangeSet = detector.check_all()

        graph = DependencyGraph()
        affected_stages = graph.get_affected_stages(changes) if changes.has_changes else []

        self.log.info(
            "Checked %d repo(s), %d changed, %d affected stage(s)",
            len(registry.list_enabled()), len(changes.changed_repos), len(affected_stages),
        )

        self.contract.output_data["repos_checked"] = len(registry.list_enabled())
        self.contract.output_data["repos_changed"] = len(changes.changed_repos)
        self.contract.output_data["affected_stages"] = affected_stages
        self.contract.output_data["has_changes"] = changes.has_changes
        self.contract.output_data["change_set"] = changes.to_dict()

        if changes.has_changes:
            for repo_name in changes.changed_repos:
                registry.mark_synced(repo_name, changes.timestamp)
            generate_sync_report(changes, affected_stages, self.settings.OUTPUT_DIR)
            generate_provenance_report(history, self.settings.OUTPUT_DIR)

        registry.close()
        history.close()

        if not changes.has_changes:
            self.log.info("No changes — skipping downstream pipeline")

        df_out = df if df is not None else pd.DataFrame()
        df_out = df_out.copy()
        df_out["_sync_timestamp"] = datetime.now().isoformat()
        df_out["_has_changes"] = changes.has_changes
        return df_out

    def run(self, df: pd.DataFrame, contract=None, **kwargs) -> AgentContract:
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
