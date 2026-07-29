import json
import time
from datetime import datetime

import pandas as pd

from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.continuous_learning.repository_registry import RepositoryRegistry
from agri_ai_agent.continuous_learning.version_history import VersionHistory
from agri_ai_agent.continuous_learning.change_detector import ChangeDetector, ChangeSet
from agri_ai_agent.continuous_learning.dependency_graph import DependencyGraph, ALL_USER_STAGES
from agri_ai_agent.continuous_learning.reports import (
    generate_sync_report,
    generate_provenance_report,
    generate_retraining_report,
)
from agri_ai_agent.utils.logging_utils import get_logger


logger = get_logger("IncrementalEngine")


class IncrementalEngine:
    def __init__(
        self,
        settings: AgriAISettings,
        registry: RepositoryRegistry,
        history: VersionHistory,
        detector: ChangeDetector,
        graph: DependencyGraph,
    ):
        self._settings = settings
        self._registry = registry
        self._history = history
        self._detector = detector
        self._graph = graph

    def execute_cycle(self, df: pd.DataFrame,
                      run_pipeline_fn=None) -> dict:
        cycle_start = time.perf_counter()
        sync_id = self._history.start_sync()
        cycle_stats: dict = {
            "started_at": datetime.now().isoformat(),
            "sync_id": sync_id,
            "repos_checked": 0,
            "repos_changed": 0,
            "stages_executed": [],
            "stages_skipped": [],
            "data_loaded": False,
            "features_engineered": False,
            "drift_detected": False,
            "model_retrained": False,
            "reckoner_updated": False,
            "duration_sec": 0.0,
            "completed_at": "",
        }

        changes: ChangeSet = self._detector.check_all()
        repos_checked = len(self._registry.list_enabled())
        repos_changed = len(changes.changed_repos)
        cycle_stats["repos_checked"] = repos_checked
        cycle_stats["repos_changed"] = repos_changed

        if not changes.has_changes:
            logger.info("No changes detected — skipping pipeline")
            cycle_stats["stages_executed"] = []
            cycle_stats["stages_skipped"] = list(ALL_USER_STAGES)
            cycle_stats["completed_at"] = datetime.now().isoformat()
            cycle_stats["duration_sec"] = round(time.perf_counter() - cycle_start, 2)
            self._history.complete_sync(
                sync_id, repos_checked, repos_changed,
                [], cycle_stats,
            )
            generate_sync_report(changes, [], self._settings.OUTPUT_DIR)
            return cycle_stats

        logger.info("Changes detected in %d repo(s): %s",
                     repos_changed, changes.changed_repos)

        affected_stages = self._graph.get_affected_stages(changes)
        logger.info("Affected stages: %s", affected_stages)

        cycle_stats["stages_executed"] = list(affected_stages)

        if run_pipeline_fn:
            for step_key in self._graph.get_pipeline_steps_for_stages(affected_stages):
                logger.info("Executing pipeline step: %s", step_key)
                try:
                    result = run_pipeline_fn(step_key, df)
                except Exception as e:
                    logger.error("Pipeline step %s failed: %s", step_key, e)

        sync_stages = cycle_stats.get("stages_executed", [])
        executed_pipeline = self._graph.get_pipeline_steps_for_stages(sync_stages)
        self._history.complete_sync(
            sync_id, repos_checked, repos_changed,
            executed_pipeline, cycle_stats,
        )

        for repo_name in changes.changed_repos:
            self._registry.mark_synced(repo_name, changes.timestamp)

        cycle_stats["completed_at"] = datetime.now().isoformat()
        cycle_stats["duration_sec"] = round(time.perf_counter() - cycle_start, 2)

        generate_sync_report(changes, sync_stages, self._settings.OUTPUT_DIR)
        generate_provenance_report(self._history, self._settings.OUTPUT_DIR)
        if cycle_stats.get("model_retrained"):
            generate_retraining_report(cycle_stats, self._settings.OUTPUT_DIR)

        self._log_cycle(cycle_stats)
        return cycle_stats

    def _log_cycle(self, stats: dict):
        log_dir = self._settings.OUTPUT_DIR / "continuous_learning"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "cycle_history.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(stats, default=str) + "\n")
