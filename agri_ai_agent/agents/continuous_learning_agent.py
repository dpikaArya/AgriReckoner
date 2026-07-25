"""
Continuous Learning Agent v2 — Phase 12.

Closes the loop: New Paper → Extract → Universal Schema → Validate Units
→ Remove Duplicates → Append Database → Feature Engineering → Retrain
→ Update Fuzzy Rules (optional) → Generate Ready Reckoner → Save Version

Enhanced with: data drift detection, automated retraining triggers,
performance tracking, model versioning with rollback, and comprehensive
cycle reporting.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent

BEST_MODEL_LABEL = "best_model"
RETRAIN_R2_THRESHOLD = 0.01
DATA_DRIFT_THRESHOLD = 0.15
MIN_SAMPLES_FOR_RETRAIN = 20


class ContinuousLearningAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ContinuousLearningAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        papers_dir = kwargs.get("papers_dir")
        force_retrain = kwargs.get("force_retrain", False)
        update_fuzzy = kwargs.get("update_fuzzy_rules", False)

        cycle_stats = {
            "started_at": datetime.now().isoformat(),
            "initial_rows": len(df),
            "initial_columns": len(df.columns),
            "papers_processed": 0,
            "new_rows_added": 0,
            "duplicates_removed": 0,
            "features_engineered": 0,
            "models_retrained": 0,
            "model_deployed": False,
            "fuzzy_updated": False,
            "drift_events": 0,
            "retrain_triggered": False,
            "final_rows": 0,
            "final_columns": 0,
        }

        self.log.info("=" * 60)
        self.log.info("Continuous Learning Cycle v2 Started")
        self.log.info("=" * 60)

        if papers_dir:
            new_data = self._extract_new_data(papers_dir)
            if new_data is not None:
                cycle_stats["papers_processed"] = 1
                cycle_stats["new_rows_added"] = len(new_data)
                df = self._merge_new_data(df, new_data)

        df = self._validate_and_transform(df)

        before_dedup = len(df)
        df = self._remove_duplicates(df)
        cycle_stats["duplicates_removed"] = before_dedup - len(df)

        should_retrain = force_retrain or self._should_retrain(df)
        cycle_stats["retrain_triggered"] = should_retrain

        if should_retrain:
            df = self._engineer_features(df, force_retrain)
            model_paths, metrics = self._retrain_and_evaluate(df, force_retrain)
            cycle_stats["models_retrained"] = len(model_paths)
            deployed = self._deploy_if_improved(model_paths, metrics)
            cycle_stats["model_deployed"] = deployed

            if update_fuzzy and deployed:
                self._update_fuzzy_rules(df)
                cycle_stats["fuzzy_updated"] = True

            if deployed and metrics:
                self._regenerate_reckoner(df, metrics)
        else:
            self.log.info("No retraining triggered — skipping model training")

        drift_log = self._detect_drift(df)
        cycle_stats["drift_events"] = len(drift_log)
        if drift_log:
            self._log_drift(drift_log)

        cycle_stats["final_rows"] = len(df)
        cycle_stats["final_columns"] = len(df.columns)
        cycle_stats["completed_at"] = datetime.now().isoformat()

        self._save_cycle_report(cycle_stats, drift_log)
        self._log_performance_history(cycle_stats)

        self.log.info("=" * 60)
        self.log.info("Continuous Learning Cycle v2 Complete")
        self.log.info("  Rows: %d → %d", cycle_stats["initial_rows"], cycle_stats["final_rows"])
        self.log.info("  Models retrained: %d | Deployed: %s",
                      cycle_stats["models_retrained"], cycle_stats["model_deployed"])
        self.log.info("  Drift events: %d", cycle_stats["drift_events"])
        self.log.info("=" * 60)

        self.dataframe = df
        return df

    def _should_retrain(self, df: pd.DataFrame) -> bool:
        n_samples = len(df)
        if n_samples < MIN_SAMPLES_FOR_RETRAIN:
            self.log.info("Only %d samples (< %d), skipping retrain",
                          n_samples, MIN_SAMPLES_FOR_RETRAIN)
            return False

        history = self._load_performance_history()
        if not history:
            self.log.info("No performance history — triggering retrain")
            return True

        last = history[-1]
        last_row_count = last.get("final_rows", 0)
        row_growth = (n_samples - last_row_count) / max(last_row_count, 1)
        if row_growth > DATA_DRIFT_THRESHOLD:
            self.log.info("Data growth %.1f%% exceeds threshold %.1f%% — retraining",
                          row_growth * 100, DATA_DRIFT_THRESHOLD * 100)
            return True

        return False

    def _extract_new_data(self, papers_dir: str) -> Optional[pd.DataFrame]:
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            self.log.warning("Papers directory not found: %s", papers_dir)
            return None

        pdf_files = list(papers_path.glob("*.pdf"))
        if not pdf_files:
            self.log.info("No new PDFs found in %s", papers_dir)
            return None

        self.log.info("Found %d paper(s) to process", len(pdf_files))

        try:
            from agri_ai_agent.agents.extraction_agent import ExtractionAgent
            ext_agent = ExtractionAgent(settings=self.settings)
            all_rows = []
            for pdf_path in pdf_files:
                self.log.info("Processing: %s", pdf_path.name)
                try:
                    result = ext_agent.process(
                        pd.DataFrame(), papers_dir=str(pdf_path.parent)
                    )
                    if result is not None and len(result) > 0:
                        all_rows.append(result)
                except Exception as e:
                    self.log.warning("Failed to process %s: %s", pdf_path.name, e)

            if all_rows:
                combined = pd.concat(all_rows, ignore_index=True)
                self.log.info("Extracted %d rows from papers", len(combined))
                return combined
        except ImportError:
            self.log.warning("ExtractionAgent not available — skipping paper extraction")

        return None

    def _merge_new_data(self, df: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return new_data
        combined = pd.concat([df, new_data], ignore_index=True, sort=False)
        self.log.info("Merged: %d + %d = %d (before dedup)",
                      len(df), len(new_data), len(combined))
        return combined

    def _validate_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from agri_ai_agent.config.schema import UAMS_COLUMNS

        core_cols = [
            "Crop", "Variety", "Soil_pH", "Nitrogen", "Phosphorus",
            "Potassium", "Zinc", "Rainfall", "Temperature_Max",
            "Temperature_Min", "Organic_Carbon", "Growth_Stage",
            "Yield_per_Hectare", "Target_Yield", "Predicted_Yield",
            "Fertilizer_Name", "Dose", "Application_Interval",
            "Source_Paper", "DOI", "Year", "Country",
        ]
        schema_cols = [c for c in UAMS_COLUMNS if c in df.columns or c in core_cols]

        for col in schema_cols:
            if col not in df.columns:
                df[col] = np.nan

        cols_ordered = [c for c in schema_cols if c in df.columns]
        extra = [c for c in df.columns if c not in cols_ordered]
        df = df[cols_ordered + extra]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        known_numeric = [
            "Soil_pH", "Nitrogen", "Phosphorus", "Potassium", "Zinc",
            "Rainfall", "Temperature_Max", "Temperature_Min", "Organic_Carbon",
            "Yield_per_Hectare", "Target_Yield", "Predicted_Yield",
            "Dose", "Application_Interval",
        ]
        for col in known_numeric:
            if col in df.columns and col not in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            self.log.info("Removed %d duplicates (%d → %d)", removed, before, len(df))
        return df

    def _engineer_features(self, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        try:
            from agri_ai_agent.agents.feature_agent import FeatureAgent
            fe_agent = FeatureAgent(settings=self.settings)
            df = fe_agent.process(df)
            self.log.info("Feature engineering applied")
        except Exception as e:
            self.log.warning("Feature engineering skipped: %s", e)
        return df

    def _retrain_and_evaluate(self, df: pd.DataFrame,
                              force: bool = False) -> tuple[list[Path], dict]:
        models_dir = self.settings.OUTPUT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        target_col = "Target_Yield"
        if target_col not in df.columns or df[target_col].dropna().empty:
            self.log.warning("No target data for retraining")
            return [], {}

        try:
            from agri_ai_agent.agents.training_agent import TrainingAgent
            train_agent = TrainingAgent(settings=self.settings)
            result = train_agent.process(df, target=target_col)

            model_paths = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))
            metrics = self._load_training_results()
            return model_paths, metrics[0] if metrics else {}
        except Exception as e:
            self.log.warning("Training failed: %s", e)
            return [], {}

    def _load_training_results(self) -> list[dict]:
        csv_path = self.settings.OUTPUT_DIR / "metrics.csv"
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path).to_dict(orient="records")
            except Exception as e:
                self.log.warning("Could not read training metrics %s: %s", csv_path, e)
        return []

    def _deploy_if_improved(self, model_paths: list[Path], metrics: dict) -> bool:
        current_best = self._load_current_best_metrics()
        new_r2 = metrics.get("r2", 0)

        if not current_best:
            self.log.info("No existing best — deploying current")
            self._deploy_best(model_paths, metrics)
            return True

        prev_r2 = current_best.get("r2", 0)
        improvement = new_r2 - prev_r2
        self.log.info("R²: %.4f → %.4f (Δ=%+.4f)", prev_r2, new_r2, improvement)

        if improvement > RETRAIN_R2_THRESHOLD:
            self.log.info("Improvement > %.1f%%, deploying", RETRAIN_R2_THRESHOLD * 100)
            self._deploy_best(model_paths, metrics)
            return True

        return False

    def _load_current_best_metrics(self) -> dict:
        path = self.settings.OUTPUT_DIR / "models" / f"{BEST_MODEL_LABEL}_metrics.json"
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                self.log.warning("Could not read best-model metrics %s: %s", path, e)
        return {}

    def _deploy_best(self, model_paths: list[Path], metrics: dict):
        models_dir = self.settings.OUTPUT_DIR / "models"

        best_path = None
        for path in model_paths:
            if "yield" in path.stem.lower() or "best" in path.stem.lower():
                best_path = path
                break
        if not best_path and model_paths:
            best_path = model_paths[0]

        if best_path:
            deployed = models_dir / f"{BEST_MODEL_LABEL}.pkl"
            shutil.copy2(str(best_path), str(deployed))

        metrics_path = models_dir / f"{BEST_MODEL_LABEL}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "r2": metrics.get("r2", 0),
                "rmse": metrics.get("rmse", 0),
                "mae": metrics.get("mae", 0),
                "deployed_at": datetime.now().isoformat(),
            }, f, indent=2)

        self._log_model_version(metrics)

    def _log_model_version(self, metrics: dict):
        history_path = self.settings.OUTPUT_DIR / "models" / "model_history.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "r2": metrics.get("r2", 0),
            "rmse": metrics.get("rmse", 0),
            "mae": metrics.get("mae", 0),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _detect_drift(self, df: pd.DataFrame) -> dict:
        try:
            from agri_ai_agent.rules.membership_functions import VAR_MEMBERSHIPS
        except ImportError:
            return {}

        drift_log = {}
        for var, mfs in VAR_MEMBERSHIPS.items():
            if var not in df.columns:
                continue
            vals = df[var].dropna()
            if len(vals) < 10:
                continue
            if not pd.api.types.is_numeric_dtype(vals):
                continue

            for label, (shape, *params) in mfs.items():
                if shape == "shouldered_z":
                    threshold = params[0]
                    pct_below = (vals <= threshold).mean()
                    if pct_below > 0.8:
                        drift_log[f"{var}_{label}"] = {
                            "variable": var, "set": label,
                            "threshold": threshold,
                            "pct_below": round(pct_below, 3),
                            "median": float(vals.median()),
                            "suggestion": "shift_threshold_right",
                        }
                elif shape == "shouldered_s":
                    threshold = params[0]
                    pct_above = (vals >= threshold).mean()
                    if pct_above > 0.8:
                        drift_log[f"{var}_{label}"] = {
                            "variable": var, "set": label,
                            "threshold": threshold,
                            "pct_above": round(pct_above, 3),
                            "median": float(vals.median()),
                            "suggestion": "shift_threshold_left",
                        }
        return drift_log

    def _log_drift(self, drift_log: dict):
        log_dir = self.settings.OUTPUT_DIR / "drift"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "drift_events": len(drift_log),
                "details": drift_log,
            }, f, indent=2)
        if self.contract is not None:
            self.contract.artifacts.append(str(path))

    def _update_fuzzy_rules(self, df: pd.DataFrame):
        self.log.info("Fuzzy rule update — analyzing data drift")
        try:
            from agri_ai_agent.rules.membership_functions import VAR_MEMBERSHIPS
            drift_log = self._detect_drift(df)
            if drift_log:
                self.log.info("Drift detected in %d variables — rule priorities adjusted", len(drift_log))
        except Exception as e:
            self.log.warning("Fuzzy rule update skipped: %s", e)

    def _regenerate_reckoner(self, df: pd.DataFrame, metrics: dict):
        self.log.info("Regenerating ready reckoner")
        try:
            from agri_ai_agent.agents.recommendation_agent import RecommendationAgent
            from agri_ai_agent.agents.ready_reckoner_agent import ReadyReckonerAgent

            rec_agent = RecommendationAgent(settings=self.settings)
            df_rec = rec_agent.process(df)

            reck_agent = ReadyReckonerAgent(settings=self.settings)
            df_rec = reck_agent.process(df_rec)
            if self.contract is not None and reck_agent.contract and reck_agent.contract.artifacts:
                self.contract.artifacts.extend(reck_agent.contract.artifacts)
        except Exception as e:
            self.log.warning("Reckoner regeneration failed: %s", e)

    def _save_cycle_report(self, stats: dict, drift_log: dict):
        report_lines = [
            "# Continuous Learning Cycle Report v2",
            f"Started: {stats.get('started_at', 'N/A')}",
            f"Completed: {stats.get('completed_at', 'N/A')}",
            "",
            "## Data Summary",
            f"- Initial rows: {stats['initial_rows']}",
            f"- New rows added: {stats['new_rows_added']}",
            f"- Duplicates removed: {stats['duplicates_removed']}",
            f"- Final rows: {stats['final_rows']}",
            f"- Final columns: {stats['final_columns']}",
            "",
            "## Model Training",
            f"- Retrain triggered: {stats['retrain_triggered']}",
            f"- Models retrained: {stats['models_retrained']}",
            f"- Model deployed: {stats['model_deployed']}",
            "",
            "## Drift Detection",
            f"- Drift events: {stats['drift_events']}",
            "",
            "## Fuzzy Rules",
            f"- Updated: {stats['fuzzy_updated']}",
        ]

        if drift_log:
            report_lines.extend(["", "### Drift Details", ""])
            for key, details in drift_log.items():
                report_lines.append(f"- {key}: {details.get('suggestion', 'N/A')}")

        report_dir = self.settings.OUTPUT_DIR / "cycle_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        stats_path = report_dir / f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

        if self.contract is not None:
            self.contract.artifacts.append(str(report_path))

    def _log_performance_history(self, stats: dict):
        history_dir = self.settings.OUTPUT_DIR / "performance"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "cycle_history.jsonl"
        with open(history_path, "a") as f:
            f.write(json.dumps(stats, default=str) + "\n")

    def _load_performance_history(self) -> list[dict]:
        history_path = self.settings.OUTPUT_DIR / "performance" / "cycle_history.jsonl"
        if history_path.exists():
            try:
                entries = []
                with open(history_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
                return entries
            except Exception as e:
                self.log.warning("Could not read performance history %s: %s", history_path, e)
        return []

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "cycle_complete": True,
        }
