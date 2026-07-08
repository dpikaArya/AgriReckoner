"""
Continuous Learning Agent
Closes the loop: New Paper → Extract → Universal Schema → Validate Units
→ Remove Duplicates → Append Database → Feature Engineering → Retrain
→ Update Fuzzy Rules (optional) → Generate Ready Reckoner → Save Version
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract

BEST_MODEL_LABEL = "best_model"


class ContinuousLearningAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ContinuousLearningAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        papers_dir = kwargs.get("papers_dir")
        force_retrain = kwargs.get("force_retrain", False)
        update_fuzzy = kwargs.get("update_fuzzy_rules", False)

        self.log.info("=" * 60)
        self.log.info("Continuous Learning Cycle Started")
        self.log.info("=" * 60)

        # Step 1: Read paper & extract tables
        if papers_dir:
            new_data = self._extract_new_data(papers_dir)
            if new_data is not None:
                df = self._merge_new_data(df, new_data)

        # Step 2: Convert to Universal Schema & validate units
        df = self._validate_and_transform(df)

        # Step 4: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 5: Recalculate engineered features
        df = self._engineer_features(df, force_retrain)

        # Step 6: Retrain ML models
        model_paths, metrics = self._retrain_and_evaluate(df, force_retrain)

        # Step 7: Deploy if improved
        deployed = self._deploy_if_improved(model_paths, metrics)

        # Step 8: Update fuzzy rules if enabled
        if update_fuzzy and deployed:
            self._update_fuzzy_rules(df)

        # Step 9: Generate new Ready Reckoner
        if deployed and metrics:
            self._regenerate_reckoner(df, metrics)

        # Step 10: Save new model version (done inside _deploy_best)

        self.log.info("=" * 60)
        self.log.info("Continuous Learning Cycle Complete")
        if deployed:
            self.log.info("New model deployed (metrics improved)")
        else:
            self.log.info("Existing model retained (no improvement)")
        self.log.info("=" * 60)

        self.dataframe = df
        return df

    # ── Step 1: Read Paper & Extract Tables ──────────────────────────

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

        from agri_ai_agent.agents.agent01_document_understanding import DocumentUnderstandingAgent
        from agri_ai_agent.agents.agent02_scientific_extraction import ScientificInformationExtractionAgent

        du_agent = DocumentUnderstandingAgent(settings=self.settings)
        se_agent = ScientificInformationExtractionAgent(settings=self.settings)

        all_rows = []
        for pdf_path in pdf_files:
            self.log.info("Processing: %s", pdf_path.name)
            try:
                du_result = du_agent.run(filepath=str(pdf_path))
                if du_result.status != "success":
                    self.log.warning("Doc understanding failed for %s", pdf_path.name)
                    continue

                se_result = se_agent.run(filepath=str(pdf_path))
                if se_result.status != "success":
                    self.log.warning("Extraction failed for %s", pdf_path.name)
                    continue

                extracted = se_result.variables_extracted
                for var in extracted:
                    var["Source_Paper"] = pdf_path.stem
                    var["DOI"] = (
                        du_result.structured_json_path
                        if hasattr(du_result, "structured_json_path")
                        else ""
                    )
                all_rows.extend(extracted)
            except Exception as e:
                self.log.warning("Failed to process %s: %s", pdf_path.name, e)

        if not all_rows:
            self.log.info("No data extracted from papers")
            return None

        new_df = pd.DataFrame(all_rows)
        self.log.info("Extracted %d rows from %d paper(s)", len(new_df), len(pdf_files))
        return new_df

    def _merge_new_data(self, df: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.log.info("Starting fresh dataset with extracted data (%d rows)", len(new_data))
            return new_data

        combined = pd.concat([df, new_data], ignore_index=True, sort=False)
        self.log.info("Merged: %d existing + %d new = %d total (before dedup)",
                      len(df), len(new_data), len(combined))
        return combined

    # ── Step 2: Convert to Universal Schema & Validate Units ─────────

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
        self.log.info("Schema aligned: %d columns (%d core + %d extra)",
                      len(df.columns), len(cols_ordered), len(extra))

        df = self._validate_units(df)

        return df

    def _validate_units(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from agri_ai_agent.agents.agent07_unit_harmonization import UnitHarmonizationAgent
            uh_agent = UnitHarmonizationAgent(settings=self.settings)
            uh_agent.run(df)
            if uh_agent.dataframe is not None:
                self.log.info("Unit harmonization applied")
                return uh_agent.dataframe
        except Exception as e:
            self.log.warning("Unit harmonization skipped: %s", e)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        self.log.info("Basic numeric coercion applied")
        return df

    # ── Step 3: Remove Duplicates ─────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        dupe_count = df.duplicated().sum()
        if dupe_count:
            df = df.drop_duplicates()
            self.log.info("Removed %d duplicate rows (%d → %d)", dupe_count, before, len(df))
        else:
            self.log.info("No duplicates found (%d rows)", before)
        return df

    # ── Step 4: Recalculate Engineered Features ──────────────────────

    def _engineer_features(self, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        from agri_ai_agent.agents.agent09_feature_engineering import FeatureEngineeringAgent

        try:
            fe_agent = FeatureEngineeringAgent(settings=self.settings)
            fe_agent.run(df)
            if fe_agent.dataframe is not None:
                df = fe_agent.dataframe
                self.log.info("Feature engineering applied")
        except Exception as e:
            self.log.warning("Feature engineering skipped: %s", e)

        return df

    # ── Step 5: Retrain ML Models ─────────────────────────────────────

    def _retrain_and_evaluate(self, df: pd.DataFrame,
                              force: bool = False) -> tuple[list[Path], dict]:
        from agri_ai_agent.agents.agent14_training import TrainingAgent

        models_dir = self.settings.OUTPUT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        target_col = "Target_Yield"
        if target_col not in df.columns or df[target_col].dropna().empty:
            self.log.warning("Target column '%s' has no data; skipping retrain", target_col)
            return [], {}

        clean_df = df.dropna(axis=1, how="all").copy()

        train_agent = TrainingAgent(settings=self.settings)
        train_agent.run(clean_df, target=target_col)

        results = self._load_training_results()
        metrics = results[0] if results else {}

        model_paths = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))
        self.log.info("Retrained %d model(s)", len(model_paths))

        return model_paths, metrics

    def _load_training_results(self) -> list[dict]:
        csv_path = self.settings.OUTPUT_DIR / "metrics.csv"
        if csv_path.exists():
            try:
                metric_df = pd.read_csv(csv_path)
                return metric_df.to_dict(orient="records")
            except Exception:
                pass
        return []

    # ── Step 6: Deploy if Improved ────────────────────────────────────

    def _deploy_if_improved(self, model_paths: list[Path],
                            metrics: dict) -> bool:
        current_best = self._load_current_best_metrics()
        new_r2 = metrics.get("r2", 0)

        if not current_best:
            self.log.info("No existing best model — deploying current")
            self._deploy_best(model_paths, metrics)
            return True

        prev_r2 = current_best.get("r2", 0)
        improvement = new_r2 - prev_r2
        self.log.info("Previous best R²: %.4f | New R²: %.4f | Δ: %+.4f",
                      prev_r2, new_r2, improvement)

        if improvement > 0.01:
            self.log.info("Improvement > 1%%, deploying new model")
            self._deploy_best(model_paths, metrics)
            return True
        elif improvement > 0:
            self.log.info("Minor improvement (< 1%%), retaining existing model")
        else:
            self.log.info("No improvement, retaining existing model")
        return False

    def _load_current_best_metrics(self) -> dict:
        path = self.settings.OUTPUT_DIR / "models" / f"{BEST_MODEL_LABEL}_metrics.json"
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
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
            self.log.info("Deployed model: %s → %s", best_path.name, deployed.name)

        metrics_path = models_dir / f"{BEST_MODEL_LABEL}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "r2": metrics.get("r2", 0),
                "rmse": metrics.get("rmse", 0),
                "mae": metrics.get("mae", 0),
                "deployed_at": datetime.now().isoformat(),
            }, f, indent=2)
        self.log.info("Saved metrics to %s", metrics_path.name)

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
        self.log.info("Model version logged to model_history.jsonl")

    # ── Step 7: Update Fuzzy Rules (if enabled) ──────────────────────

    def _update_fuzzy_rules(self, df: pd.DataFrame):
        self.log.info("Fuzzy rule update enabled — analyzing data drift")

        rules_path = Path(__file__).parent / "rules" / "fertilizer_rules.json"
        if not rules_path.exists():
            self.log.warning("Rules file not found at %s", rules_path)
            return

        with open(rules_path) as f:
            rules_data = json.load(f)

        drift_log = self._detect_drift(df)
        if drift_log:
            self._log_drift(drift_log)
            adjusted = self._adjust_rule_priorities(rules_data, drift_log)
            if adjusted:
                backup = rules_path.with_suffix(".json.bak")
                shutil.copy2(str(rules_path), str(backup))
                rules_data["metadata"]["version"] = (
                    f'{float(rules_data["metadata"]["version"]) + 0.1:.1f}'
                )
                rules_data["metadata"]["last_drift_update"] = datetime.now().isoformat()
                with open(rules_path, "w") as f:
                    json.dump(rules_data, f, indent=2)
                self.log.info("Fuzzy rules updated (version %s)", rules_data["metadata"]["version"])
                self.contract.artifacts.append(str(rules_path))
        else:
            self.log.info("No significant drift detected — fuzzy rules unchanged")

    def _detect_drift(self, df: pd.DataFrame) -> dict:
        from agri_ai_agent.rules.membership_functions import VAR_MEMBERSHIPS

        drift_log = {}
        for var, mfs in VAR_MEMBERSHIPS.items():
            if var not in df.columns:
                continue
            vals = df[var].dropna()
            if len(vals) < 10:
                continue

            for label, (shape, *params) in mfs.items():
                if shape == "shouldered_z":
                    threshold = params[0]
                    pct_below = (vals <= threshold).mean()
                    if pct_below > 0.8:
                        drift_log[f"{var}_{label}"] = {
                            "variable": var,
                            "set": label,
                            "threshold": threshold,
                            "pct_below": round(pct_below, 3),
                            "median": float(vals.median()),
                            "mean": float(vals.mean()),
                            "suggestion": "shift_threshold_right",
                        }
                elif shape == "shouldered_s":
                    threshold = params[0]
                    pct_above = (vals >= threshold).mean()
                    if pct_above > 0.8:
                        drift_log[f"{var}_{label}"] = {
                            "variable": var,
                            "set": label,
                            "threshold": threshold,
                            "pct_above": round(pct_above, 3),
                            "median": float(vals.median()),
                            "mean": float(vals.mean()),
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
        self.log.info("Drift analysis logged to %s (%d events)", path.name, len(drift_log))
        self.contract.artifacts.append(str(path))

    def _adjust_rule_priorities(self, rules_data: dict, drift_log: dict) -> bool:
        drifted_vars = {v["variable"] for v in drift_log.values()}
        if not drifted_vars:
            return False

        adjusted = 0
        for rule in rules_data.get("rules", []):
            rule_vars = {a["var"] for a in rule.get("antecedents", [])}
            overlap = rule_vars & drifted_vars
            if overlap:
                old_priority = rule.get("priority", 5)
                rule["priority"] = min(old_priority + 1, 10)
                if rule["priority"] != old_priority:
                    adjusted += 1

        if adjusted:
            self.log.info("Adjusted priorities for %d rules due to drift", adjusted)
        return adjusted > 0

    # ── Step 8: Generate New Ready Reckoner ───────────────────────────

    def _regenerate_reckoner(self, df: pd.DataFrame, metrics: dict):
        self.log.info("Regenerating full ready reckoner")

        from agri_ai_agent.agents.prediction_agent import PredictionAgent
        from agri_ai_agent.agents.fuzzy_logic_agent import FuzzyAgent
        from agri_ai_agent.agents.recommendation_agent import RecommendationAgent
        from agri_ai_agent.agents.ready_reckoner_agent import ReadyReckonerAgent

        try:
            pred_agent = PredictionAgent(settings=self.settings)
            pred_agent.run(df)
            df_pred = pred_agent.dataframe if pred_agent.dataframe is not None else df
        except Exception as e:
            self.log.warning("Prediction regeneration failed: %s", e)
            return

        try:
            fuzzy_agent = FuzzyAgent(settings=self.settings)
            fuzzy_agent.run(df_pred)
            df_fuzzy = fuzzy_agent.dataframe if fuzzy_agent.dataframe is not None else df_pred
        except Exception as e:
            self.log.warning("Fuzzy regeneration failed: %s", e)
            df_fuzzy = df_pred

        try:
            rec_agent = RecommendationAgent(settings=self.settings)
            rec_agent.run(df_fuzzy)
            df_rec = rec_agent.dataframe if rec_agent.dataframe is not None else df_fuzzy
        except Exception as e:
            self.log.warning("Recommendation regeneration failed: %s", e)
            df_rec = df_fuzzy

        try:
            reck_agent = ReadyReckonerAgent(settings=self.settings)
            reck_agent.run(df_rec)
            if reck_agent.contract and reck_agent.contract.artifacts:
                self.contract.artifacts.extend(reck_agent.contract.artifacts)
                self.log.info("Ready reckoner regenerated with %d artifact(s)",
                              len(reck_agent.contract.artifacts))
            else:
                self.log.warning("Ready reckoner produced no artifacts")
        except Exception as e:
            self.log.warning("Ready reckoner export failed: %s", e)

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "cycle_complete": True,
        }
