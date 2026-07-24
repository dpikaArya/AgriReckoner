"""
Prediction Agent
Flow: Current Conditions → Predict Yield → Simulate Fertilizer Options
      → Compare Outcomes → Recommend Best Treatment
      → Apply Expert Rules → Final Recommendation
"""

import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.schema import UAMS_COLUMNS
from agri_ai_agent.contracts.messages import PredictionResult

INPUT_FEATURE_GROUPS = {
    "Current Soil": [
        "Soil_pH", "EC", "Organic_Carbon", "Organic_Matter",
        "Nitrogen", "Phosphorus", "Potassium", "Sulphur",
        "Iron", "Copper", "Manganese", "Zinc",
        "Calcium", "Magnesium", "Boron", "Molybdenum",
    ],
    "Crop": [
        "Crop", "Scientific_Name", "Variety", "Season", "Growth_Duration_Days",
    ],
    "Growth Stage": [
        "Growth_Stage", "Growth_Duration_Days",
    ],
    "Weather": [
        "Temperature_Max", "Temperature_Min", "Average_Temperature",
        "Rainfall", "Humidity", "Latitude", "Longitude", "Altitude",
    ],
    "Present Fertilizer": [
        "Fertilizer_Name", "Organic_Fertilizer", "Biofertilizer",
        "Dose", "Application_Method", "Application_Interval",
    ],
    "Target Yield": [
        "Target_Yield",
    ],
}

FERTILIZER_FEATURES = [
    "Dose", "Application_Interval", "Fertilizer_Code",
    "Fertilizer_Name", "Organic_Fertilizer", "Biofertilizer",
]

DOSE_FACTORS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]

PREDICTION_COLUMNS = [
    "Predicted_Yield", "Expected_Biomass", "Expected_Plant_Height",
    "Recommended_Fertilizer", "Recommended_Dose",
    "Recommended_Application_Interval", "Expected_Yield_Increase",
    "Recommendation_Summary",
]

# ── Expert Rule Engine ───────────────────────────────────────────────

NITROGEN_THRESHOLDS = {"LOW": 50, "MEDIUM": 100, "HIGH": float("inf")}
PHOSPHORUS_THRESHOLDS = {"LOW": 15, "MEDIUM": 30, "HIGH": float("inf")}
POTASSIUM_THRESHOLDS = {"LOW": 100, "MEDIUM": 200, "HIGH": float("inf")}
RAINFALL_THRESHOLDS = {"LOW": 500, "MEDIUM": 1000, "HIGH": float("inf")}
SOIL_PH_OK = (5.5, 7.5)

SOIL_N_FERTILIZERS = ["Urea", "DAP", "Ammonium Sulphate", "Calcium Ammonium Nitrate", "UAN"]
POTASH_FERTILIZERS = ["MOP", "SOP", "Potassium Sulphate", "Potassium Magnesium Sulphate"]

EXPERT_RULES = [
    {
        "name": "LowN_HighRain_GoodpH",
        "conditions": {"Nitrogen": "LOW", "Rainfall": "HIGH", "Soil_pH": "GOOD"},
        "actions": {"nitrogen": "increase", "potash": "moderate", "irrigation": "reduce"},
        "priority": 10,
    },
    {
        "name": "LowN_LowRain",
        "conditions": {"Nitrogen": "LOW", "Rainfall": "LOW"},
        "actions": {"nitrogen": "increase", "irrigation": "increase"},
        "priority": 8,
    },
    {
        "name": "HighN_GoodRain",
        "conditions": {"Nitrogen": "HIGH", "Rainfall": "MEDIUM"},
        "actions": {"nitrogen": "maintain", "irrigation": "maintain"},
        "priority": 5,
    },
]


class PredictionAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "PredictionAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        models_dir = self.settings.OUTPUT_DIR / "models"

        self._report_input_groups(df)

        model_registry = self._load_models(models_dir)
        if not model_registry:
            return self._add_empty_predictions(df)

        X = self._prepare_features(df, models_dir)
        if X is None:
            return self._add_empty_predictions(df)

        # Step 1: Predict baseline yield from current conditions
        yield_model = self._resolve_yield_model(model_registry)
        if yield_model is None:
            self.log.warning("No suitable yield model found")
            return self._add_empty_predictions(df)

        self.log.info("Predicting baseline yield from current conditions")
        baseline_preds = self._ensemble_predict(yield_model, X, model_registry)
        df["Predicted_Yield"] = baseline_preds

        for col, target in [("Expected_Biomass", "Biomass_Yield"),
                            ("Expected_Plant_Height", "Plant_Height_cm")]:
            preds = self._predict_target(target, X, model_registry)
            if preds is not None:
                df[col] = preds

        # Step 2: Build fertilizer option grid
        self.log.info("Generating fertilizer simulation options")
        options = self._build_fertilizer_options(df, X, kwargs.get("fertilizer_options"))

        # Steps 3-4: Simulate each option and compare
        comparisons = self._simulate_options(X, yield_model, model_registry, options, baseline_preds)

        # Save comparison tables as artifacts
        self._save_comparison_tables(df, X, options, comparisons)

        # Step 5: Recommend best treatment per row
        df = self._recommend_best(df, comparisons)
        df = self._compute_confidence(df, model_registry, X)

        # Step 6: Apply expert rules to refine recommendations
        df = self._apply_expert_rules(df)

        # Step 7: Generate human-readable summary
        df = self._generate_summary(df)

        n_rec = df["Recommended_Fertilizer"].notna().sum()
        self.log.info("Recommendations generated for %d/%d rows", n_rec, len(df))

        self.dataframe = df
        return df

    # ── Input Validation ─────────────────────────────────────────────

    def _report_input_groups(self, df: pd.DataFrame):
        present = []
        missing = []
        for group, cols in INPUT_FEATURE_GROUPS.items():
            found = [c for c in cols if c in df.columns]
            if found:
                present.append(f"{group} ({len(found)}/{len(cols)})")
            else:
                missing.append(group)
        if present:
            self.log.info("Input groups detected: %s", ", ".join(present))
        if missing:
            self.log.warning("Input groups missing: %s", ", ".join(missing))

    # ── Step 1: Predict Yield ────────────────────────────────────────

    def _resolve_yield_model(self, model_registry: list) -> Optional[str]:
        for name, _ in model_registry:
            if "target_yield" in name.lower():
                return name
        for name, _ in model_registry:
            if "yield" in name.lower():
                return name
        return model_registry[0][0] if model_registry else None

    def _predict_target(self, target_col: str, X: pd.DataFrame,
                        model_registry: list) -> Optional[np.ndarray]:
        target_models = [
            (name, m) for name, m in model_registry
            if target_col.lower().replace(" ", "_") in name.lower()
        ]
        if not target_models:
            target_models = model_registry
        all_preds = []
        for _, m in target_models:
            try:
                all_preds.append(m.predict(X))
            except Exception:
                continue
        if not all_preds:
            return None
        return np.mean(all_preds, axis=0)

    def _ensemble_predict(self, primary_name: str, X: pd.DataFrame,
                          model_registry: list) -> np.ndarray:
        all_preds = []
        for name, model in model_registry:
            if name == primary_name:
                try:
                    all_preds.append(model.predict(X))
                except Exception:
                    continue
        if all_preds:
            return np.mean(all_preds, axis=0)
        for _, model in model_registry:
            try:
                all_preds.append(model.predict(X))
            except Exception:
                continue
        return np.mean(all_preds, axis=0) if all_preds else np.zeros(len(X))

    # ── Step 2: Build Fertilizer Options ─────────────────────────────

    def _build_fertilizer_options(self, df: pd.DataFrame, X: pd.DataFrame,
                                  explicit_options: Optional[list] = None) -> list[dict]:
        if explicit_options:
            self.log.info("Using %d explicit fertilizer options", len(explicit_options))
            return explicit_options

        options = []

        baseline = self._get_fertilizer_baseline(df) or {}
        if not baseline:
            self.log.warning("No fertilizer data found; emitting a single baseline option only")

        dose_col = "Dose" if "Dose" in df.columns else None
        interval_col = "Application_Interval" if "Application_Interval" in df.columns else None
        fert_code_col = "Fertilizer_Code" if "Fertilizer_Code" in df.columns else None
        fert_name_col = "Fertilizer_Name" if "Fertilizer_Name" in df.columns else None

        base_dose = baseline.get(dose_col)
        base_interval = baseline.get(interval_col)
        base_code = baseline.get(fert_code_col)
        base_name = baseline.get(fert_name_col)

        # Always include baseline as an option
        options.append(self._make_option("Current", base_name, base_code, base_dose, base_interval))

        # Dose sweep
        if base_dose is not None and np.issubdtype(type(base_dose), np.number):
            for factor in DOSE_FACTORS:
                dose_val = base_dose * factor
                label = f"Dose_{factor:.0%}"
                options.append(self._make_option(label, base_name, base_code, dose_val, base_interval))

        # Try alternative fertilizers from dataset
        if fert_name_col and df[fert_name_col].nunique() > 1:
            for alt_name in df[fert_name_col].dropna().unique():
                if alt_name == base_name:
                    continue
                alt_row = df[df[fert_name_col] == alt_name].iloc[0]
                alt_code = alt_row.get(fert_code_col) if fert_code_col else None
                alt_dose = alt_row.get(dose_col) if dose_col else base_dose
                alt_interval = alt_row.get(interval_col) if interval_col else base_interval
                options.append(self._make_option(alt_name, alt_name, alt_code, alt_dose, alt_interval))

        self.log.info("Generated %d fertilizer simulation options", len(options))
        return options

    def _get_fertilizer_baseline(self, df: pd.DataFrame) -> Optional[dict]:
        baseline = {}
        for col in FERTILIZER_FEATURES:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    baseline[col] = vals.iloc[0] if col == "Fertilizer_Name" else (
                        vals.mode().iloc[0] if col in ("Organic_Fertilizer", "Biofertilizer",
                                                       "Application_Method")
                        else vals.median() if np.issubdtype(vals.dtype, np.number)
                        else vals.iloc[0]
                    )
        return baseline if baseline else None

    def _make_option(self, label: str, fert_name, fert_code, dose, interval) -> dict:
        return {
            "label": label,
            "Fertilizer_Name": fert_name,
            "Fertilizer_Code": fert_code,
            "Dose": dose,
            "Application_Interval": interval,
        }

    # ── Steps 3-4: Simulate & Compare ────────────────────────────────

    def _simulate_options(self, X: pd.DataFrame, yield_model_name: str,
                          model_registry: list, options: list[dict],
                          baseline: np.ndarray) -> list[list[dict]]:
        dose_col = "Dose" if "Dose" in X.columns else None
        fert_code_col = "Fertilizer_Code" if "Fertilizer_Code" in X.columns else None

        all_comparisons = []
        for row_idx in range(len(X)):
            row_comparisons = []
            for opt in options:
                X_mod = X.iloc[[row_idx]].copy()
                if dose_col is not None and opt["Dose"] is not None:
                    X_mod[dose_col] = opt["Dose"]
                if fert_code_col is not None and opt["Fertilizer_Code"] is not None:
                    X_mod[fert_code_col] = opt["Fertilizer_Code"]

                pred = self._ensemble_predict(yield_model_name, X_mod, model_registry)
                yield_val = float(pred[0])

                row_comparisons.append({
                    "label": opt["label"],
                    "Fertilizer_Name": opt["Fertilizer_Name"],
                    "Dose": opt["Dose"],
                    "Application_Interval": opt["Application_Interval"],
                    "Predicted_Yield": yield_val,
                    "Yield_Increase": yield_val - float(baseline[row_idx]),
                })
            row_comparisons.sort(key=lambda r: r["Predicted_Yield"], reverse=True)
            all_comparisons.append(row_comparisons)

        best_overall = all_comparisons[0][0] if all_comparisons and all_comparisons[0] else {}
        self.log.info("Top option: %s → yield=%.2f (Δ%+.2f)",
                      best_overall.get("label", "?"),
                      best_overall.get("Predicted_Yield", 0),
                      best_overall.get("Yield_Increase", 0))

        return all_comparisons

    # ── Step 5: Recommend Best ───────────────────────────────────────

    def _recommend_best(self, df: pd.DataFrame,
                        comparisons: list[list[dict]]) -> pd.DataFrame:
        for row_idx, row_comparisons in enumerate(comparisons):
            if not row_comparisons:
                continue
            best = row_comparisons[0]
            df.at[df.index[row_idx], "Recommended_Fertilizer"] = best.get("Fertilizer_Name")
            df.at[df.index[row_idx], "Recommended_Dose"] = best.get("Dose")
            df.at[df.index[row_idx], "Recommended_Application_Interval"] = best.get("Application_Interval")
            df.at[df.index[row_idx], "Expected_Yield_Increase"] = best.get("Yield_Increase")
        return df

    def _compute_confidence(self, df: pd.DataFrame, model_registry: list,
                            X: pd.DataFrame) -> pd.DataFrame:
        if len(model_registry) < 2:
            df["Confidence_Score"] = 0.5
            return df
        all_preds = []
        for _, m in model_registry:
            try:
                all_preds.append(m.predict(X))
            except Exception:
                continue
        if len(all_preds) < 2:
            df["Confidence_Score"] = 0.5
            return df
        stacked = np.column_stack(all_preds)
        mean_pred = np.mean(stacked, axis=1)
        std_pred = np.std(stacked, axis=1) + 1e-10
        cv = std_pred / np.abs(mean_pred)
        df["Confidence_Score"] = np.clip(1.0 - cv, 0.0, 1.0)
        self.log.info("Confidence scores: mean=%.3f, min=%.3f, max=%.3f",
                      df["Confidence_Score"].mean(),
                      df["Confidence_Score"].min(),
                      df["Confidence_Score"].max())
        return df

    # ── Step 6: Expert Rules ──────────────────────────────────────────

    def _apply_expert_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        rules_fired = 0
        for row_idx in df.index:
            row = df.loc[row_idx]
            state = self._categorize_row(row)
            matched = self._evaluate_rules(state)
            if matched:
                self._apply_actions(df, row_idx, matched["actions"], row)
                rules_fired += 1
        if rules_fired:
            self.log.info("Expert rules applied to %d rows", rules_fired)
        return df

    def _categorize_row(self, row: pd.Series) -> dict[str, str]:
        state = {}
        if pd.notna(row.get("Nitrogen")):
            state["Nitrogen"] = self._quantize(row["Nitrogen"], NITROGEN_THRESHOLDS)
        if pd.notna(row.get("Phosphorus")):
            state["Phosphorus"] = self._quantize(row["Phosphorus"], PHOSPHORUS_THRESHOLDS)
        if pd.notna(row.get("Potassium")):
            state["Potassium"] = self._quantize(row["Potassium"], POTASSIUM_THRESHOLDS)
        if pd.notna(row.get("Rainfall")):
            state["Rainfall"] = self._quantize(row["Rainfall"], RAINFALL_THRESHOLDS)
        if pd.notna(row.get("Soil_pH")):
            state["Soil_pH"] = "GOOD" if SOIL_PH_OK[0] <= row["Soil_pH"] <= SOIL_PH_OK[1] else \
                ("LOW" if row["Soil_pH"] < SOIL_PH_OK[0] else "HIGH")
        return state

    def _quantize(self, value: float, thresholds: dict[str, float]) -> str:
        for label, upper in thresholds.items():
            if value <= upper:
                return label
        return list(thresholds.keys())[-1]

    def _evaluate_rules(self, state: dict[str, str]) -> Optional[dict]:
        matched = None
        for rule in sorted(EXPERT_RULES, key=lambda r: r["priority"], reverse=True):
            if all(state.get(var) == val for var, val in rule["conditions"].items()):
                matched = rule
                break
        return matched

    def _apply_actions(self, df: pd.DataFrame, row_idx: int,
                       actions: dict, row: pd.Series):
        n_act = actions.get("nitrogen", "")
        k_act = actions.get("potash", "")
        irr_act = actions.get("irrigation", "")

        if n_act == "increase":
            if pd.notna(row.get("Recommended_Fertilizer")):
                current = str(row["Recommended_Fertilizer"]).lower()
                n_fert = [f for f in SOIL_N_FERTILIZERS if f.lower() not in current]
                if n_fert:
                    df.at[row_idx, "Recommended_Fertilizer"] = n_fert[0]
            dose = row.get("Recommended_Dose") or row.get("Dose")
            if pd.notna(dose):
                df.at[row_idx, "Recommended_Dose"] = dose * 1.25

        elif n_act == "maintain":
            pass

        if k_act == "moderate":
            k_fert = [f for f in POTASH_FERTILIZERS]
            if k_fert:
                df.at[row_idx, "Recommended_Fertilizer"] = k_fert[0]
            dose = row.get("Recommended_Dose") or row.get("Dose")
            if pd.notna(dose):
                df.at[row_idx, "Recommended_Dose"] = dose * 0.5

        if irr_act == "reduce":
            interval = row.get("Recommended_Application_Interval") or row.get("Application_Interval")
            if pd.notna(interval) and isinstance(interval, (int, float)):
                df.at[row_idx, "Recommended_Application_Interval"] = interval * 1.5
            elif pd.isna(interval):
                df.at[row_idx, "Recommended_Application_Interval"] = 14
        elif irr_act == "increase":
            interval = row.get("Recommended_Application_Interval") or row.get("Application_Interval")
            if pd.notna(interval) and isinstance(interval, (int, float)):
                df.at[row_idx, "Recommended_Application_Interval"] = interval * 0.75

    # ── Comparison Tables ─────────────────────────────────────────────

    def _save_comparison_tables(self, df: pd.DataFrame, X: pd.DataFrame,
                                options: list[dict],
                                comparisons: list[list[dict]]):
        if not comparisons or not comparisons[0]:
            return

        rows = []
        for row_idx, row_comparisons in enumerate(comparisons):
            row_conditions = {}
            for col in ["Soil_pH", "Nitrogen", "Phosphorus", "Potassium",
                        "Rainfall", "Temperature_Max", "Crop", "Growth_Stage"]:
                val = df.iloc[row_idx].get(col) if col in df.columns else X.iloc[row_idx].get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    row_conditions[col] = val
            for opt in row_comparisons:
                rows.append({**row_conditions, **opt})

        if not rows:
            return

        tbl = pd.DataFrame(rows)

        display_cols = [c for c in ["Soil_pH", "Nitrogen", "Phosphorus",
                                     "Potassium", "Rainfall", "Temperature_Max",
                                     "Crop", "Growth_Stage"]
                        if c in tbl.columns]
        result_cols = [c for c in ["label", "Fertilizer_Name", "Dose",
                                    "Application_Interval", "Predicted_Yield",
                                    "Yield_Increase"]
                       if c in tbl.columns]
        table_cols = display_cols + result_cols
        tbl = tbl[[c for c in table_cols if c in tbl.columns]]

        csv_path = self.save_artifact(tbl, "recommendation_comparison.csv", subdir="recommendations")
        self.log.info("Saved comparison table: %s (%d rows)", csv_path.name, len(tbl))

        html = self._render_comparison_html(tbl)
        html_path = self.save_text_artifact(html, "recommendation_comparison.html", subdir="recommendations")
        self.log.info("Saved comparison HTML: %s", html_path.name)

    def _render_comparison_html(self, tbl: pd.DataFrame) -> str:
        from datetime import datetime
        cols = list(tbl.columns)
        thead = "".join(f"<th>{c}</th>" for c in cols)
        tbody = ""
        for _, row in tbl.iterrows():
            tbody += "<tr>" + "".join(
                f"<td>{v:.2f}</td>" if isinstance(v, float)
                else f"<td>{v}</td>"
                for v in row
            ) + "</tr>"
        now = datetime.now().isoformat()
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fertilizer Recommendation Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; }}
h1 {{ color: #2c3e50; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 14px; }}
th, td {{ border: 1px solid #bbb; padding: 8px; text-align: left; }}
th {{ background-color: #27ae60; color: white; font-weight: 600; }}
tr:nth-child(even) {{ background-color: #f9f9f9; }}
tr:hover {{ background-color: #e8f8f0; }}
td:last-child, th:last-child {{ font-weight: 600; }}
.footer {{ margin-top: 1em; color: #7f8c8d; font-size: 12px; }}
.highlight {{ background-color: #d4efdf !important; }}
</style>
</head>
<body>
<h1>Fertilizer Recommendation Comparison</h1>
<p>Generated: {now}</p>
<table>
<thead><tr>{thead}</tr></thead>
<tbody>{tbody}</tbody>
</table>
<div class="footer">Best option highlighted in green &mdash; sorted by Predicted Yield descending</div>
</body>
</html>"""

    # ── Step 7: Human-Readable Summary ────────────────────────────────

    SUMMARY_TRIGGERS = {
        "Nitrogen": ("Nitrogen", "ppm", "low", "high"),
        "Phosphorus": ("Phosphorus", "ppm", "low", "high"),
        "Potassium": ("Potassium", "ppm", "low", "high"),
        "Rainfall": ("Rainfall", "mm", "low", "high"),
        "Soil_pH": ("Soil pH", "", "acidic/alkaline", "optimal"),
        "Temperature_Max": ("Temperature", "°C", "cool", "hot"),
    }

    def _generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        for row_idx in df.index:
            row = df.loc[row_idx]
            parts = []

            conditions = self._describe_conditions(row)
            if conditions:
                parts.append("Current: " + "; ".join(conditions) + ".")

            rec_fert = row.get("Recommended_Fertilizer")
            rec_dose = row.get("Recommended_Dose")
            rec_interval = row.get("Recommended_Application_Interval")
            increase = row.get("Expected_Yield_Increase")
            confidence = row.get("Confidence_Score")

            rec_parts = []
            if pd.notna(rec_fert):
                rec_parts.append(str(rec_fert))
            if pd.notna(rec_dose):
                rec_parts.append(f"{rec_dose:.1f} units")
            if pd.notna(rec_interval):
                rec_parts.append(f"every {rec_interval:.0f} days")
            if rec_parts:
                parts.append(f"Apply {' '.join(rec_parts)}.")

            if pd.notna(increase):
                symbol = "+" if increase >= 0 else ""
                parts.append(f"Expected yield {symbol}{increase:.1f} vs baseline.")
            if pd.notna(confidence):
                pct = confidence * 100
                if pct >= 70:
                    parts.append(f"High confidence ({pct:.0f}%).")
                elif pct >= 40:
                    parts.append(f"Moderate confidence ({pct:.0f}%).")
                else:
                    parts.append(f"Low confidence ({pct:.0f}%).")

            df.at[row_idx, "Recommendation_Summary"] = " ".join(parts) if parts else ""

        return df

    def _describe_conditions(self, row: pd.Series) -> list[str]:
        desc = []
        for col, (label, unit, if_low, if_high) in self.SUMMARY_TRIGGERS.items():
            val = row.get(col)
            if pd.isna(val):
                continue
            if col == "Soil_pH":
                if val < 5.5:
                    desc.append(f"{label} {val:.1f} (acidic)")
                elif val > 7.5:
                    desc.append(f"{label} {val:.1f} (alkaline)")
                else:
                    desc.append(f"{label} {val:.1f} (optimal)")
            elif col == "Temperature_Max":
                if val < 20:
                    desc.append(f"{label} {val:.0f}{unit} (cool)")
                elif val > 35:
                    desc.append(f"{label} {val:.0f}{unit} (hot)")
                else:
                    desc.append(f"{label} {val:.0f}{unit} (moderate)")
            else:
                thresh_low = NITROGEN_THRESHOLDS.get("LOW", 0) if col == "Nitrogen" else (
                    PHOSPHORUS_THRESHOLDS.get("LOW", 0) if col == "Phosphorus" else (
                    POTASSIUM_THRESHOLDS.get("LOW", 0) if col == "Potassium" else (
                    RAINFALL_THRESHOLDS.get("LOW", 0))))
                if val < thresh_low:
                    desc.append(f"{label} {val:.0f}{unit} ({if_low})")
                else:
                    desc.append(f"{label} {val:.0f}{unit}")
        return desc

    # ── Model Loading & Feature Prep ─────────────────────────────────

    def _load_models(self, models_dir: Path) -> list[tuple[str, object]]:
        if not models_dir.exists():
            self.log.warning("Models directory not found: %s", models_dir)
            return []
        import joblib
        registry = []
        for path in sorted(models_dir.glob("*.joblib")):
            try:
                model = joblib.load(path)
                registry.append((path.stem, model))
                self.log.info("Loaded model: %s", path.name)
            except Exception as e:
                self.log.warning("Failed to load %s: %s", path.name, e)
        pkl = models_dir / "yield_model.pkl"
        if pkl.exists():
            try:
                model = joblib.load(pkl)
                registry.append(("yield_model", model))
                self.log.info("Loaded model: yield_model.pkl")
            except Exception as e:
                self.log.warning("Failed to load yield_model.pkl: %s", e)
        return registry

    def _prepare_features(self, df: pd.DataFrame, models_dir: Path) -> Optional[pd.DataFrame]:
        import json
        path = models_dir / "feature_list.json"
        if not path.exists():
            self.log.warning("feature_list.json not found")
            return None
        with open(path) as f:
            feature_list = json.load(f)
        available = [c for c in feature_list if c in df.columns]
        if not available:
            self.log.warning("No features from feature_list.json found in data")
            return None
        missing = set(feature_list) - set(available)
        if missing:
            self.log.warning("Missing %d/%d features", len(missing), len(feature_list))
        X = df[available].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        return X

    # ── Output Helpers ───────────────────────────────────────────────

    def _add_empty_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in PREDICTION_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        if "Confidence_Score" not in df.columns:
            df["Confidence_Score"] = np.nan
        return df

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "predicted_columns": [c for c in PREDICTION_COLUMNS if c in df.columns],
            "rows_with_recommendations": int(df["Recommended_Fertilizer"].notna().sum()),
        }
