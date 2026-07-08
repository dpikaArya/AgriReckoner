"""
Fuzzy Logic Agent
The "agronomist" of the system. Converts model outputs into human-like
decision rules using Mamdani inference + symbolic rule evaluation.

Produces per-nutrient recommendations (N, P, K, Zinc), risk assessment,
and natural-language summaries instead of raw numerical predictions.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.rules.membership_functions import fuzzify
from agri_ai_agent.rules.output_memberships import OUTPUT_MEMBERSHIPS

RULES_DIR = Path(__file__).parent.parent / "rules"
DEFAULT_RULES_PATH = RULES_DIR / "fertilizer_rules.yaml"

N_SAMPLES = 1000

INPUT_VARS = [
    "Nitrogen", "Phosphorus", "Potassium", "Soil_pH",
    "Rainfall", "Temperature_Max", "Organic_Carbon",
    "Growth_Stage", "Zinc", "Yield_Prediction",
]

NUTRIENT_ACTION_LABELS = {
    "increase": "Increase",
    "reduce": "Reduce",
    "maintain": "Maintain",
    "apply_foliar_spray": "Apply foliar spray",
    "apply_soil": "Apply to soil",
    "monitor": "Monitor",
    "review": "Review program",
}

ADJUSTMENT_MAP = {
    "Nitrogen_Adjustment": {"col": "Recommended_Dose", "scale": 0.5},
    "Potash_Adjustment": {"col": "Recommended_Dose", "scale": 0.3},
    "Irrigation_Adjustment": {"col": "Recommended_Application_Interval", "scale": 1.0},
    "Dose_Adjustment": {"col": "Recommended_Dose", "scale": 1.0},
}

ADJUST_ACTIONS = ["Nitrogen_Adjustment", "Potash_Adjustment",
                  "Irrigation_Adjustment", "Dose_Adjustment"]


class FuzzyAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "FuzzyAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        rules_path = Path(kwargs.get("rules_path", DEFAULT_RULES_PATH))
        rules = self._load_rules(rules_path)
        if not rules:
            self.log.warning("No rules loaded from %s", rules_path)
            return df

        self.log.info("Loaded %d fuzzy rules", len(rules))

        # Add Yield_Prediction if ML predictions exist
        df = self._infer_yield_prediction(df)

        out_cols = [
            "Fuzzy_N_Action", "Fuzzy_P_Action", "Fuzzy_K_Action",
            "Fuzzy_Zinc_Action", "Fuzzy_Risk", "Fuzzy_Confidence_Label",
            "Fuzzy_Summary",
        ]
        for c in out_cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype=object)

        for row_idx in df.index:
            row = df.loc[row_idx]

            fuzzy_inputs = self._fuzzify_row(row)
            if not fuzzy_inputs:
                continue

            fired_rules = self._evaluate(rules, fuzzy_inputs)
            if not fired_rules:
                continue

            numerical = self._defuzzify(fired_rules)
            self._apply_numerical(df, row_idx, numerical, row)

            symbolic = self._evaluate_symbolic(rules, fuzzy_inputs)
            self._apply_symbolic(df, row_idx, symbolic, row)

        n_applied = df["Fuzzy_Summary"].notna().sum()
        self.log.info("Fuzzy agronomist recommendations generated for %d rows", n_applied)

        self.dataframe = df
        return df

    def _infer_yield_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Yield_Prediction" in df.columns:
            return df

        pred_cols = [c for c in ["Predicted_Yield", "XGBoost_Prediction",
                                  "Regression_Prediction"]
                     if c in df.columns]
        if pred_cols:
            df["Yield_Prediction"] = df[pred_cols].mean(axis=1)
        else:
            df["Yield_Prediction"] = np.nan
        return df

    def _load_rules(self, path: Path) -> list[dict]:
        if not path.exists():
            self.log.warning("Rules file not found: %s", path)
            return []
        with open(path) as f:
            data = yaml.safe_load(f)
        return sorted(data.get("rules", []), key=lambda r: r.get("priority", 0), reverse=True)

    # ── Fuzzify ──────────────────────────────────────────────────────

    def _fuzzify_row(self, row: pd.Series) -> dict[str, dict[str, float]]:
        result = {}
        for var in INPUT_VARS:
            val = row.get(var)
            if pd.notna(val):
                mfs = fuzzify(var, float(val))
                if mfs:
                    result[var] = mfs
        return result

    # ── Numerical engine (Mamdani → centroid) ────────────────────────

    def _evaluate(self, rules: list[dict],
                  fuzzy_inputs: dict) -> dict[str, dict[str, float]]:
        fired = {}
        for rule in rules:
            ant = rule.get("antecedents", [])
            strengths = []
            for cond in ant:
                var = cond["var"]
                fset = cond["set"]
                membership = fuzzy_inputs.get(var, {}).get(fset, 0.0)
                strengths.append(membership)
            if not strengths:
                continue
            firing = min(strengths)
            if firing <= 0:
                continue
            for cons in rule.get("consequents", []):
                if cons.get("type") != "adjustment":
                    continue
                var = cons["var"]
                fset = cons["set"]
                if var not in fired:
                    fired[var] = {}
                prev = fired[var].get(fset, 0.0)
                fired[var][fset] = max(prev, firing)
        return fired

    def _defuzzify(self, aggregated: dict) -> dict[str, float]:
        from agri_ai_agent.rules.membership_functions import (
            triangle, trapezoid, shouldered_s, shouldered_z,
        )

        crisp = {}
        x_vals = np.linspace(0, 1, N_SAMPLES)

        for var in ADJUST_ACTIONS:
            if var not in aggregated:
                crisp[var] = 0.5
                continue

            mfs = OUTPUT_MEMBERSHIPS.get(var, {})
            aggregated_mf = np.zeros(N_SAMPLES)

            for fset, firing in aggregated[var].items():
                if fset not in mfs:
                    continue
                shape, *params = mfs[fset]
                if shape == "triangle":
                    a, b, c = params
                    mf_vals = triangle(x_vals, a, b, c)
                elif shape == "trapezoid":
                    a, b, c, d = params
                    mf_vals = trapezoid(x_vals, a, b, c, d)
                elif shape == "shouldered_s":
                    a, b = params
                    mf_vals = shouldered_s(x_vals, a, b)
                elif shape == "shouldered_z":
                    a, b = params
                    mf_vals = shouldered_z(x_vals, a, b)
                else:
                    continue
                aggregated_mf = np.maximum(aggregated_mf, np.minimum(mf_vals, firing))

            if aggregated_mf.sum() > 0:
                crisp[var] = np.sum(x_vals * aggregated_mf) / aggregated_mf.sum()
            else:
                crisp[var] = 0.5

        return crisp

    def _apply_numerical(self, df: pd.DataFrame, row_idx: int,
                         crisp: dict[str, float], row: pd.Series):
        n_adj = crisp.get("Nitrogen_Adjustment", 0.5)
        k_adj = crisp.get("Potash_Adjustment", 0.5)
        irr_adj = crisp.get("Irrigation_Adjustment", 0.5)
        dose_adj = crisp.get("Dose_Adjustment", 0.5)

        dose = row.get("Recommended_Dose") or row.get("Dose")
        if pd.notna(dose) and isinstance(dose, (int, float)):
            combined = n_adj * 0.4 + k_adj * 0.2 + dose_adj * 0.4
            df.at[row_idx, "Recommended_Dose"] = dose * (0.5 + combined)

        interval = (row.get("Recommended_Application_Interval")
                    or row.get("Application_Interval"))
        if pd.notna(interval) and isinstance(interval, (int, float)):
            irr_factor = 2.0 - irr_adj
            df.at[row_idx, "Recommended_Application_Interval"] = interval * irr_factor
        else:
            df.at[row_idx, "Recommended_Application_Interval"] = 7 + (1 - irr_adj) * 14

        if n_adj > 0.6:
            df.at[row_idx, "Recommended_Fertilizer"] = "Urea (Fuzzy)"

        confidence = n_adj * 0.3 + k_adj * 0.2 + irr_adj * 0.2 + dose_adj * 0.3
        df.at[row_idx, "Confidence_Score"] = round(confidence, 3)

    # ── Symbolic engine (per-nutrient actions, risk, confidence) ─────

    def _evaluate_symbolic(self, rules: list[dict],
                           fuzzy_inputs: dict) -> dict:
        nutrient_actions: dict[str, list[tuple[str, float, float]]] = {}
        risk_scores: list[tuple[str, float]] = []
        confidence_scores: list[tuple[str, float]] = []

        for rule in rules:
            ant = rule.get("antecedents", [])
            strengths = []
            for cond in ant:
                membership = fuzzy_inputs.get(cond["var"], {}).get(cond["set"], 0.0)
                strengths.append(membership)
            if not strengths:
                continue
            firing = min(strengths)
            if firing <= 0:
                continue

            for cons in rule.get("consequents", []):
                ctype = cons.get("type")

                if ctype == "nutrient":
                    n_name = cons["nutrient"]
                    if n_name not in nutrient_actions:
                        nutrient_actions[n_name] = []
                    nutrient_actions[n_name].append(
                        (cons["action"], cons.get("amount", 0), firing)
                    )
                elif ctype == "risk":
                    risk_scores.append((cons["level"], firing))
                elif ctype == "confidence":
                    confidence_scores.append((cons["level"], firing))

        return {
            "nutrient_actions": nutrient_actions,
            "risk_scores": risk_scores,
            "confidence_scores": confidence_scores,
        }

    NUTRIENT_COLUMNS = {
        "Nitrogen": "Fuzzy_N_Action",
        "Phosphorus": "Fuzzy_P_Action",
        "Potassium": "Fuzzy_K_Action",
        "Zinc": "Fuzzy_Zinc_Action",
    }

    def _apply_symbolic(self, df: pd.DataFrame, row_idx: int,
                        symbolic: dict, row: pd.Series):
        nutrient_actions = symbolic.get("nutrient_actions", {})
        risk_scores = symbolic.get("risk_scores", [])
        confidence_scores = symbolic.get("confidence_scores", [])

        for n_name, col in self.NUTRIENT_COLUMNS.items():
            actions = nutrient_actions.get(n_name, [])
            best = self._pick_best_action(actions)
            df.at[row_idx, col] = best

        df.at[row_idx, "Fuzzy_Risk"] = (
            self._pick_best_label(risk_scores, ["low", "medium", "high"])
            if risk_scores else "medium"
        )
        df.at[row_idx, "Fuzzy_Confidence_Label"] = (
            self._pick_best_label(confidence_scores, ["low", "medium", "high"])
            if confidence_scores else "medium"
        )
        df.at[row_idx, "Fuzzy_Summary"] = self._build_summary(row_idx, df)

    def _pick_best_action(self, actions: list[tuple[str, float, float]]) -> str:
        if not actions:
            return "Maintain"
        by_action: dict[str, list[float]] = {}
        for action, amount, firing in actions:
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(firing)
        action_label = max(by_action, key=lambda a: max(by_action[a]))
        max_firing = max(by_action[action_label])
        label = NUTRIENT_ACTION_LABELS.get(action_label, action_label.replace("_", " ").title())
        amount = 0
        for act, amt, firing in actions:
            if act == action_label and firing == max_firing:
                amount = amt
                break
        if amount > 0 and action_label in ("increase", "reduce"):
            return f"{label} by {amount:.0f}%"
        return label

    def _pick_best_label(self, scored: list[tuple[str, float]],
                         order: list[str]) -> str:
        by_label: dict[str, float] = {}
        for label, firing in scored:
            if label not in by_label or firing > by_label[label]:
                by_label[label] = firing
        if not by_label:
            return order[len(order) // 2]
        sorted_labels = sorted(by_label.items(), key=lambda x: x[1], reverse=True)
        return sorted_labels[0][0]

    def _build_summary(self, row_idx: int, df: pd.DataFrame) -> str:
        parts = []

        for nutrient, col in self.NUTRIENT_COLUMNS.items():
            action = df.at[row_idx, col]
            if pd.notna(action) and action != "Maintain":
                parts.append(f"{nutrient}: {action.lower()}")

        if not parts:
            parts.append("all nutrients optimal — maintain current schedule")

        risk = df.at[row_idx, "Fuzzy_Risk"]
        if pd.notna(risk):
            parts.append(f"risk: {risk.lower()}")

        conf = df.at[row_idx, "Fuzzy_Confidence_Label"]
        if pd.notna(conf):
            parts.append(f"confidence: {conf.lower()}")

        return " | ".join(parts)

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "fuzzy_applied": True,
        }
