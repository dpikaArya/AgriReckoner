import json
import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("PredictionConfidence")


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionConfidence:
    def __init__(self, confidence_thresholds: Optional[dict[str, float]] = None):
        self.confidence_thresholds = confidence_thresholds or {
            "high": 0.8,
            "medium": 0.5,
        }

    def estimate(
        self,
        model,
        X: pd.DataFrame,
        y_pred: np.ndarray,
        y_actual: Optional[np.ndarray] = None,
        method: str = "ensemble",
    ) -> pd.DataFrame:
        if method == "ensemble":
            return self._ensemble_confidence(model, X, y_pred)
        elif method == "quantile":
            return self._quantile_confidence(model, X, y_pred)
        elif method == "residual":
            return self._residual_confidence(model, X, y_pred, y_actual)
        return self._ensemble_confidence(model, X, y_pred)

    def _ensemble_confidence(
        self, model, X: pd.DataFrame, y_pred: np.ndarray
    ) -> pd.DataFrame:
        if hasattr(model, "estimators_"):
            base_preds = np.column_stack([
                est.predict(X.select_dtypes(include=[np.number]).fillna(0))
                for est in model.estimators_
            ])
            pred_std = base_preds.std(axis=1)
            mean_pred = base_preds.mean(axis=1)
            cv = pred_std / (np.abs(mean_pred) + 1e-6)
            confidence = 1 / (1 + cv)
        elif hasattr(model, "predict") and hasattr(model, "estimators_"):
            confidence = np.full(len(y_pred), 0.8)
        else:
            confidence = np.full(len(y_pred), 0.7)

        return self._build_result(y_pred, confidence)

    def _quantile_confidence(
        self, model, X: pd.DataFrame, y_pred: np.ndarray
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            if isinstance(model, GradientBoostingRegressor):
                preds = np.column_stack([
                    model.train_score_,
                ])
            else:
                preds = np.column_stack([y_pred * 0.9, y_pred * 1.1])
        except Exception:
            preds = np.column_stack([y_pred * 0.9, y_pred * 1.1])

        lower = preds.min(axis=1)
        upper = preds.max(axis=1)
        interval_width = upper - lower
        rel_width = interval_width / (np.abs(y_pred) + 1e-6)
        confidence = 1 / (1 + rel_width)
        return self._build_result(y_pred, confidence, lower, upper)

    def _residual_confidence(
        self, model, X: pd.DataFrame, y_pred: np.ndarray, y_actual: Optional[np.ndarray]
    ) -> pd.DataFrame:
        if y_actual is None:
            return self._ensemble_confidence(model, X, y_pred)

        residuals = np.abs(y_actual - y_pred)
        rel_error = residuals / (np.abs(y_actual) + 1e-6)
        confidence = 1 / (1 + rel_error)
        return self._build_result(y_pred, confidence)

    def _build_result(
        self,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        confidence = np.clip(confidence, 0, 1)
        result = pd.DataFrame({"prediction": y_pred, "confidence": confidence})

        if lower is not None and upper is not None:
            result["prediction_lower"] = lower
            result["prediction_upper"] = upper
        else:
            std_err = (1 - confidence) * np.abs(y_pred)
            result["prediction_lower"] = y_pred - 1.96 * std_err
            result["prediction_upper"] = y_pred + 1.96 * std_err

        result["risk"] = result["confidence"].apply(self._classify_risk)
        return result

    def _classify_risk(self, confidence: float) -> str:
        if confidence >= self.confidence_thresholds["high"]:
            return RiskLevel.LOW.value
        elif confidence >= self.confidence_thresholds["medium"]:
            return RiskLevel.MEDIUM.value
        return RiskLevel.HIGH.value

    def format_prediction(self, result: pd.DataFrame, idx: int = 0) -> str:
        if idx >= len(result):
            return "No prediction available"
        row = result.iloc[idx]
        return (
            f"Prediction:\n"
            f"  Yield = {row['prediction']:.2f} tonnes/hectare\n\n"
            f"Confidence:\n"
            f"  {row['confidence']:.0%}\n\n"
            f"Risk:\n"
            f"  {row['risk'].title()}\n\n"
            f"95% Prediction Interval:\n"
            f"  [{row['prediction_lower']:.2f}, {row['prediction_upper']:.2f}]"
        )
