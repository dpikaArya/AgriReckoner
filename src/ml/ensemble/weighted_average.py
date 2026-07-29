import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("WeightedAverageEnsemble")


class WeightedAverageEnsemble:
    def __init__(self, output_dir: Optional[Path] = None, random_state: int = 42):
        self.output_dir = Path(output_dir) if output_dir else Path("models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.weights_: dict[str, float] = {}
        self.models_: dict[str, object] = {}
        self.training_score_: Optional[float] = None

    def fit(
        self,
        models: dict[str, object],
        X: pd.DataFrame,
        y: pd.Series,
        auto_weight: bool = True,
        weights: Optional[dict[str, float]] = None,
    ):
        self.models_ = models

        if weights is not None:
            total = sum(weights.values())
            self.weights_ = {k: v / total for k, v in weights.items()}
        elif auto_weight and len(models) > 0:
            self._auto_weight(X, y)
        else:
            n = len(models)
            self.weights_ = {name: 1.0 / n for name in models}

        self.training_score_ = self._calc_ensemble_score(X, y)
        logger.info(
            "Ensemble weights: %s (R2: %.4f)",
            self.weights_, self.training_score_ or 0,
        )
        self._save()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models_ or not self.weights_:
            raise ValueError("Ensemble not fitted")

        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        predictions = np.zeros(len(X_filled))

        for name, model in self.models_.items():
            weight = self.weights_.get(name, 0)
            if weight > 0:
                try:
                    pred = model.predict(X_filled)
                    predictions += weight * pred
                except Exception as e:
                    logger.warning("Prediction failed for %s: %s", name, e)

        return predictions

    def _auto_weight(self, X: pd.DataFrame, y: pd.Series):
        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))

        X_train, X_val, y_train, y_val = train_test_split(
            X_filled, y, test_size=0.2, random_state=self.random_state
        )

        scores = {}
        for name, model in self.models_.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores[name] = max(0, r2_score(y_val, y_pred))
            except Exception as e:
                logger.warning("Auto-weight failed for %s: %s", name, e)
                scores[name] = 0

        total = sum(scores.values())
        if total > 0:
            self.weights_ = {k: v / total for k, v in scores.items()}
        else:
            n = len(scores)
            self.weights_ = {k: 1.0 / n for k in scores}

    def _calc_ensemble_score(self, X: pd.DataFrame, y: pd.Series) -> Optional[float]:
        try:
            y_pred = self.predict(X)
            return float(r2_score(y, y_pred))
        except Exception:
            return None

    def _save(self):
        weights_path = self.output_dir / "ensemble_weights.json"
        weights_path.write_text(
            json.dumps({
                "weights": self.weights_,
                "training_r2": self.training_score_,
                "models": list(self.models_.keys()),
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved ensemble weights to %s", weights_path)
