import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger("StackingEnsemble")


class StackingEnsemble:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.base_models_: dict[str, object] = {}
        self.meta_model_ = Ridge(alpha=1.0, random_state=random_state)
        self.is_fitted_ = False

    def fit(
        self,
        base_models: dict[str, object],
        X: pd.DataFrame,
        y: pd.Series,
    ):
        self.base_models_ = base_models
        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))

        kf = KFold(n_splits=min(self.cv_folds, len(X_filled)), shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((len(X_filled), len(base_models)))

        for i, (name, model) in enumerate(base_models.items()):
            oof_pred = np.zeros(len(X_filled))
            for train_idx, val_idx in kf.split(X_filled):
                X_tr, X_val = X_filled.iloc[train_idx], X_filled.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                try:
                    model.fit(X_tr, y_tr)
                    oof_pred[val_idx] = model.predict(X_val)
                except Exception as e:
                    logger.warning("Stacking OOF failed for %s: %s", name, e)
                    oof_pred[val_idx] = y_val.mean()
            meta_features[:, i] = oof_pred

        self.meta_model_.fit(meta_features, y)
        self.is_fitted_ = True

        train_pred = self.meta_model_.predict(meta_features)
        from sklearn.metrics import r2_score
        r2 = r2_score(y, train_pred)
        logger.info("Stacking ensemble trained (meta-R2: %.4f)", r2)
        self._save()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Stacking ensemble not fitted")

        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        meta_features = np.zeros((len(X_filled), len(self.base_models_)))

        for i, (name, model) in enumerate(self.base_models_.items()):
            try:
                meta_features[:, i] = model.predict(X_filled)
            except Exception as e:
                logger.warning("Base model %s prediction failed: %s", name, e)
                meta_features[:, i] = 0

        return self.meta_model_.predict(meta_features)

    def _save(self):
        if not self.is_fitted_:
            return
        meta_coef = self.meta_model_.coef_.tolist() if hasattr(self.meta_model_, "coef_") else []
        info = {
            "base_models": list(self.base_models_.keys()),
            "meta_model": type(self.meta_model_).__name__,
            "meta_coefficients": meta_coef,
            "meta_intercept": float(self.meta_model_.intercept_) if hasattr(self.meta_model_, "intercept_") else 0,
        }
        path = self.output_dir / "stacking_ensemble_info.json"
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        logger.info("Saved stacking ensemble info to %s", path)
