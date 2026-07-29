import numpy as np
import pandas as pd
import warnings
from typing import Optional, Any


class PreTrainingChecks:
    def __init__(self, config: dict = None):
        self.max_missing_pct = 0.5
        self.min_samples = 10
        self.max_duplicate_pct = 0.8
        self.outlier_threshold = 3.0
        self.max_outlier_pct = 0.1
        self.min_feature_variance = 1e-10
        self.min_class_count = 2

        if config is not None:
            self.max_missing_pct = config.get("max_missing_pct", self.max_missing_pct)
            self.min_samples = config.get("min_samples", self.min_samples)
            self.max_duplicate_pct = config.get("max_duplicate_pct", self.max_duplicate_pct)
            self.outlier_threshold = config.get("outlier_threshold", self.outlier_threshold)
            self.max_outlier_pct = config.get("max_outlier_pct", self.max_outlier_pct)
            self.min_feature_variance = config.get("min_feature_variance", self.min_feature_variance)
            self.min_class_count = config.get("min_class_count", self.min_class_count)

    def check_all(self, X: pd.DataFrame, y: pd.Series, task: str = "regression") -> dict:
        checks = {}
        warnings_list = []
        errors_list = []

        result = {
            "passed": True,
            "checks": checks,
            "warnings": warnings_list,
            "errors": errors_list,
        }

        checks["target_exists"] = self._check_target_exists(y)
        if checks["target_exists"]["status"] == "FAIL":
            result["passed"] = False
            errors_list.append(checks["target_exists"]["detail"])

        checks["dataset_not_empty"] = self._check_dataset_not_empty(X)
        if checks["dataset_not_empty"]["status"] == "FAIL":
            result["passed"] = False
            errors_list.append(checks["dataset_not_empty"]["detail"])

        if X is not None and not X.empty:
            checks["missing_values"] = self._check_missing_values(X)
            if checks["missing_values"]["status"] == "FAIL":
                result["passed"] = False
                errors_list.append(checks["missing_values"]["detail"])

            checks["duplicate_samples"] = self._check_duplicate_samples(X)
            if checks["duplicate_samples"]["status"] == "WARN":
                warnings_list.append(checks["duplicate_samples"]["detail"])

            checks["feature_variance"] = self._check_feature_variance(X)
            if checks["feature_variance"]["status"] == "FAIL":
                result["passed"] = False
                errors_list.append(checks["feature_variance"]["detail"])
                for col in checks["feature_variance"].get("zero_var_cols", []):
                    errors_list.append(f"Feature '{col}' has zero variance")

            checks["outliers"] = self._check_outliers(X)
            if checks["outliers"]["status"] == "WARN":
                warnings_list.append(checks["outliers"]["detail"])

        if y is not None and isinstance(y, (pd.Series, np.ndarray)) and len(y) > 0:
            if task == "classification":
                checks["class_balance"] = self._check_class_balance(y)
                if checks["class_balance"]["status"] == "FAIL":
                    result["passed"] = False
                    errors_list.append(checks["class_balance"]["detail"])

        return result

    def _check_target_exists(self, y: Any) -> dict:
        if y is None:
            return {"status": "FAIL", "detail": "Target variable y is None"}
        if isinstance(y, (pd.Series, np.ndarray)) and len(y) == 0:
            return {"status": "FAIL", "detail": "Target variable y is empty"}
        return {"status": "PASS", "detail": "Target variable exists"}

    def _check_dataset_not_empty(self, X: Any) -> dict:
        if X is None:
            return {"status": "FAIL", "detail": "Feature matrix X is None"}
        if isinstance(X, pd.DataFrame) and X.empty:
            return {"status": "FAIL", "detail": "Feature matrix X is empty"}
        if hasattr(X, "__len__") and len(X) < self.min_samples:
            return {
                "status": "FAIL",
                "detail": f"Dataset has {len(X)} samples, minimum required is {self.min_samples}",
            }
        return {"status": "PASS", "detail": "Dataset is not empty"}

    def _check_missing_values(self, X: pd.DataFrame) -> dict:
        missing_pct = X.isnull().mean().to_dict()
        bad_cols = {col: pct for col, pct in missing_pct.items() if pct > self.max_missing_pct}
        if bad_cols:
            details = ", ".join(f"{col}: {pct:.2%}" for col, pct in bad_cols.items())
            return {
                "status": "FAIL",
                "detail": f"Columns exceed max missing {self.max_missing_pct:.0%}: {details}",
                "missing_pct": missing_pct,
            }
        return {"status": "PASS", "detail": "No columns exceed max missing percentage", "missing_pct": missing_pct}

    def _check_duplicate_samples(self, X: pd.DataFrame) -> dict:
        n_total = len(X)
        if n_total == 0:
            return {"status": "PASS", "detail": "No samples to check duplicates", "duplicate_pct": 0.0}
        n_duplicates = X.duplicated().sum()
        duplicate_pct = n_duplicates / n_total
        if duplicate_pct > self.max_duplicate_pct:
            return {
                "status": "WARN",
                "detail": f"{n_duplicates}/{n_total} samples are duplicates ({duplicate_pct:.2%})",
                "duplicate_pct": duplicate_pct,
            }
        return {"status": "PASS", "detail": f"Duplicate samples: {duplicate_pct:.2%}", "duplicate_pct": duplicate_pct}

    def _check_feature_variance(self, X: pd.DataFrame) -> dict:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        zero_var_cols = []
        for col in numeric_cols:
            col_data = X[col].dropna()
            if len(col_data) < 2:
                zero_var_cols.append(col)
            elif np.std(col_data) < self.min_feature_variance:
                zero_var_cols.append(col)
        if zero_var_cols:
            return {
                "status": "FAIL",
                "detail": f"{len(zero_var_cols)} feature(s) have near-zero variance: {zero_var_cols}",
                "zero_var_cols": zero_var_cols,
            }
        return {"status": "PASS", "detail": "All numeric features have sufficient variance", "zero_var_cols": []}

    def _check_outliers(self, X: pd.DataFrame) -> dict:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outlier_pcts = {}
        for col in numeric_cols:
            col_data = X[col].dropna()
            if len(col_data) < 3:
                outlier_pcts[col] = 0.0
                continue
            mean = col_data.mean()
            std = col_data.std()
            if std == 0:
                outlier_pcts[col] = 0.0
                continue
            z_scores = np.abs((col_data - mean) / std)
            pct = (z_scores > self.outlier_threshold).mean()
            outlier_pcts[col] = float(pct)

        bad_cols = {col: pct for col, pct in outlier_pcts.items() if pct > self.max_outlier_pct}
        if bad_cols:
            details = ", ".join(f"{col}: {pct:.2%}" for col, pct in bad_cols.items())
            return {
                "status": "WARN",
                "detail": f"Columns exceed max outlier pct {self.max_outlier_pct:.0%}: {details}",
                "outlier_pct": outlier_pcts,
            }
        return {"status": "PASS", "detail": "No columns exceed max outlier percentage", "outlier_pct": outlier_pcts}

    def _check_train_test_overlap(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
        train_tuples = X_train.astype(str).apply(tuple, axis=1).tolist()
        test_tuples = X_test.astype(str).apply(tuple, axis=1).tolist()
        overlap = set(train_tuples) & set(test_tuples)
        if overlap:
            return {
                "status": "FAIL",
                "detail": f"Found {len(overlap)} overlapping sample(s) between train and test sets",
            }
        return {"status": "PASS", "detail": "No overlap detected between train and test sets"}

    def _check_data_leakage(self, X: pd.DataFrame, target_col: str) -> dict:
        if target_col in X.columns:
            return {
                "status": "FAIL",
                "detail": f"Target column '{target_col}' found in feature matrix X",
            }
        return {"status": "PASS", "detail": "No data leakage detected"}

    def _check_class_balance(self, y: pd.Series) -> dict:
        counts = y.value_counts()
        small_classes = counts[counts < self.min_class_count]
        if not small_classes.empty:
            details = ", ".join(f"class '{c}': {n}" for c, n in small_classes.items())
            return {
                "status": "FAIL",
                "detail": f"Classes with fewer than {self.min_class_count} samples: {details}",
            }
        return {"status": "PASS", "detail": "All classes have sufficient samples"}

    def raise_if_failed(self, result: dict):
        if not result.get("passed", True):
            summary = "Pre-training checks failed:\n"
            for check_name, check_result in result.get("checks", {}).items():
                if check_result.get("status") == "FAIL":
                    summary += f"  - {check_name}: {check_result.get('detail', '')}\n"
            raise ValueError(summary)

    @staticmethod
    def check_and_report(
        X: pd.DataFrame, y: pd.Series, task: str = "regression"
    ) -> dict:
        checker = PreTrainingChecks()
        result = checker.check_all(X, y, task=task)

        print("=" * 70)
        print(f"{'Pre-Training Check Report':^70}")
        print("=" * 70)
        print(f"{'Result':10} {'Check':30} {'Detail'}")
        print("-" * 70)
        for check_name, check_result in result["checks"].items():
            status = check_result.get("status", "N/A")
            detail = check_result.get("detail", "")
            print(f"{status:10} {check_name:30} {detail}")
        print("-" * 70)
        print(f"{'PASS' if result['passed'] else 'FAIL':10} {'OVERALL':30}")
        if result["warnings"]:
            print("\nWarnings:")
            for w in result["warnings"]:
                print(f"  - {w}")
        if result["errors"]:
            print("\nErrors:")
            for e in result["errors"]:
                print(f"  - {e}")
        print("=" * 70)
        return result
