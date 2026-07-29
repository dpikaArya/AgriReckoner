import json
import time
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import mlflow
    import mlflow.pyfunc

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str = "agri_ai_training",
        tracking_uri: str = "mlruns",
        run_name: str = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self._run_id = None
        self._start_time = None

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
        else:
            warnings.warn(
                "MLflow is not installed. Logging to local files instead."
            )

    def start_run(self, run_name: str = None) -> Optional[str]:
        self._start_time = time.time()
        effective_run_name = run_name or self.run_name

        if MLFLOW_AVAILABLE:
            run = mlflow.start_run(run_name=effective_run_name)
            self._run_id = run.info.run_id
            return self._run_id

        self._run_id = effective_run_name or f"run_{int(self._start_time)}"
        return self._run_id

    def end_run(self):
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run()
            return

        duration = time.time() - self._start_time if self._start_time else 0
        metrics_path = self._get_run_path() / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)
            existing["training_duration_seconds"] = round(duration, 3)
            with open(metrics_path, "w") as f:
                json.dump(existing, f, indent=2)

        self._run_id = None
        self._start_time = None

    def log_params(self, params: dict):
        if MLFLOW_AVAILABLE:
            mlflow.log_params(params)
            return

        path = self._get_run_path() / "params.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        else:
            existing = {}
        existing.update(params)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def log_metrics(self, metrics: dict, step: int = None):
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)
            return

        path = self._get_run_path() / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        else:
            existing = {}
        for k, v in metrics.items():
            if step is not None:
                existing.setdefault(k, []).append({"value": v, "step": step})
            else:
                existing[k] = v
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def log_artifact(self, local_path: str):
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path)
            return

        src = Path(local_path)
        dst = self._get_run_path() / "artifacts" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    def log_model(self, model, model_name: str, signature=None):
        if MLFLOW_AVAILABLE:
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=model,
                signature=signature,
            )
            return

        import pickle

        path = self._get_run_path() / f"{model_name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def set_tag(self, key: str, value: str):
        if MLFLOW_AVAILABLE:
            mlflow.set_tag(key, value)
            return

        path = self._get_run_path() / "tags.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        else:
            existing = {}
        existing[key] = value
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def log_dataset(self, df: pd.DataFrame, name: str = "training_data"):
        info = {
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "hash": hashlib.md5(
                pd.util.hash_pandas_object(df, index=True).values.tobytes()
            ).hexdigest(),
            "null_counts": df.isnull().sum().to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if MLFLOW_AVAILABLE:
            mlflow.log_dict(info, f"dataset_{name}.json")
            return

        path = self._get_run_path() / f"dataset_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

    def get_best_run(
        self, experiment_name: str, metric: str = "r2"
    ) -> dict:
        if MLFLOW_AVAILABLE:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                return {}
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1,
            )
            if runs.empty:
                return {}
            best = runs.iloc[0].to_dict()
            return {
                "run_id": best.get("run_id"),
                "params": {
                    k.replace("params.", ""): v
                    for k, v in best.items()
                    if k.startswith("params.")
                },
                "metrics": {
                    k.replace("metrics.", ""): v
                    for k, v in best.items()
                    if k.startswith("metrics.")
                },
                "tags": {
                    k.replace("tags.", ""): v
                    for k, v in best.items()
                    if k.startswith("tags.")
                },
            }

        exp_path = Path(self.tracking_uri) / experiment_name
        if not exp_path.exists():
            return {}

        best_run = None
        best_val = float("-inf")
        for run_dir in exp_path.iterdir():
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            val = metrics.get(metric, float("-inf"))
            if isinstance(val, (int, float)) and val > best_val:
                best_val = val
                params_path = run_dir / "params.json"
                params = {}
                if params_path.exists():
                    with open(params_path) as f:
                        params = json.load(f)
                tags_path = run_dir / "tags.json"
                tags = {}
                if tags_path.exists():
                    with open(tags_path) as f:
                        tags = json.load(f)
                best_run = {
                    "run_id": run_dir.name,
                    "params": params,
                    "metrics": metrics,
                    "tags": tags,
                }

        return best_run or {}

    def list_runs(self, experiment_name: str = None) -> list[dict]:
        exp_name = experiment_name or self.experiment_name

        if MLFLOW_AVAILABLE:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                return []
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id]
            )
            if runs.empty:
                return []
            result = []
            for _, row in runs.iterrows():
                result.append(
                    {
                        "run_id": row.get("run_id"),
                        "params": {
                            k.replace("params.", ""): v
                            for k, v in row.items()
                            if k.startswith("params.")
                        },
                        "metrics": {
                            k.replace("metrics.", ""): v
                            for k, v in row.items()
                            if k.startswith("metrics.")
                        },
                        "tags": {
                            k.replace("tags.", ""): v
                            for k, v in row.items()
                            if k.startswith("tags.")
                        },
                        "status": row.get("status"),
                        "start_time": row.get("start_time"),
                        "artifact_uri": row.get("artifact_uri"),
                    }
                )
            return result

        exp_path = Path(self.tracking_uri) / exp_name
        if not exp_path.exists():
            return []

        result = []
        for run_dir in exp_path.iterdir():
            if not run_dir.is_dir():
                continue
            params = {}
            params_path = run_dir / "params.json"
            if params_path.exists():
                with open(params_path) as f:
                    params = json.load(f)
            metrics = {}
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            tags = {}
            tags_path = run_dir / "tags.json"
            if tags_path.exists():
                with open(tags_path) as f:
                    tags = json.load(f)

            result.append(
                {
                    "run_id": run_dir.name,
                    "params": params,
                    "metrics": metrics,
                    "tags": tags,
                }
            )

        return result

    def save_metrics(self, metrics: dict, path: Path = None) -> Path:
        if path is None:
            path = self._get_run_path() / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        return path

    def save_feature_importance(
        self, importance: dict, path: Path = None
    ) -> Path:
        if path is None:
            path = self._get_run_path() / "feature_importance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(importance, f, indent=2)
        return path

    def _get_run_path(self) -> Path:
        base = Path(self.tracking_uri) / self.experiment_name
        if self._run_id:
            return base / self._run_id
        return base / "unknown_run"
