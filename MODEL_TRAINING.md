# Model Training — Agricultural Intelligence Framework

## Training Workflow

```
Feature Matrix (146 cols) ──► ModelSelectionAgent ──► TrainingAgent ──► PredictionAgent
                                      │                      │
                                      ▼                      ▼
                              Adaptive pool           Cross-validated
                              GridSearchCV            evaluation + export
                              Nested CV               Feature importance
                              Leakage check           Model persistence
```

The training pipeline flows through three agents:

1. **ModelSelectionAgent** (step 16) — selects candidate models, tunes hyperparameters, performs nested CV
2. **TrainingAgent** (step 17) — trains all viable models, evaluates, exports metrics and feature importance
3. **PredictionAgent** (step 18) — loads best models, generates predictions for all targets

## Supported Model Types

### Model Selection Pool (`ModelSelectionAgent._get_model_pool`)

| Model | Min Samples | Parameter Grid |
|-------|-------------|----------------|
| Ridge | 1 | `alpha`: [0.01, 0.1, 1.0, 10.0, 100.0] |
| Random Forest | 25 | `n_estimators`: [100, 200], `max_depth`: [None, 10, 20], `min_samples_split`: [2, 5] |
| Gradient Boosting | 50 | `n_estimators`: [100, 200], `learning_rate`: [0.01, 0.1, 0.2], `max_depth`: [3, 5] |
| XGBoost | 40 | `n_estimators`: [100, 200], `learning_rate`: [0.01, 0.1, 0.2], `max_depth`: [3, 5, 7] |
| MLP (Neural Net) | 150 | `hidden_layer_sizes`: [(64, 32), (128, 64)], `alpha`: [0.0001, 0.001] |

### Training Pool (`TrainingAgent._get_models`)

| Model | Min Samples | Notes |
|-------|-------------|-------|
| XGBoost | 50 | Requires `libomp` on macOS; skipped if import fails |
| Random Forest | 30 | `n_estimators=100`, `n_jobs=-1` |
| Linear Regression | 10 | Baseline model, no hyperparameters |

### Sample Size Guidelines

| Sample Count | Recommended Models |
|--------------|-------------------|
| < 25 | Ridge only |
| 25–39 | Ridge + Random Forest |
| 40–49 | Ridge + Random Forest + XGBoost |
| 50–149 | Ridge + Random Forest + XGBoost + Gradient Boosting |
| ≥ 150 | All models including MLP |

## Feature Selection and Leakage Prevention

### Leakage Detection (`agri_ai_agent/ml/leakage.py`)

A feature must never encode information unavailable at prediction time. The `is_leaky_feature()` function detects leaks at two levels:

**Name-based detection:**
- Columns in `OUTCOME_COLUMNS` (post-harvest + target + prediction groups from UAMS)
- Columns matching `OUTCOME_INTERACTION_ROOTS` ("yield", "biomass")
- Engineered features derived from outcomes (matched via compact/acronym signatures)
- Same-family collinearity: features sharing significant tokens with the target column
- Stripped engineered suffixes: `_calc`, `_squared`, `_cubed`, `_log1p`, `_log`, `_sqrt`, `_zscore`, `_7d_ma`, etc.

**Correlation-based detection:**
- `drop_suspected_leaks()` drops features with |Pearson correlation| ≥ 0.999 to the target
- Safety net for outcome-derived features the name-based guard might miss

### Feature Matrix Construction

`select_feature_columns()` returns numeric columns that are safe predictors:

```python
def select_feature_columns(df, target_col, base_exclude=frozenset()):
    numeric_cols = df.select_dtypes(include="number").columns
    return [
        col for col in numeric_cols
        if col != target_col
        and col not in NON_FEATURE_COLS
        and col not in base_exclude
        and not is_leaky_feature(col, target_col)
    ]
```

`NON_FEATURE_COLS` excludes identifiers, metadata, categorical codes, and design variables. `POST_HARVEST_VARIABLES` excludes yield outcomes, grain quality, derived yield metrics, and economic parameters.

## Hyperparameter Tuning

The `ModelSelectionAgent` uses `GridSearchCV` for hyperparameter optimisation:

```python
search = GridSearchCV(model, param_grid, cv=inner_cv, scoring=scoring,
                      n_jobs=-1, refit=True, error_score=np.nan)
```

- **Inner CV folds**: 3 (for nested cross-validation)
- **Scoring**: `neg_root_mean_squared_error` (regression) or `accuracy` (classification)
- **Refit**: True — best estimator is retained
- Tuning is skipped when `optimize=False` or param_grid is empty

### Nested Cross-Validation

When sample size permits (`n_samples ≥ outer_splits × 3`), nested CV provides an unbiased generalisation estimate:

```
Outer loop (5-fold):
    Inner loop (3-fold GridSearchCV on outer training fold):
        Tune hyperparameters
    Evaluate best model on outer test fold
```

For smaller datasets, single-level CV reports the in-sample tuning score, which is recognised as optimistically biased.

## Cross-Validation Strategy

Honest evaluation is enforced by `agri_ai_agent/ml/evaluation.py`:

| Sample Count | CV Scheme | Robustness | Behaviour |
|-------------|-----------|------------|-----------|
| < 8 | None | N/A | Refuse to report any metric ("insufficient data") |
| 8–29 | Leave-One-Out | Advisory | Cross-validated but flagged non-robust |
| ≥ 30 | Repeated 5×3-fold | Robust | RepeatedKFold(n_splits=5, n_repeats=3) |

**Key integrity rule:** `SimpleImputer(strategy="median")` is fitted inside each fold via a `Pipeline`, ensuring no test-set information leaks into training imputation.

## Model Evaluation Metrics

The `evaluate_model()` function returns:

| Metric | Description |
|--------|-------------|
| `r2` | Coefficient of determination (cross-validated mean) |
| `r2_std` | Standard deviation across CV folds |
| `rmse` | Root mean squared error |
| `rmse_std` | RMSE standard deviation across folds |
| `n` | Number of samples used |
| `cv_scheme` | `"leave-one-out"` or `"repeated-5-fold"` |
| `robust` | `True` if n ≥ 30 |
| `caveat` | Advisory note for small-n evaluations |

### Model Readiness Report

Before training, `TrainingAgent._check_readiness()` produces a readiness report covering:
- Compatible vs incompatible models (with issue details)
- Data quality: missing values, unencoded categoricals, sample size
- Per-model compatibility: `XGBoost` requires n≥50 and no missing values; `Random Forest` requires n≥30; `Linear Regression` requires n≥10

## Experiment Tracking with MLflow

The `ExperimentTracker` (`src/ml/experiment_tracker.py`) provides MLflow integration with automatic local fallback:

### MLflow Mode (when `mlflow` is installed)

```python
tracker = ExperimentTracker(
    experiment_name="agri_ai_training",
    tracking_uri="mlruns"
)
tracker.start_run(run_name="yield_v1")
tracker.log_params({"model": "XGBoost", "learning_rate": 0.1})
tracker.log_metrics({"r2": 0.85, "rmse": 120.5})
tracker.log_model(model, "xgboost_yield")
tracker.log_dataset(X_train, "training_data")
tracker.end_run()
```

### Local Fallback (when `mlflow` is not available)

Metrics, parameters, tags, and artifacts are saved to:
```
mlruns/{experiment_name}/{run_id}/
├── params.json
├── metrics.json
├── tags.json
├── dataset_{name}.json
├── feature_importance.json
└── artifacts/
    └── model.pkl
```

### Run Management

| Method | Purpose |
|--------|---------|
| `start_run(run_name)` | Start a new run, return run_id |
| `end_run()` | End run, log duration |
| `log_params(params)` | Log hyperparameters |
| `log_metrics(metrics, step)` | Log evaluation metrics |
| `log_artifact(local_path)` | Store artifact file |
| `log_model(model, name)` | Serialise and store model |
| `log_dataset(df, name)` | Log dataset snapshot with hash |
| `set_tag(key, value)` | Set run metadata tag |
| `get_best_run(experiment, metric)` | Retrieve best run by metric |
| `list_runs(experiment)` | List all experiment runs |

### Training Agent Artifacts

The `TrainingAgent` produces:

| Artifact | Format | Contents |
|----------|--------|----------|
| `metrics.csv` | CSV | Per-model R², RMSE, n, CV scheme, robustness |
| `training_report.html` | HTML | Leaderboard table with feature importance |
| `Training_Documentation.md` | Markdown | Model list, scores, best model |
| `Feature_Dictionary.csv` | CSV | Column profiles: type, unique values, missing % |
| `feature_importance.csv` | CSV | Per-model feature importance scores |
| `{model}_{target}.joblib` | Joblib | Serialised impute+model Pipeline |
| `yield_model.pkl` | Pickle | Best overall model |
| `Model_Readiness_Report.md` | Markdown | Pre-training readiness assessment |
| `model_selection_report.md` | Markdown | Model selection leaderboard |
| `model_selection_leaderboard.csv` | CSV | CV scores across candidate models |

### Output Location

All artifacts are written to `{OUTPUT_DIR}/models/` (default: `outputs/models/`).

## Continuous Learning Workflow

The `Orchestrator.run_continuous()` method (`agri_ai_agent/orchestrator.py`) supports two paths:

### Full Cycle Mode

```python
agent = ContinuousLearningAgent(settings)
contract = agent.run(df=dataframe, papers_dir=papers_dir)
```

Runs the full continuous learning pipeline including drift detection, registry updates, and model versioning.

### Incremental Engine Mode

```python
engine = IncrementalEngine(settings, registry, history, detector, graph)
cycle_result = engine.execute_cycle(dataframe, run_pipeline_fn=run_step)
```

Uses `ChangeDetector` to identify which data sources have changed, resolves the dependency graph to determine which pipeline steps need re-execution, and runs only those steps. `VersionHistory` tracks model and dataset versions across cycles.

### Continuous Learning Components

| Component | File | Purpose |
|-----------|------|---------|
| `IncrementalEngine` | `continuous_learning/incremental_engine.py` | Orchestrates change-driven partial pipeline runs |
| `ChangeDetector` | `continuous_learning/change_detector.py` | Diff-based detection of source changes |
| `RepositoryRegistry` | `continuous_learning/repository_registry.py` | Tracks source repository versions |
| `VersionHistory` | `continuous_learning/version_history.py` | Model and dataset version ledger |
| `DependencyGraph` | `continuous_learning/dependency_graph.py` | Maps step dependencies for minimal re-execution |

## Pre-Training Safety Checks

The `PreTrainingChecks` class (`src/ml/pre_training_checks.py`) validates the feature matrix before any model is trained:

| Check | Threshold | Failure Behaviour |
|-------|-----------|-------------------|
| Target exists | y is not None and non-empty | FAIL |
| Dataset not empty | n ≥ 10 samples | FAIL |
| Missing values | ≤ 50% per column | FAIL if exceeded |
| Duplicate samples | ≤ 80% duplicates | WARN if exceeded |
| Feature variance | ≥ 1e-10 | FAIL for zero-variance features |
| Outliers (Z-score) | ≤ 10% beyond ±3σ | WARN if exceeded |
| Train/test overlap | 0 overlapping rows | FAIL if any |
| Data leakage | Target not in X | FAIL if target in features |
| Class balance | ≥ 2 per class | FAIL (classification only) |

Usage:

```python
from src.ml.pre_training_checks import PreTrainingChecks

checker = PreTrainingChecks(config={
    "max_missing_pct": 0.5,
    "min_samples": 10,
})
result = checker.check_all(X, y, task="regression")
if not result["passed"]:
    checker.raise_if_failed(result)
```
