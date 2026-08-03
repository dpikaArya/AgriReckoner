# Training Readiness Blockers

**Verdict:** FAIL - retraining skipped
**Verified papers:** 325 (min 50)
**UAMS rows:** 332 | **Columns:** 305
**Missing value ratio:** 0.977 (max 0.3)

## Failing gates
- **min_observations_per_target**: actual=1, threshold=100 (gte))
- **min_samples_per_crop**: actual=1, threshold=10 (gte))
- **min_samples_per_feature**: actual=0, threshold=10 (gte))
- **max_missing_value_ratio**: actual=0.9773, threshold=0.3 (lte))
- **duplicate_removal_complete**: actual=False, threshold=True (gte))

## Why retraining is skipped
The Training Readiness Gate must be fully green before any model in the zoo (mlr, random_forest, xgboost, lightgbm, catboost, svr, ann, ...) is retrained. The pipeline therefore continues without a model cycle; a diagnostic is emitted and the gate will be re-evaluated after the next harvest expands verified original field-experiment coverage toward the targets.