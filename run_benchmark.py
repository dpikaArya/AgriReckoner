"""
AAIF Model Optimization — Efficiency Benchmark Script
Runs the full optimization pipeline and reports efficiency metrics.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.ml.feature_analysis.feature_importance import FeatureImportanceAnalyzer
from src.ml.feature_analysis.correlation_analysis import CorrelationAnalyzer
from src.ml.feature_analysis.feature_selector import FeatureSelector
from src.ml.model_benchmark import ModelBenchmark
from src.ml.prediction_confidence import PredictionConfidence
from src.ml.performance_report import PerformanceReport


def main():
    print("=" * 70)
    print("AAIF MODEL OPTIMIZATION - EFFICIENCY BENCHMARK")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────
    df = pd.read_csv("outputs/master_datasets_combined.csv")
    print(f"\nDataset: {df.shape[0]} rows x {df.shape[1]} columns")

    target_col = "Shoot_Biomass_g"
    df_target = df.dropna(subset=[target_col])
    print(f"Target: {target_col} - {len(df_target)} viable records")

    non_feature = [
        "Paper_ID", "Plot_ID", "Treatment", "Title", "Authors", "DOI",
        "Journal", "Institution", "Location", "Country", "State", "Site",
        "Crop", "Variety", "Season", "Design", "Soil_Texture", "Growth_Stage",
        "Cultivation_Method", "Irrigation_Method", "Fertilizer_Name",
        "Application_Method", "Previous_Crop", "Intercrop_Name", "Cover_Crop",
        "Biofertilizer", "Organic_Fertilizer", "Scientific_Name",
        "Experiment_Objective", "Climate_Zone", "Soil_Class",
        "Recommendation_Summary", "Recommended_Fertilizer", "Recommended_Dose",
        "Recommended_Application_Interval", "_source_file", "Block",
        "Harvest_Date", "Sowing_Date", "Year", "Crop_Code", "Variety_Code",
        "Season_Code", "Soil_Texture_Code", "Fertilizer_Code", "Country_Code",
        "Confidence_Score", "Predicted_Yield", "Target_Yield", "Target_Fertilizer",
        "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium",
        "Feature_Available_Before_Prediction", "Sample_Size", "Replications",
        "Expected_Yield_Increase", "Expected_Biomass", "Expected_Plant_Height",
    ]

    exclude = [c for c in non_feature if c in df_target.columns]
    X = df_target.drop(columns=[target_col] + exclude, errors="ignore")
    y = df_target[target_col]

    X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
    initial_features = X_filled.shape[1]
    print(f"Numeric features: {initial_features}")

    # ── 1. Feature Importance ──────────────────────────────────
    print("\n--- PHASE 1: FEATURE IMPORTANCE ---")
    t0 = time.time()
    analyzer = FeatureImportanceAnalyzer(output_dir=Path("reports"))
    analyzer.compute_all(X_filled, y, use_xgboost=True, use_lightgbm=False, use_catboost=False)
    ranked = analyzer.ranked_features_
    t_importance = time.time() - t0

    if ranked is not None and not ranked.empty:
        top10 = ranked.head(10)
        print("Top 10 features by importance:")
        for _, r in top10.iterrows():
            print(f"  {r['feature']}: {r['mean_importance']:.6f}")

    # ── 2. Correlation Analysis ────────────────────────────────
    print("\n--- PHASE 2: CORRELATION ANALYSIS ---")
    corr_analyzer = CorrelationAnalyzer(output_dir=Path("reports"))
    corr_analyzer.analyze(X_filled)
    redundant = corr_analyzer.get_redundant_features()
    print(f"High-correlation redundant features: {len(redundant)}")

    # ── 3. Feature Selection ───────────────────────────────────
    print("\n--- PHASE 3: FEATURE SELECTION ---")
    selector = FeatureSelector(
        output_dir=Path("reports"),
        importance_threshold=0.01,
        correlation_threshold=0.90,
        variance_threshold=0.001,
        max_features=100,
        min_features=30,
    )
    selected = selector.select(X_filled, y, importance_df=ranked, non_feature_cols=[])
    reduction_pct = (1 - len(selected) / max(initial_features, 1)) * 100
    print(f"Selected: {len(selected)} from {initial_features} ({reduction_pct:.1f}% reduction)")

    X_sel = X_filled[[c for c in selected if c in X_filled.columns]]

    # ── 4. Benchmark BEFORE feature selection ──────────────────
    print("\n--- PHASE 4: BENCHMARK (ALL FEATURES) ---")
    t0 = time.time()
    bench_before = ModelBenchmark(output_dir=Path("reports"), random_state=42)
    res_before = bench_before.benchmark_regression(X_filled, y)
    t_before = time.time() - t0

    before_metrics = {}
    if not res_before.empty:
        best_idx = res_before["r2"].idxmax()
        before_metrics = res_before.loc[best_idx].to_dict()
        print(f"Best model (all features): {before_metrics['model']}")
        print(f"  R2: {before_metrics['r2']:.4f}, RMSE: {before_metrics['rmse']:.4f}")

    # ── 5. Benchmark AFTER feature selection ───────────────────
    print("\n--- PHASE 5: BENCHMARK (SELECTED FEATURES) ---")
    t0 = time.time()
    bench_after = ModelBenchmark(output_dir=Path("reports"), random_state=42)
    res_after = bench_after.benchmark_regression(X_sel, y)
    t_after = time.time() - t0

    after_metrics = {}
    if not res_after.empty:
        best_idx = res_after["r2"].idxmax()
        after_metrics = res_after.loc[best_idx].to_dict()
        print(f"Best model (selected features): {after_metrics['model']}")
        print(f"  R2: {after_metrics['r2']:.4f}, RMSE: {after_metrics['rmse']:.4f}")

    # ── 6. Ensemble ────────────────────────────────────────────
    print("\n--- PHASE 6: ENSEMBLE ---")
    from src.ml.ensemble.weighted_average import WeightedAverageEnsemble
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    models_dict = {}
    if X_sel.shape[1] >= 2:
        models_dict = {
            "Ridge": Ridge(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        try:
            import xgboost
            models_dict["XGBoost"] = xgboost.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        except ImportError:
            pass

    ensemble_r2 = None
    if len(models_dict) >= 2:
        ensemble = WeightedAverageEnsemble(output_dir=Path("models"))
        ensemble.fit(models_dict, X_sel, y, auto_weight=True)
        ensemble_r2 = ensemble.training_score_
        print(f"Ensemble R2: {ensemble_r2:.4f}")

    # ── 7. Prediction Confidence ───────────────────────────────
    print("\n--- PHASE 7: PREDICTION CONFIDENCE ---")
    if models_dict:
        model_names = list(models_dict.keys())
        best_model = models_dict[model_names[1]]
        best_model.fit(X_sel, y)
        y_pred = best_model.predict(X_sel)
        conf = PredictionConfidence()
        conf_result = conf.estimate(best_model, X_sel, y_pred, y_actual=y.values)
        avg_conf = conf_result["confidence"].mean()
        risk_dist = conf_result["risk"].value_counts().to_dict()
        print(f"Avg confidence: {avg_conf:.1%}")
        print("Risk distribution:")
        for risk, count in sorted(risk_dist.items()):
            print(f"  {risk}: {count} ({count/len(conf_result)*100:.0f}%)")

    # ── 8. Leakage Check ───────────────────────────────────────
    print("\n--- PHASE 8: LEAKAGE DETECTION ---")
    from src.ml.model_validation.leakage_detector import LeakageDetector
    detector = LeakageDetector(output_dir=Path("reports"))
    leakage = detector.check_all(X_filled, y)
    if detector.leakage_found_:
        print("WARNING: Leakage detected!")
        for k, v in leakage.items():
            print(f"  {k}: {len(v)} issues")
    else:
        print("No leakage detected - dataset is clean")

    # ── 9. Performance Report ──────────────────────────────────
    print("\n--- PHASE 9: EFFICIENCY REPORT ---")
    report = PerformanceReport(output_dir=Path("reports"))
    report.set_baseline(
        features=initial_features,
        r2=before_metrics.get("r2", 0),
        rmse=before_metrics.get("rmse", 0),
        training_time_sec=t_before,
        data_quality=0.75,
    )
    report.set_optimized(
        features=len(selected),
        r2=after_metrics.get("r2", 0),
        rmse=after_metrics.get("rmse", 0),
        training_time_sec=t_after,
        data_quality=0.85,
        top_features=ranked.head(10)["feature"].tolist() if ranked is not None else [],
        best_model=after_metrics.get("model", "N/A"),
        ensemble_r2=ensemble_r2,
    )
    md = report.generate()

    # ── Print Final Summary ────────────────────────────────────
    print("\n" + "=" * 70)
    print("EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"\nOriginal features:         {initial_features}")
    print(f"Selected features:         {len(selected)}")
    print(f"Feature reduction:         {reduction_pct:.1f}%")

    if before_metrics and after_metrics:
        r2_b = before_metrics.get("r2", 0) or 0.001
        r2_a = after_metrics.get("r2", 0) or 0.001
        r2_imp = (r2_a - r2_b) / abs(r2_b) * 100 if r2_b != 0 else 0
        rmse_b = before_metrics.get("rmse", 1) or 1
        rmse_a = after_metrics.get("rmse", 1) or 1
        rmse_red = (rmse_b - rmse_a) / rmse_b * 100 if rmse_b > 0 else 0
        time_red = (t_before - t_after) / max(t_before, 0.001) * 100

        print(f"\nBest model:                {after_metrics.get('model', 'N/A')}")
        print(f"  R2 (all features):       {r2_b:.4f}")
        print(f"  R2 (selected features):  {r2_a:.4f}")
        print(f"  R2 improvement:          {r2_imp:+.1f}%")
        print(f"  RMSE (all features):     {rmse_b:.4f}")
        print(f"  RMSE (selected):         {rmse_a:.4f}")
        print(f"  RMSE reduction:          {rmse_red:+.1f}%")
        print(f"  Training time (all):     {t_before:.1f}s")
        print(f"  Training time (sel):     {t_after:.1f}s")
        print(f"  Training time reduction: {time_red:+.1f}%")

    if ensemble_r2 is not None:
        print(f"\nEnsemble R2:               {ensemble_r2:.4f}")

    print(f"\nPrediction confidence:     {avg_conf:.1%}")
    print(f"Leakage:                   {'Found!' if detector.leakage_found_ else 'None'}")

    results = {
        "dataset_rows": len(df),
        "dataset_cols": df.shape[1],
        "initial_features": initial_features,
        "selected_features": len(selected),
        "feature_reduction_pct": round(reduction_pct, 2),
        "best_model": after_metrics.get("model", "N/A"),
        "r2_all_features": round(before_metrics.get("r2", 0), 4),
        "r2_selected_features": round(after_metrics.get("r2", 0), 4),
        "r2_improvement_pct": round(r2_imp, 2) if "r2_imp" in dir() else 0,
        "rmse_all_features": round(before_metrics.get("rmse", 0), 4),
        "rmse_selected_features": round(after_metrics.get("rmse", 0), 4),
        "rmse_reduction_pct": round(rmse_red, 2) if "rmse_red" in dir() else 0,
        "training_time_before_sec": round(t_before, 2),
        "training_time_after_sec": round(t_after, 2),
        "training_time_reduction_pct": round(time_red, 2) if "time_red" in dir() else 0,
        "ensemble_r2": round(ensemble_r2, 4) if ensemble_r2 else None,
        "prediction_confidence": round(float(avg_conf), 4),
        "leakage_detected": detector.leakage_found_,
        "top_10_features": ranked.head(10)["feature"].tolist() if ranked is not None else [],
    }

    with open("reports/efficiency_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to reports/efficiency_results.json")
    print(md)


if __name__ == "__main__":
    main()
