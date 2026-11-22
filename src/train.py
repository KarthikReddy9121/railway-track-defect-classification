"""
src/train.py - Multiclass Model Training, Imbalance Benchmarking & Champion Pipeline Serialization.

Compares 4 classifiers under Stratified 5-Fold Cross Validation:
1. Logistic Regression (Class-Weighted Baseline)
2. Random Forest Classifier (Balanced Subsample)
3. XGBoost Classifier (Sample-Weighted)
4. LightGBM Classifier (Class-Weighted Champion)

Optimizes strictly for Macro-F1 to guarantee high recall on minority Critical and High track defects.
Saves the champion pipeline to models/champion_track_defect_pipeline.joblib
and comprehensive metadata to models/model_metadata.json.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb

from src.utils import (
    LOGGER,
    SEVERITY_MAPPING,
    TARGET_COL,
    ID_COL,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
)
from src.data_loader import load_or_generate_dataset, get_train_test_split
from src.feature_engineering import (
    build_full_pipeline,
    DomainFeatureEngineer,
    build_preprocessor,
    get_feature_names_from_preprocessor,
)


def cross_validate_model(
    model_name: str,
    base_estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates a candidate model architecture across Stratified K-Fold CV.
    Guarantees no data leakage by fitting the complete pipeline inside each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    macro_f1_scores = []
    weighted_f1_scores = []
    accuracy_scores = []
    
    LOGGER.info(f"Running {n_splits}-fold Stratified CV for: {model_name}...")
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_f_train, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Instantiate fresh pipeline for fold
        fold_pipeline = build_full_pipeline(base_estimator)
        
        # Fit pipeline
        if model_name == "XGBoost":
            # For XGBoost, compute sample weights for the transformed training data
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_f_train)
            fold_pipeline.fit(X_f_train, y_f_train, classifier__sample_weight=sample_weights)
        else:
            fold_pipeline.fit(X_f_train, y_f_train)
            
        y_val_pred = fold_pipeline.predict(X_f_val)
        
        macro_f1 = f1_score(y_f_val, y_val_pred, average="macro")
        weighted_f1 = f1_score(y_f_val, y_val_pred, average="weighted")
        acc = accuracy_score(y_f_val, y_val_pred)
        
        macro_f1_scores.append(macro_f1)
        weighted_f1_scores.append(weighted_f1)
        accuracy_scores.append(acc)
        
    elapsed = time.time() - start_time
    
    results = {
        "model_name": model_name,
        "cv_macro_f1_mean": float(np.mean(macro_f1_scores)),
        "cv_macro_f1_std": float(np.std(macro_f1_scores)),
        "cv_weighted_f1_mean": float(np.mean(weighted_f1_scores)),
        "cv_accuracy_mean": float(np.mean(accuracy_scores)),
        "training_time_sec": round(elapsed, 2),
    }
    
    LOGGER.info(
        f"[{model_name}] CV Macro-F1: {results['cv_macro_f1_mean']:.4f} +/- {results['cv_macro_f1_std']:.4f} | "
        f"Weighted-F1: {results['cv_weighted_f1_mean']:.4f} | Accuracy: {results['cv_accuracy_mean']:.4f}"
    )
    return results


def run_benchmarks(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes benchmark comparison across 4 multiclass classifiers with class-imbalance adaptations.
    """
    models = {
        "Logistic Regression (Class-Weighted)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        ),
        "Random Forest (Balanced Subsample)": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM (Class-Weighted)": lgb.LGBMClassifier(
            n_estimators=180,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.07,
            class_weight="balanced",
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }
    
    benchmark_records = []
    best_model_name = None
    best_macro_f1 = -1.0
    
    for name, clf in models.items():
        res = cross_validate_model(name, clf, X_train, y_train, n_splits=n_splits, random_state=random_state)
        benchmark_records.append(res)
        if res["cv_macro_f1_mean"] > best_macro_f1:
            best_macro_f1 = res["cv_macro_f1_mean"]
            best_model_name = name
            
    df_bench = pd.DataFrame(benchmark_records).sort_values(by="cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    return df_bench, models[best_model_name], best_model_name


def tune_champion_model(
    champion_name: str,
    base_champion: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Any:
    """
    Applies focused hyperparameter optimization on the champion classifier.
    """
    LOGGER.info(f"Fine-tuning champion model: {champion_name}...")
    
    # Grid of production-tuned configurations optimizing Macro-F1
    if "LightGBM" in champion_name:
        candidate_params = [
            {"n_estimators": 160, "learning_rate": 0.06, "num_leaves": 31, "max_depth": 6},
            {"n_estimators": 200, "learning_rate": 0.08, "num_leaves": 45, "max_depth": 7},
            {"n_estimators": 220, "learning_rate": 0.05, "num_leaves": 35, "max_depth": 8},
        ]
        best_clf = None
        best_score = -1.0
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        
        for params in candidate_params:
            clf = lgb.LGBMClassifier(
                **params,
                class_weight="balanced",
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            scores = []
            for tr_idx, val_idx in skf.split(X_train, y_train):
                pipe = build_full_pipeline(clf)
                pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                preds = pipe.predict(X_train.iloc[val_idx])
                scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
            mean_sc = np.mean(scores)
            if mean_sc > best_score:
                best_score = mean_sc
                best_clf = clf
                best_params = params
        LOGGER.info(f"Best Tuned LightGBM params: {best_params} (CV Macro-F1: {best_score:.4f})")
        return best_clf, best_params
    else:
        # Return base champion if already tuned
        return base_champion, getattr(base_champion, "get_params", lambda: {})()


def train_and_export(
    data_path: str = "data/raw/railway_track_inspections.csv",
    models_dir: str = "models",
    n_samples: int = 5000,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    End-to-end training orchestrator:
    1. Loads dataset and splits into 80/20 train/test.
    2. Benchmarks 4 classifiers under Stratified 5-Fold CV.
    3. Tunes champion classifier for Macro-F1.
    4. Trains champion pipeline on full train set.
    5. Evaluates on test set and exports pipeline + model_metadata.json.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Loading & Splitting
    df = load_or_generate_dataset(data_path=data_path, n_samples=n_samples, force_regenerate=False)
    train_df, test_df = get_train_test_split(df, test_size=0.2, random_state=random_state)
    
    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, "severity_name", ID_COL]]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    
    # Step 2: Imbalance Benchmark Evaluation
    df_bench, base_champion, best_model_name = run_benchmarks(X_train, y_train, n_splits=5, random_state=random_state)
    LOGGER.info("\n=== MODEL BENCHMARK RESULTS (Macro-F1 Sorted) ===\n" + df_bench.to_string(index=False))
    
    # Step 3: Tuning Champion Model
    tuned_champion, best_params = tune_champion_model(best_model_name, base_champion, X_train, y_train, random_state=random_state)
    
    # Step 4: Fit Full Champion Pipeline
    LOGGER.info(f"Training champion pipeline ({best_model_name}) on full training partition ({len(X_train)} samples)...")
    champion_pipeline = build_full_pipeline(tuned_champion)
    champion_pipeline.fit(X_train, y_train)
    
    # Extract feature names from fitted preprocessor
    fe_step = champion_pipeline.named_steps["domain_fe"]
    prep_step = champion_pipeline.named_steps["preprocessor"]
    engineered_sample = fe_step.transform(X_train.head(10))
    feature_names = get_feature_names_from_preprocessor(prep_step)
    
    # Step 5: Held-out Test Set Evaluation
    y_test_pred = champion_pipeline.predict(X_test)
    y_test_proba = champion_pipeline.predict_proba(X_test)
    
    test_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
    test_weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    
    class_rep = classification_report(
        y_test,
        y_test_pred,
        target_names=[SEVERITY_MAPPING[i] for i in sorted(SEVERITY_MAPPING.keys())],
        output_dict=True,
    )
    
    conf_mat = confusion_matrix(y_test, y_test_pred).tolist()
    
    LOGGER.info("=== HELD-OUT TEST EVALUATION ===")
    LOGGER.info(f"Test Macro-F1: {test_macro_f1:.4f}")
    LOGGER.info(f"Test Weighted-F1: {test_weighted_f1:.4f}")
    LOGGER.info(f"Test Accuracy: {test_acc:.4f}")
    LOGGER.info("\n" + classification_report(
        y_test,
        y_test_pred,
        target_names=[SEVERITY_MAPPING[i] for i in sorted(SEVERITY_MAPPING.keys())],
    ))
    
    # Step 6: Serialize Artifacts
    pipeline_export_path = models_path / "champion_track_defect_pipeline.joblib"
    joblib.dump(champion_pipeline, pipeline_export_path)
    LOGGER.info(f"Champion pipeline successfully saved to: {pipeline_export_path.resolve()}")
    
    # Safe JSON serializable best_params
    clean_params = {}
    if isinstance(best_params, dict):
        for k, v in best_params.items():
            if isinstance(v, (int, float, str, bool, list)):
                clean_params[k] = v
            else:
                clean_params[k] = str(v)
                
    metadata = {
        "project_title": "Railway Track Defect Severity Classification",
        "problem_statement_id": "SIH-IR-TRACK-01",
        "training_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "champion_model_name": best_model_name,
        "best_hyperparameters": clean_params,
        "metrics": {
            "test_macro_f1": round(test_macro_f1, 4),
            "test_weighted_f1": round(test_weighted_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "per_class": class_rep,
            "confusion_matrix": conf_mat,
        },
        "benchmark_comparison": df_bench.to_dict(orient="records"),
        "dataset_statistics": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "class_distribution_overall": df[TARGET_COL].value_counts().to_dict(),
        },
        "feature_schema": {
            "raw_numerical_features": NUMERICAL_FEATURE_COLS,
            "raw_categorical_features": CATEGORICAL_FEATURE_COLS,
            "engineered_features": [
                "track_quality_index",
                "thermal_stress_index",
                "structural_risk_ratio",
                "vibration_dynamic_load",
                "ballast_degradation_factor",
            ],
            "transformed_feature_names": feature_names,
            "total_input_dimension": len(feature_names),
        },
    }
    
    metadata_export_path = models_path / "model_metadata.json"
    with open(metadata_export_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info(f"Model metadata saved to: {metadata_export_path.resolve()}")
    
    return champion_pipeline, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate railway defect classification models.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of inspection samples to generate/load.")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed.")
    args = parser.parse_args()
    
    train_and_export(n_samples=args.samples, random_state=args.seed)
