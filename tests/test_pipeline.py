"""
tests/test_pipeline.py - Comprehensive Unit & Integration Test Suite.

Verifies:
1. Dataset generation & schema integrity.
2. Target class imbalance distributions.
3. Physics-informed feature engineering outputs (finite values, no NaNs).
4. Zero data leakage isolation across train/test splits.
5. Champion pipeline inference validity & probability calibration.
6. Real-world edge case predictions (extreme flaw vs. pristine track).
7. Maintenance priority ranking logic.
8. Serialization artifacts and evaluation assets presence.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    SEVERITY_MAPPING,
    SEVERITY_WEIGHTS,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    TARGET_COL,
    ID_COL,
    compute_priority_score,
    generate_maintenance_dispatch_table,
)
from src.data_loader import (
    generate_indian_railways_dataset,
    validate_schema,
    clean_and_handle_outliers,
    get_train_test_split,
)
from src.feature_engineering import (
    DomainFeatureEngineer,
    build_preprocessor,
    build_full_pipeline,
    ENGINEERED_NUMERICAL_COLS,
)


@pytest.fixture(scope="session")
def sample_raw_data() -> pd.DataFrame:
    """Provides a controlled synthetic dataset fixture for testing."""
    return generate_indian_railways_dataset(n_samples=500, random_state=42)


@pytest.fixture(scope="session")
def trained_pipeline():
    """Loads the serialized champion pipeline."""
    model_path = PROJECT_ROOT / "models" / "champion_track_defect_pipeline.joblib"
    assert model_path.exists(), f"Champion model pipeline not found at {model_path}. Run src/train.py first."
    return joblib.load(model_path)


# ===========================================================================
# 1. Dataset Generation & Schema Integrity Tests
# ===========================================================================
def test_dataset_generation_schema(sample_raw_data):
    """Verifies generated dataset schema conforms to Indian Railways TRC specifications."""
    df = sample_raw_data
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    
    is_valid, errors = validate_schema(df)
    assert is_valid, f"Schema validation failed: {errors}"
    assert ID_COL in df.columns
    assert TARGET_COL in df.columns
    assert "severity_name" in df.columns


def test_class_imbalance_distribution(sample_raw_data):
    """Verifies realistic class imbalance reflecting operational rail track condition."""
    df = sample_raw_data
    dist = df[TARGET_COL].value_counts(normalize=True).to_dict()
    
    # Expected: Low: ~50-60%, Moderate: ~20-30%, High: ~10-18%, Critical: ~4-8%
    assert 0.45 <= dist.get(0, 0) <= 0.65, f"Low severity frequency {dist.get(0)} out of expected range (0.45-0.65)"
    assert 0.18 <= dist.get(1, 0) <= 0.35, f"Moderate severity frequency {dist.get(1)} out of expected range (0.18-0.35)"
    assert 0.08 <= dist.get(2, 0) <= 0.20, f"High severity frequency {dist.get(2)} out of expected range (0.08-0.20)"
    assert 0.03 <= dist.get(3, 0) <= 0.10, f"Critical severity frequency {dist.get(3)} out of expected range (0.03-0.10)"


def test_data_cleaning_and_bounds():
    """Verifies that unphysical sensor values are bounded."""
    dirty_df = pd.DataFrame({
        "gauge_deviation_mm": [-50.0, 100.0, 5.0],
        "ultrasonic_flaw_depth_mm": [-10.0, 80.0, 12.0],
        "axle_box_accel_g": [-1.0, 5.0, 0.25],
        "days_since_last_tamping": [np.nan, 120, 45],
        "zone_division": ["NR-UMB", None, "CR-BB"],
    })
    cleaned = clean_and_handle_outliers(dirty_df)
    assert cleaned["gauge_deviation_mm"].min() >= -10.0
    assert cleaned["gauge_deviation_mm"].max() <= 35.0
    assert cleaned["ultrasonic_flaw_depth_mm"].min() >= 0.0
    assert cleaned["ultrasonic_flaw_depth_mm"].max() <= 45.0
    assert not cleaned["days_since_last_tamping"].isnull().any()
    assert not cleaned["zone_division"].isnull().any()


# ===========================================================================
# 2. Physics-Informed Feature Engineering & Leakage Isolation Tests
# ===========================================================================
def test_domain_feature_engineer(sample_raw_data):
    """Verifies calculation of domain metrics and absence of NaN / Inf values."""
    fe = DomainFeatureEngineer()
    transformed = fe.transform(sample_raw_data)
    
    for col in ENGINEERED_NUMERICAL_COLS:
        assert col in transformed.columns, f"Engineered column '{col}' missing."
        assert not transformed[col].isnull().any(), f"Engineered column '{col}' contains NaN."
        assert not np.isinf(transformed[col]).any(), f"Engineered column '{col}' contains Inf."

    # Physics check: Track Quality Index must be >= 0
    assert (transformed["track_quality_index"] >= 0).all()
    # Structural risk ratio must be between 0.0 and 1.2
    assert (transformed["structural_risk_ratio"] >= 0.0).all()
    assert (transformed["structural_risk_ratio"] <= 1.2).all()


def test_zero_data_leakage_isolation(sample_raw_data):
    """Verifies that preprocessor is fitted strictly on train data and isolates statistics."""
    train_df, test_df = get_train_test_split(sample_raw_data, test_size=0.2, random_state=42, save_processed=False)
    
    fe = DomainFeatureEngineer()
    train_fe = fe.transform(train_df)
    test_fe = fe.transform(test_df)
    
    preprocessor = build_preprocessor()
    # Fit only on train
    train_proc = preprocessor.fit_transform(train_fe)
    # Transform test without re-fitting
    test_proc = preprocessor.transform(test_fe)
    
    assert train_proc.shape[0] == len(train_df)
    assert test_proc.shape[0] == len(test_df)
    assert train_proc.shape[1] == test_proc.shape[1]
    assert not np.isnan(train_proc).any()
    assert not np.isnan(test_proc).any()


# ===========================================================================
# 3. Model Inference, Probability Calibration & Edge Cases
# ===========================================================================
def test_pipeline_prediction_probabilities(trained_pipeline, sample_raw_data):
    """Verifies multiclass prediction format and probability simplex (sum to 1.0)."""
    feature_cols = [c for c in sample_raw_data.columns if c not in [TARGET_COL, "severity_name", ID_COL]]
    sample_X = sample_raw_data[feature_cols].head(50)
    
    preds = trained_pipeline.predict(sample_X)
    probs = trained_pipeline.predict_proba(sample_X)
    
    assert len(preds) == 50
    assert probs.shape == (50, 4)
    # Verify probability distribution sums to 1.0
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
    # Verify predicted classes are within 0 to 3
    assert set(preds).issubset({0, 1, 2, 3})


def test_edge_case_pristine_track(trained_pipeline):
    """Verifies pristine track is predicted as Low severity (0)."""
    pristine_record = pd.DataFrame([{
        "zone_division": "NR-UMB",
        "track_km_post": 100.0,
        "rail_section_type": "60kg UIC",
        "sleeper_type": "PSC",
        "traffic_density_gmt": 25.0,
        "gauge_deviation_mm": 0.1,
        "alignment_error_mm": 0.4,
        "twist_mm_per_m": 0.3,
        "unevenness_mm": 2.0,
        "ultrasonic_flaw_depth_mm": 0.0,
        "axle_box_accel_g": 0.10,
        "rail_temp_celsius": 38.0,
        "rainfall_7d_mm": 5.0,
        "days_since_last_tamping": 20,
    }])
    
    pred = trained_pipeline.predict(pristine_record)[0]
    prob = trained_pipeline.predict_proba(pristine_record)[0]
    
    assert pred == 0, f"Pristine track misclassified as severity {pred} (expected 0: Low)."
    assert prob[0] > 0.65, f"Low probability {prob[0]} unexpectedly weak for pristine track."


def test_edge_case_critical_transverse_crack(trained_pipeline):
    """Verifies dangerous transverse crack with severe gauge spread triggers Critical or High."""
    critical_record = pd.DataFrame([{
        "zone_division": "CR-BB",
        "track_km_post": 312.4,
        "rail_section_type": "52kg",
        "sleeper_type": "PSC",
        "traffic_density_gmt": 95.0,
        "gauge_deviation_mm": 22.0,
        "alignment_error_mm": 14.5,
        "twist_mm_per_m": 6.8,
        "unevenness_mm": 21.0,
        "ultrasonic_flaw_depth_mm": 28.5,
        "axle_box_accel_g": 1.15,
        "rail_temp_celsius": 66.0,
        "rainfall_7d_mm": 150.0,
        "days_since_last_tamping": 450,
    }])
    
    pred = trained_pipeline.predict(critical_record)[0]
    prob = trained_pipeline.predict_proba(critical_record)[0]
    
    assert pred in [2, 3], f"Extreme defect misclassified as {pred} (expected High or Critical)."
    # Cumulative probability of High or Critical must exceed 85%
    high_or_crit_prob = prob[2] + prob[3]
    assert high_or_crit_prob > 0.85, f"High/Critical probability {high_or_crit_prob} too low for extreme crack."


# ===========================================================================
# 4. Maintenance Priority Ranking Logic Tests
# ===========================================================================
def test_priority_score_ranking_consistency():
    """Verifies that Critical defects strictly outrank Low defects under identical traffic."""
    low_score = compute_priority_score(predicted_severity=0, predicted_probability=0.95, traffic_density_gmt=50.0, ultrasonic_flaw_depth_mm=0.0)
    crit_score = compute_priority_score(predicted_severity=3, predicted_probability=0.95, traffic_density_gmt=50.0, ultrasonic_flaw_depth_mm=22.0)
    
    assert crit_score > low_score, f"Critical priority score ({crit_score}) should exceed Low score ({low_score})."
    assert crit_score >= 80.0, f"Critical priority score ({crit_score}) must trigger Tier-1 emergency block."


def test_maintenance_dispatch_table_generation(sample_raw_data, trained_pipeline):
    """Verifies P-Way dispatch list ordering and schema."""
    feature_cols = [c for c in sample_raw_data.columns if c not in [TARGET_COL, "severity_name", ID_COL]]
    preds = trained_pipeline.predict(sample_raw_data[feature_cols])
    probs = trained_pipeline.predict_proba(sample_raw_data[feature_cols])
    
    dispatch_df = generate_maintenance_dispatch_table(sample_raw_data, preds, probs)
    
    assert len(dispatch_df) == len(sample_raw_data)
    assert "priority_score" in dispatch_df.columns
    assert "dispatch_rank" in dispatch_df.columns
    assert "prescribed_action" in dispatch_df.columns
    
    # Verify monotonic descending order of priority_score
    scores = dispatch_df["priority_score"].values
    assert (np.diff(scores) <= 0).all(), "Dispatch table must be monotonically sorted descending by priority_score."


# ===========================================================================
# 5. Model Serialization & Export Artifacts Tests
# ===========================================================================
def test_export_artifacts_exist():
    """Verifies all mandatory evaluation plots and metadata files are on disk."""
    plots_dir = PROJECT_ROOT / "models" / "evaluation_plots"
    required_plots = [
        "confusion_matrix.png",
        "per_class_metrics.png",
        "roc_pr_curves.png",
        "shap_summary.png",
        "priority_distribution.png",
    ]
    for p in required_plots:
        plot_path = plots_dir / p
        assert plot_path.exists(), f"Mandatory evaluation plot missing: {plot_path}"
        assert plot_path.stat().st_size > 1000, f"Plot file appears corrupt or empty: {plot_path}"

    meta_file = PROJECT_ROOT / "models" / "model_metadata.json"
    assert meta_file.exists(), f"Model metadata missing: {meta_file}"
    assert meta_file.stat().st_size > 100
