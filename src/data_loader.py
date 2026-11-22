"""
src/data_loader.py - Dataset Generator, Schema Validator, and Stratified Splitter.

Generates realistic Indian Railways Track Geometry & Ultrasonic Flaw Detection (USFD)
inspection data calibrated against Indian Railways Permanent Way Manual (IRPWM) tolerances.
Also supports loading external raw inspection CSVs with validation and outlier handling.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    LOGGER,
    SEVERITY_MAPPING,
    IR_ZONE_DIVISIONS,
    RAIL_SECTIONS,
    SLEEPER_TYPES,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    TARGET_COL,
    ID_COL,
)


def generate_indian_railways_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a domain-calibrated Indian Railways Track Inspection & Sensor Dataset.
    
    Includes realistic class imbalance:
        - Low (Normal Monitoring): ~55%
        - Moderate (Scheduled Maintenance): ~25%
        - High (Urgent Speed Restriction): ~14%
        - Critical (Immediate Line Block & Rail Clamp): ~6%
        
    Incorporates physical relationships between track geometry parameters,
    USFD ultrasonic flaw depths, dynamic axle acceleration, rail temperature,
    traffic density (GMT), and subgrade rainfall conditions.
    """
    np.random.seed(random_state)
    
    # 1. Base Track & Operational Attributes
    observation_ids = [f"IR-TRC-{2024}-{i+1:05d}" for i in range(n_samples)]
    zone_divisions = np.random.choice(
        IR_ZONE_DIVISIONS,
        size=n_samples,
        p=[0.16, 0.15, 0.14, 0.13, 0.14, 0.12, 0.08, 0.08],
    )
    track_km_posts = np.round(np.random.uniform(5.0, 950.0, size=n_samples), 2)
    rail_sections = np.random.choice(RAIL_SECTIONS, size=n_samples, p=[0.35, 0.65])
    sleeper_types = np.random.choice(SLEEPER_TYPES, size=n_samples, p=[0.85, 0.05, 0.10])
    
    # Traffic density (Gross Million Tonnes per annum) - heavier on freight / trunk corridors
    traffic_density_gmt = np.round(np.random.gamma(shape=5.0, scale=8.0, size=n_samples) + 10.0, 1)
    traffic_density_gmt = np.clip(traffic_density_gmt, 5.0, 115.0)
    
    # Maintenance interval (days since last tamping cycle)
    days_since_last_tamping = np.random.exponential(scale=140.0, size=n_samples) + 15.0
    days_since_last_tamping = np.clip(days_since_last_tamping, 5.0, 600.0).round().astype(int)
    
    # Environmental factors
    # Rail temperature (-5C in Northern winter to 70C on Western/Central summer afternoons)
    rail_temp_celsius = np.round(np.random.normal(loc=38.0, scale=14.0, size=n_samples), 1)
    rail_temp_celsius = np.clip(rail_temp_celsius, -4.0, 71.0)
    
    # 7-day cumulative rainfall (monsoon surges cause ballast dilation)
    rainfall_7d_mm = np.random.exponential(scale=35.0, size=n_samples)
    rainfall_7d_mm = np.clip(np.round(rainfall_7d_mm, 1), 0.0, 320.0)
    
    # 2. Defect Tier Latent Assignment (strictly targeting ~55% Low, ~25% Moderate, ~14% High, ~6% Critical)
    # Using uniform quantiles to ensure exact target class distribution
    raw_scores = np.random.uniform(0.0, 1.0, size=n_samples)
    
    # Target cutoffs:
    # 0.00 to 0.55 -> Low (55%)
    # 0.55 to 0.80 -> Moderate (25%)
    # 0.80 to 0.94 -> High (14%)
    # 0.94 to 1.00 -> Critical (6%)
    
    gauge_deviations = []
    alignment_errors = []
    twists = []
    unevennesses = []
    flaw_depths = []
    axle_accels = []
    severity_levels = []
    
    for score in raw_scores:
        if score < 0.55:
            # Low Severity: Normal track geometry within IRPWM tolerance
            g = np.random.normal(loc=0.5, scale=2.0)
            a = np.random.exponential(scale=1.5) + 0.3
            tw = np.random.exponential(scale=0.7) + 0.2
            un = np.random.normal(loc=3.2, scale=1.4)
            # USFD flaw: 96% zero or microscopic hairline < 2.0 mm
            fl = 0.0 if np.random.rand() > 0.06 else np.random.uniform(0.5, 2.0)
            acc = np.random.normal(loc=0.14, scale=0.04)
            assigned = 0 if np.random.rand() > 0.04 else 1
        elif score < 0.80:
            # Moderate Severity: Progressive wear, requires scheduled tamping / grinding
            g = np.random.normal(loc=6.2, scale=2.2)
            a = np.random.exponential(scale=2.5) + 2.8
            tw = np.random.exponential(scale=1.0) + 2.0
            un = np.random.normal(loc=8.5, scale=2.0)
            # Incipient flaw: 2.5 to 8.0 mm
            fl = np.random.choice([0.0, np.random.uniform(2.5, 8.0)], p=[0.30, 0.70])
            acc = np.random.normal(loc=0.32, scale=0.07)
            assigned = 1 if np.random.rand() > 0.05 else (0 if np.random.rand() > 0.5 else 2)
        elif score < 0.94:
            # High Severity: Significant geometry faults or growing transverse defect
            g = np.random.normal(loc=11.5, scale=2.5)
            a = np.random.exponential(scale=3.5) + 6.0
            tw = np.random.exponential(scale=1.2) + 3.8
            un = np.random.normal(loc=13.8, scale=2.5)
            # USFD flaw: 9.0 to 17.0 mm
            fl = np.random.uniform(9.0, 17.5)
            acc = np.random.normal(loc=0.54, scale=0.09)
            assigned = 2 if np.random.rand() > 0.05 else (1 if np.random.rand() > 0.5 else 3)
        else:
            # Critical Severity: Fractured rail risk, severe gauge spread, immediate derailment hazard
            g = np.random.normal(loc=18.5, scale=3.0)
            a = np.random.exponential(scale=4.5) + 9.5
            tw = np.random.exponential(scale=1.8) + 5.5
            un = np.random.normal(loc=18.5, scale=3.0)
            # Severe USFD transverse fissure or web crack: 18.0 to 34.0 mm
            fl = np.random.uniform(18.0, 33.0)
            acc = np.random.normal(loc=0.85, scale=0.16)
            assigned = 3 if np.random.rand() > 0.05 else 2
            
        gauge_deviations.append(g)
        alignment_errors.append(a)
        twists.append(tw)
        unevennesses.append(un)
        flaw_depths.append(fl)
        axle_accels.append(acc)
        severity_levels.append(assigned)
        
    gauge_deviation_mm = np.clip(np.round(gauge_deviations, 2), -5.5, 28.0)
    alignment_error_mm = np.clip(np.round(alignment_errors, 2), 0.1, 24.0)
    twist_mm_per_m = np.clip(np.round(twists, 2), 0.1, 9.5)
    unevenness_mm = np.clip(np.round(unevennesses, 2), 0.5, 26.0)
    ultrasonic_flaw_depth_mm = np.clip(np.round(flaw_depths, 2), 0.0, 35.0)
    axle_box_accel_g = np.clip(np.round(axle_accels, 3), 0.04, 1.45)
    
    severity_levels = np.array(severity_levels)
    severity_names = [SEVERITY_MAPPING[s] for s in severity_levels]
    
    df = pd.DataFrame({
        ID_COL: observation_ids,
        "zone_division": zone_divisions,
        "track_km_post": track_km_posts,
        "rail_section_type": rail_sections,
        "sleeper_type": sleeper_types,
        "traffic_density_gmt": traffic_density_gmt,
        "gauge_deviation_mm": gauge_deviation_mm,
        "alignment_error_mm": alignment_error_mm,
        "twist_mm_per_m": twist_mm_per_m,
        "unevenness_mm": unevenness_mm,
        "ultrasonic_flaw_depth_mm": ultrasonic_flaw_depth_mm,
        "axle_box_accel_g": axle_box_accel_g,
        "rail_temp_celsius": rail_temp_celsius,
        "rainfall_7d_mm": rainfall_7d_mm,
        "days_since_last_tamping": days_since_last_tamping,
        TARGET_COL: severity_levels,
        "severity_name": severity_names,
    })
    
    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates presence and basic types of mandatory inspection columns.
    
    Returns:
        (is_valid, list_of_missing_or_invalid_columns)
    """
    errors = []
    expected_cols = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    for col in expected_cols:
        if col not in df.columns:
            errors.append(f"Missing mandatory column: '{col}'")
            
    # Check for target column if training
    if TARGET_COL in df.columns:
        invalid_labels = set(df[TARGET_COL].dropna().unique()) - set(SEVERITY_MAPPING.keys())
        if invalid_labels:
            errors.append(f"Target column contains invalid severity codes: {invalid_labels}")
            
    is_valid = len(errors) == 0
    return is_valid, errors


def clean_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw inspection dataset, caps unphysical sensor readings,
    and fills potential missing values using domain medians.
    """
    cleaned = df.copy()
    
    # Cap gauge deviation within realistic physical limits (-10mm to +35mm)
    if "gauge_deviation_mm" in cleaned.columns:
        cleaned["gauge_deviation_mm"] = cleaned["gauge_deviation_mm"].clip(-10.0, 35.0)
        
    # Flaw depth cannot be negative or exceed rail total profile depth (45mm)
    if "ultrasonic_flaw_depth_mm" in cleaned.columns:
        cleaned["ultrasonic_flaw_depth_mm"] = cleaned["ultrasonic_flaw_depth_mm"].clip(0.0, 45.0)
        
    # Axle box acceleration in g (0.01 to 2.5g)
    if "axle_box_accel_g" in cleaned.columns:
        cleaned["axle_box_accel_g"] = cleaned["axle_box_accel_g"].clip(0.01, 2.5)
        
    # Fill any null values if present
    for col in NUMERICAL_FEATURE_COLS:
        if col in cleaned.columns and cleaned[col].isnull().any():
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            
    for col in CATEGORICAL_FEATURE_COLS:
        if col in cleaned.columns and cleaned[col].isnull().any():
            mode_val = cleaned[col].mode()[0]
            cleaned[col] = cleaned[col].fillna(mode_val)
            
    return cleaned


def load_or_generate_dataset(
    data_path: str = "data/raw/railway_track_inspections.csv",
    n_samples: int = 5000,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """
    Loads raw inspection dataset from CSV, or generates a synthetic Indian Railways
    domain dataset if the file does not exist or force_regenerate is True.
    """
    path = Path(data_path)
    if path.exists() and not force_regenerate:
        LOGGER.info(f"Loading existing raw dataset from: {path.resolve()}")
        df = pd.read_csv(path)
        valid, errors = validate_schema(df)
        if not valid:
            LOGGER.warning(f"Schema validation warnings in {path}: {errors}. Re-generating clean dataset.")
            df = generate_indian_railways_dataset(n_samples=n_samples)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
    else:
        LOGGER.info(f"Generating synthetic Indian Railways track inspection dataset ({n_samples} samples)...")
        df = generate_indian_railways_dataset(n_samples=n_samples)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        LOGGER.info(f"Saved generated raw dataset to: {path.resolve()}")
        
    df = clean_and_handle_outliers(df)
    
    # Log class distribution
    class_dist = df[TARGET_COL].value_counts(normalize=True).sort_index().to_dict()
    LOGGER.info(f"Dataset loaded successfully with {len(df)} samples.")
    LOGGER.info(f"Class distribution: { {SEVERITY_MAPPING[k]: f'{v*100:.1f}%' for k, v in class_dist.items()} }")
    
    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_processed: bool = True,
    processed_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/test split on target severity_level and optionally saves
    train.csv and test.csv to data/processed/.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    if save_processed:
        p_dir = Path(processed_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        train_path = p_dir / "train.csv"
        test_path = p_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        LOGGER.info(f"Stratified train ({len(train_df)}) saved to: {train_path.resolve()}")
        LOGGER.info(f"Stratified test ({len(test_df)}) saved to: {test_path.resolve()}")
        
    return train_df, test_df


if __name__ == "__main__":
    df = load_or_generate_dataset(force_regenerate=True)
    train_df, test_df = get_train_test_split(df)
    print("Data loading & splitting complete.")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
