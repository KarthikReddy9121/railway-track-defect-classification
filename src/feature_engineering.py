"""
src/feature_engineering.py - Physics-Informed Feature Engineering & Leakage-Free Preprocessing.

Constructs domain interaction metrics based on Indian Railways Permanent Way engineering
and encapsulates all transformations inside a zero-leakage Scikit-Learn Pipeline.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from src.utils import (
    LOGGER,
    RAIL_HEAD_DEPTH_MM,
    STRESS_FREE_TEMP_CELSIUS,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
)

ENGINEERED_NUMERICAL_COLS: List[str] = [
    "track_quality_index",
    "thermal_stress_index",
    "structural_risk_ratio",
    "vibration_dynamic_load",
    "ballast_degradation_factor",
]

ALL_NUMERICAL_COLS: List[str] = NUMERICAL_FEATURE_COLS + ENGINEERED_NUMERICAL_COLS


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer computing physics-informed railway track metrics:
    
    1. track_quality_index (TQI): Composite Euclidean norm of geometry anomalies
       (gauge deviation, alignment error, twist on 3.6m base, vertical unevenness).
    2. thermal_stress_index: Rail thermal divergence from stress-free temperature (40C)
       scaled by dynamic traffic density (GMT).
    3. structural_risk_ratio: Ultrasonic internal flaw depth relative to rail head thickness
       (38.0mm for 52kg vs. 43.5mm for 60kg UIC).
    4. vibration_dynamic_load: Axle box peak vertical acceleration multiplied by traffic GMT factor.
    5. ballast_degradation_factor: Interacting wear from tamping age and monsoon rainfall.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DomainFeatureEngineer expects a pandas DataFrame input.")

        # 1. Composite Track Quality Index (TQI)
        gauge = np.abs(df.get("gauge_deviation_mm", 0.0))
        align = np.abs(df.get("alignment_error_mm", 0.0))
        twist = np.abs(df.get("twist_mm_per_m", 0.0)) * 3.6  # Scale over standard Indian 3.6m base
        uneven = np.abs(df.get("unevenness_mm", 0.0))
        df["track_quality_index"] = np.sqrt(gauge**2 + align**2 + twist**2 + uneven**2)

        # 2. Thermal Stress Index
        rail_temp = df.get("rail_temp_celsius", 35.0)
        traffic_gmt = df.get("traffic_density_gmt", 40.0)
        temp_delta = np.abs(rail_temp - STRESS_FREE_TEMP_CELSIUS)
        df["thermal_stress_index"] = temp_delta * (traffic_gmt / 50.0)

        # 3. Structural Risk Ratio (Flaw Depth / Head Depth)
        flaw_depth = df.get("ultrasonic_flaw_depth_mm", 0.0)
        rail_sections = df.get("rail_section_type", "60kg UIC")
        # Map rail head depth depending on section
        head_depths = np.array([
            RAIL_HEAD_DEPTH_MM.get(str(sec), 43.5) for sec in rail_sections
        ])
        df["structural_risk_ratio"] = np.clip(flaw_depth / head_depths, 0.0, 1.2)

        # 4. Vibration Dynamic Load
        axle_acc = df.get("axle_box_accel_g", 0.15)
        df["vibration_dynamic_load"] = axle_acc * (traffic_gmt / 40.0)

        # 5. Ballast Degradation Factor
        tamp_days = df.get("days_since_last_tamping", 60.0)
        rain_7d = df.get("rainfall_7d_mm", 10.0)
        df["ballast_degradation_factor"] = (tamp_days / 100.0) * (1.0 + (rain_7d / 100.0))

        return df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for numerical scaling and categorical encoding.
    Ensures zero data leakage: Imputers and Scalers compute statistics only during fit.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ALL_NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_FEATURE_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(classifier: Any) -> Pipeline:
    """
    Assembles the full end-to-end classification pipeline:
    Domain Feature Engineering -> Column Transformer Preprocessor -> Classifier.
    """
    pipeline = Pipeline([
        ("domain_fe", DomainFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
    return pipeline


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts output feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    
    # Numerical feature names
    feature_names.extend(ALL_NUMERICAL_COLS)
    
    # Categorical one-hot names
    cat_transformer = preprocessor.named_transformers_.get("cat")
    if cat_transformer is not None and hasattr(cat_transformer, "named_steps"):
        ohe = cat_transformer.named_steps.get("onehot")
        if ohe is not None and hasattr(ohe, "get_feature_names_out"):
            cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURE_COLS))
            feature_names.extend(cat_names)
            
    return feature_names
