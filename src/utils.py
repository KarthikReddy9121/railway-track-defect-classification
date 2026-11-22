"""
src/utils.py - Logging, Constants, and Indian Railways Maintenance Priority Rules.

Standardized domain mappings compliant with Indian Railways Permanent Way Manual
(IRPWM 2020) and Ultrasonic Flaw Detection (USFD) guidelines.
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
def setup_logger(name: str = "railway_defect_logger", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()

# ---------------------------------------------------------------------------
# Indian Railways Domain Constants
# ---------------------------------------------------------------------------
SEVERITY_MAPPING: Dict[int, str] = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical",
}

SEVERITY_REVERSE_MAPPING: Dict[str, int] = {v: k for k, v in SEVERITY_MAPPING.items()}

SEVERITY_COLORS: Dict[str, str] = {
    "Low": "#28a745",        # Green
    "Moderate": "#ffc107",   # Amber
    "High": "#fd7e14",       # Orange
    "Critical": "#dc3545",   # Crimson / Red
}

SEVERITY_WEIGHTS: Dict[int, float] = {
    0: 1.0,   # Low severity weight
    1: 1.75,  # Moderate severity weight
    2: 3.0,   # High severity weight
    3: 5.0,   # Critical severity weight
}

IR_ACTION_MAPPING: Dict[int, str] = {
    3: "CRITICAL ACTION: Impose Immediate Line Block (Traffic Interruption). Dispatch Emergency Gang for rail clamping or rail piece replacement within 2 hours (IRPWM Para 615).",
    2: "HIGH ACTION: Impose 30 km/h Caution Order. Initiate USFD confirmation re-test within 24 hours. Schedule rail joggled fishplating/splicing within 48 hours (IRPWM Para 618).",
    1: "MODERATE ACTION: Log defect in Track Management System (TMS). Schedule systematic hydraulic tamping and rail profile grinding within 15 days.",
    0: "LOW ACTION: Normal track geometry within permissible tolerances. Continue routine quarterly Track Recording Car (TRC) monitoring.",
}

IR_ZONE_DIVISIONS: List[str] = [
    "NR-UMB",   # Northern Railway - Ambala
    "SR-MAS",   # Southern Railway - Chennai
    "CR-BB",    # Central Railway - Mumbai CSMT
    "ER-HWH",   # Eastern Railway - Howrah
    "SCR-SC",   # South Central Railway - Secunderabad
    "WR-BCT",   # Western Railway - Mumbai Central
    "NCR-PRYJ", # North Central Railway - Prayagraj
    "ECoR-KUR", # East Coast Railway - Khurda Road
]

RAIL_SECTIONS: List[str] = ["52kg", "60kg UIC"]
SLEEPER_TYPES: List[str] = ["PSC", "Wooden", "Steel"]

# Rail head dimension benchmarks (in mm)
RAIL_HEAD_DEPTH_MM: Dict[str, float] = {
    "52kg": 38.0,
    "60kg UIC": 43.5,
}

# Standard Broad Gauge nominal width in Indian Railways is 1676 mm
STANDARD_BG_GAUGE_MM = 1676.0

# Stress-free rail temperature for Indian Railways (52kg / 60kg rails in Zone III/IV)
STRESS_FREE_TEMP_CELSIUS = 40.0

NUMERICAL_FEATURE_COLS: List[str] = [
    "track_km_post",
    "traffic_density_gmt",
    "gauge_deviation_mm",
    "alignment_error_mm",
    "twist_mm_per_m",
    "unevenness_mm",
    "ultrasonic_flaw_depth_mm",
    "axle_box_accel_g",
    "rail_temp_celsius",
    "rainfall_7d_mm",
    "days_since_last_tamping",
]

CATEGORICAL_FEATURE_COLS: List[str] = [
    "zone_division",
    "rail_section_type",
    "sleeper_type",
]

TARGET_COL = "severity_level"
ID_COL = "observation_id"

# ---------------------------------------------------------------------------
# Priority Scoring & Dispatch Ranking Engine
# ---------------------------------------------------------------------------
def compute_priority_score(
    predicted_severity: int,
    predicted_probability: float,
    traffic_density_gmt: float,
    ultrasonic_flaw_depth_mm: float = 0.0,
) -> float:
    """
    Computes a risk-calibrated Maintenance Priority Score (0-100) for P-Way teams.
    
    Formula:
        Score = Base_Severity_Weight * Probability * Traffic_Multiplier + Flaw_Penalty
    Scaled between 0.0 and 100.0.
    """
    sev_weight = SEVERITY_WEIGHTS.get(int(predicted_severity), 1.0)
    # Higher traffic routes (e.g. Dedicated Freight Corridors or Golden Quadrilateral > 60 GMT)
    # escalate maintenance urgency due to rapid dynamic cyclic fatigue.
    traffic_factor = 1.0 + float(np.clip(traffic_density_gmt / 80.0, 0.1, 1.5))
    
    # Flaw depth penalty (deep cracks above 15mm rapidly accelerate to transverse fractures)
    flaw_penalty = float(np.clip(ultrasonic_flaw_depth_mm * 0.8, 0.0, 20.0))
    
    raw_score = (sev_weight * 15.0 * predicted_probability * traffic_factor) + flaw_penalty
    return float(np.clip(round(raw_score, 2), 0.0, 100.0))


def categorize_priority_tier(score: float) -> str:
    """Categorizes priority score into operational P-Way dispatch tiers."""
    if score >= 75.0:
        return "Tier-1 (Immediate Line Block & Emergency Gang)"
    elif score >= 50.0:
        return "Tier-2 (Urgent Caution Order 30 km/h)"
    elif score >= 30.0:
        return "Tier-3 (Scheduled Maintenance Window < 15 Days)"
    else:
        return "Tier-4 (Routine Surveillance)"


def generate_maintenance_dispatch_table(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Constructs an Indian Railways Permanent Way (P-Way) dispatch priority list.
    
    Parameters:
        df: Input inspection DataFrame containing track metadata & sensor readings.
        predictions: Array of predicted severity integers (0-3).
        probabilities: Array of predicted class probabilities (N x 4).
        
    Returns:
        Sorted DataFrame ordered by operational risk priority.
    """
    dispatch_df = df.copy()
    dispatch_df["predicted_severity_code"] = predictions
    dispatch_df["predicted_severity"] = [SEVERITY_MAPPING.get(int(p), "Unknown") for p in predictions]
    
    # Extract prediction confidence (probability of predicted class)
    confidences = [float(probabilities[i, pred]) for i, pred in enumerate(predictions)]
    dispatch_df["prediction_confidence"] = np.round(confidences, 4)
    
    # Compute priority score for each observation
    scores = []
    tiers = []
    actions = []
    for i in range(len(dispatch_df)):
        pred_code = int(predictions[i])
        prob = confidences[i]
        gmt = float(dispatch_df["traffic_density_gmt"].iloc[i]) if "traffic_density_gmt" in dispatch_df.columns else 40.0
        flaw = float(dispatch_df["ultrasonic_flaw_depth_mm"].iloc[i]) if "ultrasonic_flaw_depth_mm" in dispatch_df.columns else 0.0
        
        sc = compute_priority_score(pred_code, prob, gmt, flaw)
        scores.append(sc)
        tiers.append(categorize_priority_tier(sc))
        actions.append(IR_ACTION_MAPPING.get(pred_code, "Inspect and verify track geometry."))
        
    dispatch_df["priority_score"] = scores
    dispatch_df["priority_tier"] = tiers
    dispatch_df["prescribed_action"] = actions
    
    # Sort strictly descending: higher priority scores come first
    dispatch_df = dispatch_df.sort_values(by=["priority_score", "predicted_severity_code"], ascending=[False, False]).reset_index(drop=True)
    dispatch_df["dispatch_rank"] = np.arange(1, len(dispatch_df) + 1)
    
    return dispatch_df
