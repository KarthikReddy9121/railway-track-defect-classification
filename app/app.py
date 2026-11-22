"""
app/app.py - Interactive Streamlit Engineering Dashboard for Indian Railways P-Way Maintenance.

Problem Statement ID: SIH-IR-TRACK-01
Predictive Track Maintenance & Defect Severity Prioritization.
Provides:
1. Real-Time Track Observation Classifier with Presets & IRPWM Safety Directives.
2. Batch Inspection CSV Upload & Maintenance Severity-Priority Dispatch Queue.
3. Model Explainability, SHAP Attributions & Publication Diagnostics.
4. System Architecture, Telemetry & Real-Time Inference Latency Benchmarking.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    SEVERITY_MAPPING,
    SEVERITY_COLORS,
    IR_ZONE_DIVISIONS,
    RAIL_SECTIONS,
    SLEEPER_TYPES,
    IR_ACTION_MAPPING,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    TARGET_COL,
    ID_COL,
    compute_priority_score,
    categorize_priority_tier,
    generate_maintenance_dispatch_table,
)
from src.feature_engineering import DomainFeatureEngineer

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Railways Track Defect Severity Classifier",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Indian Railways Theme Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0b2545 0%, #134074 50%, #8da9c4 100%);
        padding: 22px 28px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: #eef4f8;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .main-header p {
        color: #dbe4ee;
        font-size: 15px;
        margin: 0;
    }
    .badge-critical {
        background-color: #dc3545;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
    }
    .badge-high {
        background-color: #fd7e14;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #ffc107;
        color: #212529;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
    }
    .badge-low {
        background-color: #28a745;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
    }
    .action-box {
        background-color: #f8f9fa;
        border-left: 6px solid #0b2545;
        padding: 16px;
        border-radius: 4px;
        margin-top: 15px;
        font-size: 14.5px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching Model & Metadata
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model_path = PROJECT_ROOT / "models" / "champion_track_defect_pipeline.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_metadata():
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


pipeline = load_pipeline()
metadata = load_metadata()

# Initialize Session Audit Log
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🚆 Indian Railways — Predictive Track Defect Severity Classification</h1>
    <p>Predictive Track Maintenance & Permanent Way (P-Way) Safety Dispatch Engine</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Configuration & Presets
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/4/45/Indian_Railways_logo.svg/1200px-Indian_Railways_logo.svg.png",
    width=90,
)
st.sidebar.title("Operational Controls")

# Presets for Quick Testing
st.sidebar.subheader("Inspection Presets")
preset_choice = st.sidebar.selectbox(
    "Load Real-World Track Condition Preset:",
    [
        "Custom Configuration",
        "Normal BG Track (Routine Monitoring)",
        "Moderate Wear & Ballast Degradation (Scheduled Tamping)",
        "Excessive Gauge Spread (30 km/h Caution Order)",
        "Severe USFD Transverse Flaw (Immediate Line Block)",
    ],
)

# Set defaults based on preset
if preset_choice == "Normal BG Track (Routine Monitoring)":
    default_vals = {
        "zone": "NR-UMB", "km": 142.3, "rail": "60kg UIC", "sleeper": "PSC", "gmt": 35.0,
        "gauge": 0.4, "align": 0.8, "twist": 0.5, "uneven": 2.5, "flaw": 0.0,
        "accel": 0.12, "temp": 38.0, "rain": 5.0, "tamp": 40,
    }
elif preset_choice == "Moderate Wear & Ballast Degradation (Scheduled Tamping)":
    default_vals = {
        "zone": "SR-MAS", "km": 305.5, "rail": "60kg UIC", "sleeper": "PSC", "gmt": 58.0,
        "gauge": 6.8, "align": 5.2, "twist": 2.6, "uneven": 8.8, "flaw": 4.5,
        "accel": 0.34, "temp": 42.0, "rain": 90.0, "tamp": 210,
    }
elif preset_choice == "Excessive Gauge Spread (30 km/h Caution Order)":
    default_vals = {
        "zone": "CR-BB", "km": 78.4, "rail": "52kg", "sleeper": "PSC", "gmt": 75.0,
        "gauge": 13.5, "align": 9.2, "twist": 4.2, "uneven": 14.0, "flaw": 11.5,
        "accel": 0.56, "temp": 52.0, "rain": 40.0, "tamp": 320,
    }
elif preset_choice == "Severe USFD Transverse Flaw (Immediate Line Block)":
    default_vals = {
        "zone": "WR-BCT", "km": 412.8, "rail": "60kg UIC", "sleeper": "PSC", "gmt": 92.0,
        "gauge": 19.5, "align": 13.0, "twist": 6.5, "uneven": 20.5, "flaw": 26.0,
        "accel": 0.95, "temp": 64.0, "rain": 160.0, "tamp": 480,
    }
else:
    default_vals = {
        "zone": "SCR-SC", "km": 210.0, "rail": "60kg UIC", "sleeper": "PSC", "gmt": 45.0,
        "gauge": 3.0, "align": 2.5, "twist": 1.2, "uneven": 4.5, "flaw": 1.0,
        "accel": 0.20, "temp": 39.0, "rain": 25.0, "tamp": 90,
    }

st.sidebar.markdown("---")
# Champion Model Card
if metadata:
    st.sidebar.subheader("Champion Model Card")
    st.sidebar.info(
        f"**Architecture:** {metadata.get('champion_model_name', 'Random Forest')}\n\n"
        f"**Macro-F1 Score:** {metadata.get('metrics', {}).get('test_macro_f1', 0.96):.4f}\n\n"
        f"**Accuracy:** {metadata.get('metrics', {}).get('test_accuracy', 0.96):.4f}\n\n"
        f"**Standard:** Indian Railways P-Way Manual (2020)"
    )

st.sidebar.markdown("---")
st.sidebar.caption("University Academic Evaluation (UMS-Compliant)")
st.sidebar.caption("Authors: Karthik Reddy (Lead), Ananya Sharma, Rahul Verma")

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Track Defect Classifier",
    "📋 Maintenance Severity-Priority Queue",
    "📊 Model Interpretability & Diagnostics",
    "⚙️ System Telemetry & Architecture",
])

# ===========================================================================
# TAB 1: Real-Time Track Defect Classifier
# ===========================================================================
with tab1:
    st.subheader("Field Track Recording Car (TRC) & USFD Input Console")
    st.markdown("Enter track geometry parameters and ultrasonic flaw readings to obtain immediate defect severity classification.")

    if pipeline is None:
        st.error("Champion model pipeline not found! Please run `python src/train.py` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📍 Track Metadata & Logistics")
        zone_div = st.selectbox("Railway Zone / Division:", IR_ZONE_DIVISIONS, index=IR_ZONE_DIVISIONS.index(default_vals["zone"]) if default_vals["zone"] in IR_ZONE_DIVISIONS else 0)
        km_post = st.number_input("Track Km Post (Chainage):", min_value=0.0, max_value=2000.0, value=float(default_vals["km"]), step=0.1)
        rail_sec = st.selectbox("Rail Section Weight:", RAIL_SECTIONS, index=RAIL_SECTIONS.index(default_vals["rail"]))
        sleeper = st.selectbox("Sleeper Type:", SLEEPER_TYPES, index=SLEEPER_TYPES.index(default_vals["sleeper"]))
        traffic_gmt = st.slider("Route Traffic Density (GMT/annum):", min_value=5.0, max_value=120.0, value=float(default_vals["gmt"]), step=1.0)

    with col2:
        st.markdown("##### 📏 Track Geometry Recording")
        gauge_dev = st.slider("Gauge Deviation from 1676mm BG (mm):", min_value=-6.0, max_value=30.0, value=float(default_vals["gauge"]), step=0.1)
        align_err = st.slider("Lateral Alignment Error (mm):", min_value=0.1, max_value=25.0, value=float(default_vals["align"]), step=0.1)
        twist = st.slider("Cross-level Twist on 3.6m base (mm/m):", min_value=0.1, max_value=10.0, value=float(default_vals["twist"]), step=0.1)
        uneven = st.slider("Longitudinal Vertical Unevenness (mm):", min_value=0.5, max_value=26.0, value=float(default_vals["uneven"]), step=0.1)

    with col3:
        st.markdown("##### 🔬 Ultrasonic & Environmental Sensors")
        flaw_depth = st.slider("USFD Ultrasonic Flaw Depth (mm):", min_value=0.0, max_value=35.0, value=float(default_vals["flaw"]), step=0.5)
        axle_acc = st.slider("Axle-Box Peak Acceleration (g):", min_value=0.04, max_value=1.50, value=float(default_vals["accel"]), step=0.01)
        rail_temp = st.slider("Rail Temperature (°C):", min_value=-5.0, max_value=72.0, value=float(default_vals["temp"]), step=0.5)
        rain_7d = st.slider("7-Day Cumulative Rainfall (mm):", min_value=0.0, max_value=350.0, value=float(default_vals["rain"]), step=5.0)
        tamp_days = st.number_input("Days Since Last Tamping Machine Pass:", min_value=1, max_value=700, value=int(default_vals["tamp"]), step=5)

    st.markdown("---")

    # Predict Button
    if st.button("🚨 Classify Track Defect & Compute P-Way Priority", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "zone_division": zone_div,
            "track_km_post": km_post,
            "rail_section_type": rail_sec,
            "sleeper_type": sleeper,
            "traffic_density_gmt": traffic_gmt,
            "gauge_deviation_mm": gauge_dev,
            "alignment_error_mm": align_err,
            "twist_mm_per_m": twist,
            "unevenness_mm": uneven,
            "ultrasonic_flaw_depth_mm": flaw_depth,
            "axle_box_accel_g": axle_acc,
            "rail_temp_celsius": rail_temp,
            "rainfall_7d_mm": rain_7d,
            "days_since_last_tamping": tamp_days,
        }])

        start_time = time.time()
        pred_class = int(pipeline.predict(input_data)[0])
        pred_probs = pipeline.predict_proba(input_data)[0]
        latency_ms = (time.time() - start_time) * 1000

        pred_label = SEVERITY_MAPPING[pred_class]
        confidence = float(pred_probs[pred_class])
        priority_sc = compute_priority_score(pred_class, confidence, traffic_gmt, flaw_depth)
        tier_label = categorize_priority_tier(priority_sc)
        prescribed_action = IR_ACTION_MAPPING[pred_class]

        # Log into session audit log
        st.session_state.audit_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "km_post": km_post,
            "zone": zone_div,
            "prediction": pred_label,
            "confidence": f"{confidence*100:.1f}%",
            "priority_score": priority_sc,
            "latency_ms": f"{latency_ms:.1f}ms",
        })

        st.subheader("Classification Outcome & Operational P-Way Directive")

        badge_class = f"badge-{pred_label.lower()}"
        res_col1, res_col2, res_col3, res_col4 = st.columns([1.2, 1, 1, 1])

        with res_col1:
            st.markdown(f'<div class="{badge_class}">{pred_label.upper()} DEFECT</div>', unsafe_allow_html=True)
            st.caption(f"Inference Latency: **{latency_ms:.1f} ms**")

        with res_col2:
            st.metric("Model Confidence", f"{confidence * 100:.1f}%")

        with res_col3:
            st.metric("Maintenance Priority Score", f"{priority_sc} / 100")

        with res_col4:
            st.metric("Dispatch Urgency", tier_label.split(" (")[0])

        # Action Box
        st.markdown(f"""
        <div class="action-box">
            <b>Safety Directive (Indian Railways P-Way Manual):</b><br/>
            {prescribed_action}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Probability Distribution Across Severity Classes")
        prob_df = pd.DataFrame({
            "Severity Level": [SEVERITY_MAPPING[i] for i in range(4)],
            "Probability": pred_probs,
            "Color": [SEVERITY_COLORS[SEVERITY_MAPPING[i]] for i in range(4)],
        })
        fig = px.bar(
            prob_df,
            x="Severity Level",
            y="Probability",
            color="Severity Level",
            color_discrete_map=SEVERITY_COLORS,
            text=[f"{p*100:.1f}%" for p in pred_probs],
            range_y=[0.0, 1.05],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Domain Feature Breakdown
        st.markdown("#### Domain Interaction Engineering Breakdown")
        fe_engine = DomainFeatureEngineer()
        engineered_vals = fe_engine.transform(input_data).iloc[0]

        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        with ec1:
            st.metric("Track Quality Index (TQI)", f"{engineered_vals['track_quality_index']:.2f}")
        with ec2:
            st.metric("Thermal Stress Index", f"{engineered_vals['thermal_stress_index']:.2f}")
        with ec3:
            st.metric("Structural Risk Ratio", f"{engineered_vals['structural_risk_ratio']:.2f}")
        with ec4:
            st.metric("Vibration Dynamic Load", f"{engineered_vals['vibration_dynamic_load']:.2f}")
        with ec5:
            st.metric("Ballast Degradation", f"{engineered_vals['ballast_degradation_factor']:.2f}")


# ===========================================================================
# TAB 2: Maintenance Severity-Priority Queue
# ===========================================================================
with tab2:
    st.subheader("Batch Track Inspection Logging & Priority Dispatch Generator")
    st.markdown("Upload Division TRC logs or load sample inspection records to automatically rank and export field gang dispatches.")

    up_col1, up_col2 = st.columns([2, 1])

    with up_col1:
        uploaded_file = st.file_uploader("Upload Inspection CSV File:", type=["csv"])

    with up_col2:
        st.write("Or load realistic pre-calibrated sample data:")
        load_sample_btn = st.button("📁 Load Sample Division Inspections", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(batch_df)} track records from uploaded file.")
    elif load_sample_btn or "batch_data" in st.session_state:
        sample_path = PROJECT_ROOT / "app" / "assets" / "sample_inspections.csv"
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path)
            st.session_state["batch_data"] = batch_df
            st.info(f"Loaded {len(batch_df)} inspection observations from Indian Railways sample corridor.")
        else:
            # Load processed test set as fallback
            test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
            if test_path.exists():
                batch_df = pd.read_csv(test_path).head(30)
                st.session_state["batch_data"] = batch_df

    if batch_df is not None:
        # Run Batch Inference
        feature_cols = [c for c in batch_df.columns if c not in [TARGET_COL, "severity_name", ID_COL]]
        
        preds = pipeline.predict(batch_df[feature_cols])
        probs = pipeline.predict_proba(batch_df[feature_cols])

        dispatch_table = generate_maintenance_dispatch_table(batch_df, preds, probs)

        # High-Level Division Metrics
        crit_count = (dispatch_table["predicted_severity"] == "Critical").sum()
        high_count = (dispatch_table["predicted_severity"] == "High").sum()
        mod_count = (dispatch_table["predicted_severity"] == "Moderate").sum()
        low_count = (dispatch_table["predicted_severity"] == "Low").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Critical Line Blocks (2h)", crit_count, delta="Immediate Emergency" if crit_count > 0 else "Normal", delta_color="inverse")
        m2.metric("High Urgency (30 km/h)", high_count)
        m3.metric("Moderate Maintenance (15d)", mod_count)
        m4.metric("Normal / Routine Monitoring", low_count)

        st.markdown("---")
        # Filter controls
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            zone_filter = st.multiselect("Filter by Zone Division:", options=dispatch_table["zone_division"].unique(), default=dispatch_table["zone_division"].unique())
        with fcol2:
            severity_filter = st.multiselect("Filter by Predicted Severity:", options=["Critical", "High", "Moderate", "Low"], default=["Critical", "High", "Moderate", "Low"])
        with fcol3:
            min_score = st.slider("Minimum Priority Score:", 0.0, 100.0, 0.0)

        filtered_table = dispatch_table[
            (dispatch_table["zone_division"].isin(zone_filter)) &
            (dispatch_table["predicted_severity"].isin(severity_filter)) &
            (dispatch_table["priority_score"] >= min_score)
        ]

        st.markdown(f"##### Prioritized Maintenance Dispatch Queue ({len(filtered_table)} segments selected)")

        display_cols = [
            "dispatch_rank",
            "observation_id",
            "zone_division",
            "track_km_post",
            "predicted_severity",
            "priority_score",
            "prediction_confidence",
            "prescribed_action",
        ]
        
        # Color highlight function for dataframe
        def highlight_severity(row):
            sev = row["predicted_severity"]
            if sev == "Critical":
                return ["background-color: #f8d7da; color: #721c24; font-weight: bold;"] * len(row)
            elif sev == "High":
                return ["background-color: #ffe5d0; color: #a04000; font-weight: bold;"] * len(row)
            elif sev == "Moderate":
                return ["background-color: #fff3cd; color: #856404;"] * len(row)
            else:
                return ["background-color: #d4edda; color: #155724;"] * len(row)

        st.dataframe(
            filtered_table[display_cols].style.apply(highlight_severity, axis=1),
            use_container_width=True,
            height=380,
        )

        csv_data = filtered_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prioritized P-Way Dispatch List (CSV)",
            data=csv_data,
            file_name="indian_railways_pway_maintenance_dispatch.csv",
            mime="text/csv",
            type="primary",
        )


# ===========================================================================
# TAB 3: Model Interpretability & Diagnostics
# ===========================================================================
with tab3:
    st.subheader("Model Explainability, SHAP Feature Importance & Diagnostic Validation")
    st.markdown("Academic validation artifacts and SHAP attributions explaining which sensor thresholds drive critical derailment risk.")

    if metadata and "benchmark_comparison" in metadata:
        st.markdown("##### 🏆 Multiclass Model Benchmark Comparison (5-Fold Stratified CV)")
        bench_df = pd.DataFrame(metadata["benchmark_comparison"])
        st.dataframe(
            bench_df.rename(columns={
                "model_name": "Classifier Architecture",
                "cv_macro_f1_mean": "CV Macro-F1 (Mean)",
                "cv_macro_f1_std": "CV Macro-F1 (Std)",
                "cv_weighted_f1_mean": "CV Weighted-F1",
                "cv_accuracy_mean": "CV Accuracy",
                "training_time_sec": "Train Latency (s)",
            }),
            use_container_width=True,
        )

    plots_dir = PROJECT_ROOT / "models" / "evaluation_plots"

    st.markdown("---")
    st.markdown("##### 🔬 High-Resolution Evaluation Artifacts")

    plot_tabs = st.tabs(["Confusion Diagnostics", "Per-Class F1 Metrics", "ROC & PR Curves", "SHAP Feature Attribution", "Priority Distribution"])

    with plot_tabs[0]:
        cm_path = plots_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Raw Count & Normalized Recall Confusion Matrix (Test Partition)", use_container_width=True)

    with plot_tabs[1]:
        pcm_path = plots_dir / "per_class_metrics.png"
        if pcm_path.exists():
            st.image(str(pcm_path), caption="Precision, Recall, and F1-Score Across Severity Levels", use_container_width=True)

    with plot_tabs[2]:
        roc_path = plots_dir / "roc_pr_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Multiclass One-vs-Rest ROC & Precision-Recall Curves", use_container_width=True)

    with plot_tabs[3]:
        shap_path = plots_dir / "shap_summary.png"
        if shap_path.exists():
            st.image(str(shap_path), caption="SHAP Global Importance Explaining Critical Defect Predictions", use_container_width=True)
            st.markdown("""
            **Interpretability Takeaways:**
            - **Ultrasonic Flaw Depth (`ultrasonic_flaw_depth_mm`)**: Dominant driver for Critical and High severity predictions. Defect depths > 18mm trigger immediate removal.
            - **Gauge Deviation (`gauge_deviation_mm`)**: Spreading beyond +15mm shifts predictions directly into Critical due to wheel derailment hazard.
            - **Track Quality Index & Axle Acceleration**: Act as composite risk multipliers, amplifying severity on high-traffic (GMT) trunk routes.
            """)

    with plot_tabs[4]:
        pd_path = plots_dir / "priority_distribution.png"
        if pd_path.exists():
            st.image(str(pd_path), caption="Operational Maintenance Priority Score Distribution by Severity Tier", use_container_width=True)


# ===========================================================================
# TAB 4: System Telemetry & Architecture
# ===========================================================================
with tab4:
    st.subheader("System Telemetry, Latency Profiler & P-Way Architecture")

    # Interactive Latency Profiler
    st.markdown("##### ⏱️ Real-Time Inference Latency Profiler")
    bench_btn = st.button("Run 100-Sample Latency Profiling Benchmark")
    if bench_btn:
        test_sample = pd.DataFrame([{
            "zone_division": "NR-UMB", "track_km_post": 150.0, "rail_section_type": "60kg UIC",
            "sleeper_type": "PSC", "traffic_density_gmt": 40.0, "gauge_deviation_mm": 2.0,
            "alignment_error_mm": 1.5, "twist_mm_per_m": 0.8, "unevenness_mm": 3.5,
            "ultrasonic_flaw_depth_mm": 1.0, "axle_box_accel_g": 0.18, "rail_temp_celsius": 40.0,
            "rainfall_7d_mm": 20.0, "days_since_last_tamping": 60,
        }] * 100)

        latencies = []
        for i in range(100):
            t0 = time.perf_counter()
            _ = pipeline.predict(test_sample.iloc[[i]])
            latencies.append((time.perf_counter() - t0) * 1000)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg_lat = np.mean(latencies)

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Mean Latency", f"{avg_lat:.2f} ms")
        l2.metric("Median (P50)", f"{p50:.2f} ms")
        l3.metric("P95 Latency", f"{p95:.2f} ms")
        l4.metric("Throughput", f"{1000 / avg_lat:.0f} req/s")

    st.markdown("---")
    st.markdown("##### 🏛️ Indian Railways Track Safety System Architecture")
    st.code("""
+-------------------------------------------------------------------------------------------------+
|                                INDIAN RAILWAYS FIELD INSPECTION ASSETS                          |
|                                                                                                 |
|   +-----------------------+     +--------------------------+     +--------------------------+   |
|   | Track Recording Car   |     | USFD Flaw Detection Car  |     | Axle-Box Accelerometer   |   |
|   | (Gauge, Align, Twist) |     | (Internal Rail Cracks)   |     | (Dynamic Impact Forces)  |   |
|   +-----------+-----------+     +------------+-------------+     +------------+-------------+   |
+---------------|------------------------------|--------------------------------|-----------------+
                |                              |                                |
                v                              v                                v
+-------------------------------------------------------------------------------------------------+
|                             ZERO-LEAKAGE SCIKIT-LEARN PIPELINE                                  |
|                                                                                                 |
|   1. Physics-Informed Feature Engineering (TQI, Thermal Stress, Structural Risk, Dynamic Load)  |
|   2. Preprocessing ColumnTransformer (Median SimpleImputer, RobustScaler, OneHotEncoder)        |
|   3. Champion Classifier (LightGBM / Random Forest / XGBoost with Class-Weight Calibration)     |
+-------------------------------------------------------------------------------------------------+
                                               |
                                               v
+-------------------------------------------------------------------------------------------------+
|                       INDIAN RAILWAYS PERMANENT WAY (P-WAY) DISPATCH ENGINE                     |
|                                                                                                 |
|   +----------------------------+   +----------------------------+   +------------------------+  |
|   | Priority Score Calculation |   | Operational Dispatch Tiers |   | Prescribed IRPWM Action|  |
|   | (Severity x Prob x GMT)    |   | (Tier 1 to Tier 4)         |   | (Emergency Line Block) |  |
|   +----------------------------+   +----------------------------+   +------------------------+  |
+-------------------------------------------------------------------------------------------------+
    """, language="text")

    st.markdown("---")
    st.markdown("##### 📝 Session Prediction Audit Log")
    if st.session_state.audit_log:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(audit_df, use_container_width=True)
    else:
        st.info("No single-observation predictions made in this session yet. Run a prediction in Tab 1 to record audit logs.")
