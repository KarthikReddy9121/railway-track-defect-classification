"""
src/evaluate.py - High-Resolution Diagnostic Plots, SHAP Interpretability & Maintenance Dispatch Export.

Loads the champion pipeline and held-out test data, computes multi-class diagnostics:
1. Dual Confusion Matrix (Raw counts & Row-Normalized Recall)
2. Per-Class Precision, Recall, and F1 bar charts
3. Multiclass One-vs-Rest ROC & Precision-Recall curves
4. SHAP feature attribution summary plot
5. Maintenance priority score distribution
6. Generates data/processed/maintenance_priority_list.csv for P-Way gangs.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for robust PNG export
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)
from sklearn.preprocessing import label_binarize
import shap

from src.utils import (
    LOGGER,
    SEVERITY_MAPPING,
    SEVERITY_COLORS,
    TARGET_COL,
    ID_COL,
    generate_maintenance_dispatch_table,
)
from src.feature_engineering import get_feature_names_from_preprocessor


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    """Exports publication-grade Raw and Normalized Confusion Matrix heatmaps."""
    class_names = [SEVERITY_MAPPING[i] for i in sorted(SEVERITY_MAPPING.keys())]
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw counts
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=True,
        linewidths=1.0,
        linecolor="#e0e0e0",
    )
    axes[0].set_title("Raw Count Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Predicted Severity Level", fontsize=11)
    axes[0].set_ylabel("True Ground Truth Severity", fontsize=11)

    # Normalized (Recall per class)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar=True,
        linewidths=1.0,
        linecolor="#e0e0e0",
    )
    axes[1].set_title("Normalized Confusion Matrix (Per-Class Recall)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Predicted Severity Level", fontsize=11)
    axes[1].set_ylabel("True Ground Truth Severity", fontsize=11)

    plt.suptitle("Indian Railways Track Defect Classification - Confusion Diagnostics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved confusion matrix plot: {output_path}")


def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    output_path: str,
) -> None:
    """Exports grouped bar chart showing Precision, Recall, and F1 across severity tiers."""
    class_names = [SEVERITY_MAPPING[i] for i in sorted(SEVERITY_MAPPING.keys())]
    
    precisions = [report_dict[cls]["precision"] for cls in class_names]
    recalls = [report_dict[cls]["recall"] for cls in class_names]
    f1s = [report_dict[cls]["f1-score"] for cls in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#3498db", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x, recalls, width, label="Recall", color="#2ecc71", edgecolor="black", linewidth=0.8)
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#e74c3c", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_title("Per-Class Performance Metrics Across Defect Severity Tiers", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.1)
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1, label="Target Benchmark (0.85)")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Attach value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved per-class metrics plot: {output_path}")


def plot_multiclass_roc_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str,
) -> None:
    """Exports One-vs-Rest Multiclass ROC and Precision-Recall Curves."""
    n_classes = len(SEVERITY_MAPPING)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]
    class_names = [SEVERITY_MAPPING[i] for i in range(n_classes)]

    # 1. ROC Curves
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2.2,
            label=f"{class_names[i]} (AUC = {roc_auc:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (0.50)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].set_title("Multiclass One-vs-Rest ROC Curves", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Precision-Recall Curves
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        axes[1].plot(
            recall,
            precision,
            color=colors[i],
            lw=2.2,
            label=f"{class_names[i]} (PR-AUC = {pr_auc:.3f})",
        )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Multiclass One-vs-Rest Precision-Recall Curves", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Discriminative Power Diagnostics across Defect Severities", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved ROC & PR curves plot: {output_path}")


def plot_priority_distribution(
    dispatch_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Exports distribution of Maintenance Priority Scores by predicted severity."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    palette = [SEVERITY_COLORS[SEVERITY_MAPPING[i]] for i in range(4)]

    # Boxplot
    sns.boxplot(
        data=dispatch_df,
        x="predicted_severity",
        y="priority_score",
        hue="predicted_severity",
        order=["Low", "Moderate", "High", "Critical"],
        palette=palette,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Priority Score Spread Across Predicted Severity", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Severity Level", fontsize=11)
    axes[0].set_ylabel("Maintenance Priority Score (0-100)", fontsize=11)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Count of priority tiers
    tier_counts = dispatch_df["priority_tier"].value_counts()
    axes[1].barh(tier_counts.index, tier_counts.values, color="#2c3e50", edgecolor="gold", linewidth=1.2)
    axes[1].set_title("Operational P-Way Dispatch Tiers Count", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Track Segments", fontsize=11)
    for i, v in enumerate(tier_counts.values):
        axes[1].text(v + 5, i, str(v), va="center", fontweight="bold", fontsize=10)
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)

    plt.suptitle("Indian Railways P-Way Maintenance Priority Allocation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved priority distribution plot: {output_path}")


def plot_shap_summary(
    pipeline: Any,
    X_test: pd.DataFrame,
    output_path: str,
    max_display: int = 12,
) -> None:
    """Computes and exports SHAP feature importance explaining model predictions."""
    try:
        LOGGER.info("Computing SHAP feature attributions on test sample...")
        fe_step = pipeline.named_steps["domain_fe"]
        prep_step = pipeline.named_steps["preprocessor"]
        clf_step = pipeline.named_steps["classifier"]

        # Transform features up to classifier
        X_fe = fe_step.transform(X_test.head(300))
        X_trans = prep_step.transform(X_fe)
        feature_names = get_feature_names_from_preprocessor(prep_step)

        # Build SHAP TreeExplainer
        explainer = shap.TreeExplainer(clf_step)
        shap_values = explainer.shap_values(X_trans)

        plt.figure(figsize=(11, 7))
        # If multiclass shap_values is a list of arrays (one per class) or 3D array
        if isinstance(shap_values, list):
            # Focus on Critical severity class (index 3)
            crit_shap = shap_values[3] if len(shap_values) > 3 else shap_values[-1]
            shap.summary_plot(
                crit_shap,
                X_trans,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
                plot_type="dot",
            )
        elif len(shap_values.shape) == 3:
            shap.summary_plot(
                shap_values[:, :, 3],
                X_trans,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
                plot_type="dot",
            )
        else:
            shap.summary_plot(
                shap_values,
                X_trans,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
                plot_type="dot",
            )

        plt.title("SHAP Feature Attribution: Critical Track Defect Drivers", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        LOGGER.info(f"Saved SHAP summary plot: {output_path}")
    except Exception as e:
        LOGGER.warning(f"SHAP explanation generation fallback: {e}")
        # Generate feature importance bar chart directly from classifier as robust fallback
        try:
            clf_step = pipeline.named_steps["classifier"]
            prep_step = pipeline.named_steps["preprocessor"]
            feature_names = get_feature_names_from_preprocessor(prep_step)
            importances = clf_step.feature_importances_
            
            top_idx = np.argsort(importances)[::-1][:max_display]
            plt.figure(figsize=(10, 6))
            plt.barh(
                [feature_names[i] for i in top_idx][::-1],
                importances[top_idx][::-1],
                color="#2980b9",
                edgecolor="black",
            )
            plt.title("Tree Feature Importance (Gini / Split Gain)", fontsize=13, fontweight="bold")
            plt.xlabel("Importance Score", fontsize=11)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            LOGGER.info(f"Saved feature importance fallback plot: {output_path}")
        except Exception as ex2:
            LOGGER.error(f"Failed to export feature importance: {ex2}")


def run_evaluation(
    model_path: str = "models/champion_track_defect_pipeline.joblib",
    test_path: str = "data/processed/test.csv",
    output_plots_dir: str = "models/evaluation_plots",
    priority_list_path: str = "data/processed/maintenance_priority_list.csv",
) -> Dict[str, Any]:
    """Orchestrates comprehensive model evaluation, plot exports, and dispatch table generation."""
    plots_dir = Path(output_plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading champion pipeline from: {model_path}")
    pipeline = joblib.load(model_path)

    LOGGER.info(f"Loading test dataset from: {test_path}")
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in test_df.columns if c not in [TARGET_COL, "severity_name", ID_COL]]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    class_names = [SEVERITY_MAPPING[i] for i in sorted(SEVERITY_MAPPING.keys())]
    rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # 1. Export Diagnostic Plots
    plot_confusion_matrices(y_test, y_pred, str(plots_dir / "confusion_matrix.png"))
    plot_per_class_metrics(rep, str(plots_dir / "per_class_metrics.png"))
    plot_multiclass_roc_pr(y_test, y_proba, str(plots_dir / "roc_pr_curves.png"))
    plot_shap_summary(pipeline, X_test, str(plots_dir / "shap_summary.png"))

    # 2. Generate Maintenance Dispatch Table
    dispatch_df = generate_maintenance_dispatch_table(test_df, y_pred, y_proba)
    dispatch_df.to_csv(priority_list_path, index=False)
    LOGGER.info(f"Exported P-Way Maintenance Priority Dispatch Table: {Path(priority_list_path).resolve()}")

    # 3. Export Priority Distribution Plot
    plot_priority_distribution(dispatch_df, str(plots_dir / "priority_distribution.png"))

    LOGGER.info("=== EVALUATION COMPLETED SUCCESSFULLY ===")
    LOGGER.info(f"Test Set Macro-F1: {macro_f1:.4f}")
    LOGGER.info(f"Test Set Weighted-F1: {weighted_f1:.4f}")
    LOGGER.info(f"Top 5 P-Way Emergency Dispatches:\n" + dispatch_df[[
        "dispatch_rank", "observation_id", "zone_division", "track_km_post",
        "predicted_severity", "priority_score", "prescribed_action"
    ]].head(5).to_string(index=False))

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "dispatch_records": len(dispatch_df),
    }


if __name__ == "__main__":
    run_evaluation()
