"""
evaluate.py
Phase 5: Model Evaluation & Visualizations.
Generates 4 publication-quality evaluation figures and classification report on the unseen test set.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)

def run_evaluation_pipeline(processed_dir: str, figures_dir: str, reports_dir: str):
    """
    Evaluates the champion Deep Learning model on the strictly held-out test set:
    1. Plots Training & Validation Learning Curves (Loss & PR-AUC).
    2. Generates Confusion Matrix Heatmap (Raw counts and percentages).
    3. Plots Precision-Recall Curve with Average Precision benchmark.
    4. Plots ROC-AUC Curve with random baseline.
    5. Outputs full classification metrics report.
    """
    print("=" * 75)
    print("PHASE 5: MODEL EVALUATION & VISUALIZATION (TEST SET BENCHMARK)")
    print("=" * 75)

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Test Ground Truth and Model Probabilities
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    test_probs = np.load(os.path.join(processed_dir, "test_probs.npy"))
    test_preds = (test_probs >= 0.50).astype(int)

    # Load training history
    history_file = os.path.join(processed_dir, "best_model_history.json")
    with open(history_file, 'r') as f:
        history = json.load(f)

    # -------------------------------------------------------------------------
    # FIGURE 8: TRAINING & VALIDATION LEARNING CURVES
    # -------------------------------------------------------------------------
    print("\n[1] Generating Figure 8: Training & Validation Learning Curves...")
    epochs_range = range(1, len(history['loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(epochs_range, history['loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs_range, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    axes[0].set_title("Binary Cross-Entropy Loss vs Epochs", fontweight='bold')
    axes[0].set_xlabel("Epoch Number", fontweight='bold')
    axes[0].set_ylabel("Cross-Entropy Loss", fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # PR-AUC curves
    axes[1].plot(epochs_range, history['pr_auc'], 'b-o', label='Training PR-AUC', linewidth=2, markersize=4)
    axes[1].plot(epochs_range, history['val_pr_auc'], 'g-^', label='Validation PR-AUC', linewidth=2, markersize=4)
    axes[1].set_title("Precision-Recall AUC vs Epochs", fontweight='bold')
    axes[1].set_xlabel("Epoch Number", fontweight='bold')
    axes[1].set_ylabel("PR-AUC Score", fontweight='bold')
    axes[1].legend(loc='lower right', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Deep Learning Training Dynamics (Multi-Layer Perceptron)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    f8_path = os.path.join(figures_dir, "fig8_learning_curves.png")
    plt.savefig(f8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - Saved: {f8_path}")

    # -------------------------------------------------------------------------
    # FIGURE 9: CONFUSION MATRIX HEATMAP (TEST SET)
    # -------------------------------------------------------------------------
    print("\n[2] Generating Figure 9: Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    tn, fp, fn, tp = cm.ravel()
    print(f"    - True Negatives (Genuine identified as Genuine):  {tn:,} ({cm_norm[0,0]:.2f}%)")
    print(f"    - False Positives (Genuine flagged as Fraud):       {fp:,} ({cm_norm[0,1]:.2f}%)")
    print(f"    - False Negatives (Frauds missed):                  {fn:,} ({cm_norm[1,0]:.2f}%)")
    print(f"    - True Positives (Frauds successfully caught):      {tp:,} ({cm_norm[1,1]:.2f}%)")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    annot_matrix = np.array([
        [f"True Negative\n{tn:,}\n({cm_norm[0,0]:.2f}%)", f"False Positive\n{fp:,}\n({cm_norm[0,1]:.2f}%)"],
        [f"False Negative\n{fn:,}\n({cm_norm[1,0]:.2f}%)", f"True Positive\n{tp:,}\n({cm_norm[1,1]:.2f}%)"]
    ])
    
    sns.heatmap(cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=True, ax=ax,
                linewidths=1.5, linecolor='black', annot_kws={"size": 11, "fontweight": "bold"})
    ax.set_title("Test Set Confusion Matrix: Deep Learning Fraud Detection", fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Label", fontweight='bold', labelpad=10)
    ax.set_ylabel("Actual Ground Truth Label", fontweight='bold', labelpad=10)
    ax.set_xticklabels(["Genuine (Class 0)", "Fraudulent (Class 1)"], fontweight='bold')
    ax.set_yticklabels(["Genuine (Class 0)", "Fraudulent (Class 1)"], fontweight='bold')
    
    plt.tight_layout()
    f9_path = os.path.join(figures_dir, "fig9_confusion_matrix.png")
    plt.savefig(f9_path, dpi=300)
    plt.close()
    print(f"    - Saved: {f9_path}")

    # -------------------------------------------------------------------------
    # FIGURE 10: PRECISION-RECALL CURVE (TEST SET)
    # -------------------------------------------------------------------------
    print("\n[3] Generating Figure 10: Precision-Recall Curve...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs)
    ap_score = average_precision_score(y_test, test_probs)
    baseline_pr = (y_test == 1).mean()

    plt.figure(figsize=(8.5, 6))
    plt.plot(recalls, precisions, color='#2980b9', linewidth=2.5, label=f'Deep Learning MLP (AP = {ap_score:.4f})')
    plt.axhline(baseline_pr, color='red', linestyle='--', linewidth=1.8, label=f'No-Skill Baseline (Rate = {baseline_pr:.4f})')
    plt.title("Precision-Recall (PR) Curve on Held-Out Test Set", fontweight='bold', pad=12)
    plt.xlabel("Recall (Sensitivity: Frauds Detected / Total Frauds)", fontweight='bold')
    plt.ylabel("Precision (Positive Predictive Value: True Frauds / Flags)", fontweight='bold')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, fontsize=11)
    
    # Annotate operating point (threshold = 0.50)
    current_rec = tp / (tp + fn)
    current_prec = tp / (tp + fp)
    plt.plot(current_rec, current_prec, marker='o', markersize=9, color='#e74c3c')
    plt.annotate(f"Operating Point (Thr=0.50)\nRecall: {current_rec:.2%}\nPrecision: {current_prec:.2%}",
                 xy=(current_rec, current_prec),
                 xytext=(current_rec - 0.28, current_prec - 0.22),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#fff9c4", ec="#fbc02d"))

    plt.tight_layout()
    f10_path = os.path.join(figures_dir, "fig10_pr_curve.png")
    plt.savefig(f10_path, dpi=300)
    plt.close()
    print(f"    - Saved: {f10_path}")

    # -------------------------------------------------------------------------
    # FIGURE 11: RECEIVER OPERATING CHARACTERISTIC (ROC) CURVE
    # -------------------------------------------------------------------------
    print("\n[4] Generating Figure 11: Receiver Operating Characteristic (ROC) Curve...")
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_probs)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8.5, 6))
    plt.plot(fpr, tpr, color='#8e44ad', linewidth=2.5, label=f'Deep Learning MLP (ROC-AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guessing (AUC = 0.5000)')
    plt.title("Receiver Operating Characteristic (ROC) Curve on Test Set", fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight='bold')
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight='bold')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, fontsize=11)

    plt.tight_layout()
    f11_path = os.path.join(figures_dir, "fig11_roc_curve.png")
    plt.savefig(f11_path, dpi=300)
    plt.close()
    print(f"    - Saved: {f11_path}")

    # -------------------------------------------------------------------------
    # CLASSIFICATION REPORT & METRIC SUMMARY
    # -------------------------------------------------------------------------
    print("\n[5] Comprehensive Test Set Classification Report:")
    report_dict = classification_report(y_test, test_preds, target_names=["Genuine (0)", "Fraud (1)"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df.round(4).to_string())

    report_csv = os.path.join(reports_dir, "test_classification_report.csv")
    report_df.to_csv(report_csv)
    print(f"\n    - Saved classification report to: {report_csv}")

    print("=" * 75)
    print("PHASE 5 COMPLETE: ALL EVALUATION FIGURES GENERATED AND SAVED")
    print("=" * 75)

if __name__ == "__main__":
    p_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\data\processed"
    f_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\reports\figures"
    r_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\reports"
    run_evaluation_pipeline(p_dir, f_dir, r_dir)
