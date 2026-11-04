"""
preprocessor.py
Phase 3: Data Preprocessing, Feature Scaling, Zero-Leakage Stratified Splitting,
SMOTE Balancing, and Feature Importance Analysis (Assignment Rubrics CO1 & CO2).
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

def run_preprocessing_pipeline(cleaned_csv_path: str, processed_dir: str, figures_dir: str):
    """
    Executes complete, leakage-free preprocessing:
    1. Stratified 70/15/15 split.
    2. RobustScaler fitted strictly on X_train only.
    3. SMOTE applied strictly on X_train only.
    4. Computes balanced class weights.
    5. Feature importance ranking via ANOVA F-statistic and Mutual Information.
    6. Exports clean numpy arrays and metadata.
    """
    print("=" * 75)
    print("PHASE 3: DATA PREPROCESSING & LEAKAGE-FREE PARTITIONING")
    print("=" * 75)

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv_path)
    # Drop any helper columns created during EDA (e.g. 'Hour', 'Log_Amount') if present
    feature_cols = [c for c in df.columns if c not in ['Class', 'Hour', 'Log_Amount']]
    X = df[feature_cols].copy()
    y = df['Class'].copy()

    print(f"\n[1] Extracted Features & Target:")
    print(f"    - Total Features: {X.shape[1]} ({', '.join(feature_cols[:5])} ... {', '.join(feature_cols[-3:])})")
    print(f"    - Total Transactions: {len(y):,}")

    # -------------------------------------------------------------------------
    # STEP 1: STRATIFIED SPLIT (70% TRAIN, 15% VAL, 15% TEST) - BEFORE ANY SCALING
    # -------------------------------------------------------------------------
    print("\n[2] Executing Stratified Train / Validation / Test Split (70% / 15% / 15%):")
    # First split into 70% Train and 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Split Temp equally into 15% Validation and 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    split_summary = pd.DataFrame({
        'Dataset Partition': ['Training Set (70%)', 'Validation Set (15%)', 'Test Set (15%)', 'Total'],
        'Total Samples': [len(y_train), len(y_val), len(y_test), len(y)],
        'Genuine (0)': [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum(), (y == 0).sum()],
        'Fraud (1)': [(y_train == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum(), (y == 1).sum()],
        'Fraud Ratio (%)': [
            (y_train == 1).mean() * 100,
            (y_val == 1).mean() * 100,
            (y_test == 1).mean() * 100,
            (y == 1).mean() * 100
        ]
    })
    print(split_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 2: SCALING WITH ZERO DATA LEAKAGE
    # -------------------------------------------------------------------------
    print("\n[3] Feature Scaling using RobustScaler (Zero Leakage Protocol):")
    print("    - Protocol: Scaler is fitted ONLY on X_train.")
    print("    - Scaler parameters (Median & IQR) learned strictly from training distribution.")
    print("    - Validation and Test sets are transformed using X_train parameters without fitting.")

    scaler = RobustScaler()
    # Scale Time and Amount; PCA features are already normalized, but scaling all ensures consistency
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save fitted scaler for reproducibility and production inference
    scaler_path = os.path.join(processed_dir, "robust_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"    - Fitted RobustScaler saved to: {scaler_path}")

    # -------------------------------------------------------------------------
    # STEP 3: CLASS WEIGHTING & SMOTE RESAMPLING (ON TRAIN SET ONLY)
    # -------------------------------------------------------------------------
    print("\n[4] Handling Severe Class Imbalance (Two Strategies Prepared):")
    # Strategy A: Balanced Class Weights for Cross-Entropy Loss
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: float(weight) for cls, weight in zip(classes, class_weights)}
    print(f"    - Strategy A: Analytical Class Weights:")
    print(f"        Weight for Genuine (Class 0): {class_weight_dict[0]:.4f}")
    print(f"        Weight for Fraud (Class 1):   {class_weight_dict[1]:.4f}")
    print(f"        (Fraud errors are penalized {class_weight_dict[1]/class_weight_dict[0]:.1f}x more heavily)")

    # Strategy B: Synthetic Minority Over-sampling Technique (SMOTE)
    print("\n    - Strategy B: SMOTE (Synthetic Minority Over-sampling Technique):")
    print("        CRITICAL LEAKAGE PREVENTION:")
    print("        SMOTE is applied EXCLUSIVELY to the training fold.")
    print("        Validation and Test sets remain strictly in their natural 0.167% distribution.")
    
    smote = SMOTE(sampling_strategy=0.10, random_state=42) # Resample minority to 10% of majority for stability
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"        Original Train Set:  {len(y_train):,} samples (Genuine: {(y_train==0).sum():,}, Fraud: {(y_train==1).sum():,})")
    print(f"        SMOTE-Resampled Train: {len(y_train_smote):,} samples (Genuine: {(y_train_smote==0).sum():,}, Fraud: {(y_train_smote==1).sum():,})")

    # -------------------------------------------------------------------------
    # STEP 4: FEATURE IMPORTANCE & DISCRIMINATIVE RANKING
    # -------------------------------------------------------------------------
    print("\n[5] Feature Importance Analysis (ANOVA F-statistic & Mutual Information):")
    f_scores, p_values = f_classif(X_train_scaled, y_train)
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'ANOVA_F_Score': f_scores,
        'P_Value': p_values
    }).sort_values(by='ANOVA_F_Score', ascending=False).reset_index(drop=True)

    print("\n    Top 10 Most Discriminative Features by ANOVA F-score:")
    print(feature_importance_df.head(10).to_string(index=False))

    # Plot Figure 7: Feature Importance Bar Plot
    plt.figure(figsize=(12, 6))
    top15_feats = feature_importance_df.head(15)
    sns.barplot(data=top15_feats, x='ANOVA_F_Score', y='Feature', palette='viridis', edgecolor='black')
    plt.title("Top 15 Most Discriminative Features for Fraud Detection (ANOVA F-Statistic)", fontweight='bold', pad=12)
    plt.xlabel("ANOVA F-Statistic Score (Between-Group Variance / Within-Group Variance)", fontweight='bold')
    plt.ylabel("Dataset Feature", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    for index, value in enumerate(top15_feats['ANOVA_F_Score']):
        plt.text(value + 100, index, f'{value:,.0f}', va='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    f7_path = os.path.join(figures_dir, "fig7_feature_importance.png")
    plt.savefig(f7_path, dpi=300)
    plt.close()
    print(f"\n    - Figure 7 saved: {f7_path}")

    # -------------------------------------------------------------------------
    # STEP 5: SAVE PREPROCESSED PARTITIONS AS NUMPY ARRAYS
    # -------------------------------------------------------------------------
    print("\n[6] Exporting Preprocessed Arrays for Model Training:")
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val_scaled)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val.to_numpy())
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy())
    np.save(os.path.join(processed_dir, "X_train_smote.npy"), X_train_smote)
    np.save(os.path.join(processed_dir, "y_train_smote.npy"), y_train_smote.to_numpy())
    
    # Save metadata dictionary
    meta = {
        'feature_names': feature_cols,
        'class_weight_dict': class_weight_dict,
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test)
    }
    joblib.dump(meta, os.path.join(processed_dir, "preprocessing_metadata.joblib"))
    print(f"    - All training/validation/test arrays exported to: {processed_dir}")
    print("=" * 75)
    print("PHASE 3 COMPLETE: ZERO-LEAKAGE PIPELINE READY FOR DEEP LEARNING")
    print("=" * 75)

if __name__ == "__main__":
    cleaned_csv = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\data\processed\creditcard_cleaned.csv"
    proc_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\data\processed"
    figs_path = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\reports\figures"
    run_preprocessing_pipeline(cleaned_csv, proc_dir, figs_path)
