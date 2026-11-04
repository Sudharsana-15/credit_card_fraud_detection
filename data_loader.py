"""
data_loader.py
Phase 1: Dataset Inspection, Duplicate Audit, and Data Cleaning.
Academic Project: Credit Card Fraud Detection Using EDA and Deep Learning
"""

import os
import pandas as pd
import numpy as np

def load_and_audit_dataset(raw_path: str, output_path: str):
    """
    Loads raw credit card dataset, performs strict data audit,
    inspects duplicate records (distinguishing genuine vs fraud),
    analyzes outliers, and outputs the cleaned dataset.
    """
    print("=" * 70)
    print("PHASE 1: DATA AUDIT, DUPLICATE ANALYSIS & DATA CLEANING")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1] Loading raw dataset from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    initial_rows, initial_cols = df.shape
    print(f"    - Initial Dimensions: {initial_rows:,} rows x {initial_cols} columns")

    # 2. Missing Value Check
    print("\n[2] Checking for Missing / Null Values:")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"    - Total Null Values across all columns: {total_nulls}")
    if total_nulls == 0:
        print("    - Result: Clean! No imputation required.")
    else:
        print(f"    - Columns with nulls:\n{null_counts[null_counts > 0]}")

    # 3. Data Types Check
    print("\n[3] Feature Data Types:")
    dtype_counts = df.dtypes.value_counts().to_dict()
    for dtype, count in dtype_counts.items():
        print(f"    - {dtype}: {count} features")

    # 4. In-Depth Duplicate Inspection
    print("\n[4] In-Depth Duplicate Analysis:")
    # boolean mask of duplicates (all occurrences)
    dup_all_mask = df.duplicated(keep=False)
    num_total_dups_all = dup_all_mask.sum()
    
    # boolean mask of duplicates (excluding first occurrence)
    dup_redundant_mask = df.duplicated(keep='first')
    num_redundant_dups = dup_redundant_mask.sum()
    
    print(f"    - Total rows involved in duplicate clusters: {num_total_dups_all:,}")
    print(f"    - Redundant rows to be dropped (keep='first'): {num_redundant_dups:,}")

    # Inspect duplicate breakdown by class
    dups_df = df[dup_all_mask]
    dup_class_counts = dups_df['Class'].value_counts()
    redundant_class_counts = df[dup_redundant_mask]['Class'].value_counts()

    print("\n    Breakdown of Redundant Duplicate Rows by Class:")
    for cls in [0, 1]:
        cnt = redundant_class_counts.get(cls, 0)
        cls_name = "Genuine (0)" if cls == 0 else "Fraudulent (1)"
        print(f"      * {cls_name}: {cnt} redundant records")

    # Inspect sample duplicate fraud transactions
    dup_frauds = dups_df[dups_df['Class'] == 1].sort_values(by=['Time', 'Amount'])
    print(f"\n    Duplicate Fraud Sample Inspection (Total duplicate fraud records: {len(dup_frauds)}):")
    if len(dup_frauds) > 0:
        print(dup_frauds[['Time', 'Amount', 'Class', 'V1', 'V2', 'V3']].head(6).to_string())

    print("\n    Duplicate Audit Justification for Removal:")
    print("      - Keeping duplicate rows causes DATA LEAKAGE: an exact duplicate transaction")
    print("        could end up in both the training set and the test set, allowing the deep learning")
    print("        model to score artificial accuracy through memorization rather than generalization.")
    print("      - Removing redundant duplicates ensures strict statistical integrity.")

    # Drop duplicates
    df_cleaned = df.drop_duplicates(keep='first').copy()
    cleaned_rows, cleaned_cols = df_cleaned.shape
    print(f"\n    - Post-Cleaning Dimensions: {cleaned_rows:,} rows x {cleaned_cols} columns")
    print(f"    - Removed exactly {initial_rows - cleaned_rows:,} duplicate records.")

    # 5. Class Distribution Post-Cleaning
    print("\n[5] Class Distribution Post-Cleaning:")
    class_counts = df_cleaned['Class'].value_counts()
    class_pcts = df_cleaned['Class'].value_counts(normalize=True) * 100
    for cls in [0, 1]:
        cls_name = "Genuine (0)" if cls == 0 else "Fraudulent (1)"
        print(f"    - {cls_name}: {class_counts[cls]:,} transactions ({class_pcts[cls]:.4f}%)")
    print(f"    - Imbalance Ratio: 1 Fraud per every {class_counts[0] // class_counts[1]} Genuine transactions.")

    # 6. Statistical Summary & Skewness Analysis
    print("\n[6] Statistical Summary (Selected Key Features):")
    stats_features = ['Amount', 'Time', 'V1', 'V2', 'V3', 'V14', 'V17']
    summary_table = pd.DataFrame({
        'Mean': df_cleaned[stats_features].mean(),
        'Median': df_cleaned[stats_features].median(),
        'Std Dev': df_cleaned[stats_features].std(),
        'Variance': df_cleaned[stats_features].var(),
        'IQR': df_cleaned[stats_features].quantile(0.75) - df_cleaned[stats_features].quantile(0.25),
        'Skewness': df_cleaned[stats_features].skew(),
        'Kurtosis': df_cleaned[stats_features].kurtosis()
    })
    print(summary_table.round(4).to_string())

    # Amount breakdown by class
    print("\n[7] Transaction Amount Comparison by Class:")
    amt_genuine = df_cleaned[df_cleaned['Class'] == 0]['Amount']
    amt_fraud = df_cleaned[df_cleaned['Class'] == 1]['Amount']
    amt_comp = pd.DataFrame({
        'Genuine (Class=0)': [
            amt_genuine.count(), amt_genuine.mean(), amt_genuine.median(),
            amt_genuine.std(), amt_genuine.min(), amt_genuine.max(), amt_genuine.skew()
        ],
        'Fraud (Class=1)': [
            amt_fraud.count(), amt_fraud.mean(), amt_fraud.median(),
            amt_fraud.std(), amt_fraud.min(), amt_fraud.max(), amt_fraud.skew()
        ]
    }, index=['Count', 'Mean ($)', 'Median ($)', 'Std Dev ($)', 'Min ($)', 'Max ($)', 'Skewness'])
    print(amt_comp.round(2).to_string())

    # Outlier Analysis for Amount
    print("\n[8] Outlier Analysis on Transaction Amount:")
    q1 = df_cleaned['Amount'].quantile(0.25)
    q3 = df_cleaned['Amount'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers_count = (df_cleaned['Amount'] > upper_bound).sum()
    outliers_fraud = ((df_cleaned['Amount'] > upper_bound) & (df_cleaned['Class'] == 1)).sum()
    print(f"    - Amount Q1: ${q1:.2f}, Median: ${df_cleaned['Amount'].median():.2f}, Q3: ${q3:.2f}, IQR: ${iqr:.2f}")
    print(f"    - Tukey Upper Outlier Threshold (Q3 + 1.5*IQR): ${upper_bound:.2f}")
    print(f"    - Transactions exceeding threshold: {outliers_count:,} ({outliers_count/cleaned_rows*100:.2f}%)")
    print(f"    - Fraudulent transactions exceeding threshold: {outliers_fraud:,} ({outliers_fraud/class_counts[1]*100:.2f}% of all frauds)")
    print("    - Crucial Data Science Takeaway: We must NOT delete these high-amount rows because")
    print("      frauds frequently occur in these outlier amounts. Instead, we use ROBUST SCALING.")

    # 8. Save Processed Dataset
    print(f"\n[9] Saving Cleaned Dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    print(f"    - Saved successfully ({os.path.getsize(output_path) / (1024*1024):.2f} MB).")
    print("=" * 70)
    print("PHASE 1 COMPLETE: DATASET READY FOR EDA & PROBABILITY ANALYSIS")
    print("=" * 70)

if __name__ == "__main__":
    raw_csv = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\data\raw\creditcard.csv"
    proc_csv = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\data\processed\creditcard_cleaned.csv"
    load_and_audit_dataset(raw_csv, proc_csv)
