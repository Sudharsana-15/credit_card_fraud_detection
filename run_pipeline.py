"""
run_pipeline.py
Master execution script for the Credit Card Fraud Detection college project.
Runs all 5 phases sequentially with formatted console logs.
"""

import os
import sys
import time

def main():
    print("=" * 75)
    print("   CREDIT CARD FRAUD DETECTION: COMPLETE EXECUTION PIPELINE")
    print("   Course Outcome Rubrics: CO1, CO2, CO3, CO4, CO5")
    print("=" * 75)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")
    sys.path.append(src_dir)

    raw_csv = os.path.join(base_dir, "data", "raw", "creditcard.csv")
    cleaned_csv = os.path.join(base_dir, "data", "processed", "creditcard_cleaned.csv")
    processed_dir = os.path.join(base_dir, "data", "processed")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    reports_dir = os.path.join(base_dir, "reports")

    # Step 1: Ingestion, Duplicate Audit, Cleaning
    start_time = time.time()
    from data_loader import load_and_audit_dataset
    load_and_audit_dataset(raw_csv, cleaned_csv)

    # Step 2: EDA & Probability Distributions
    from eda_visualizer import run_eda_and_probability_analysis
    run_eda_and_probability_analysis(cleaned_csv, figures_dir)

    # Step 3: Preprocessing, Scaling, Zero-Leakage Splitting & SMOTE
    from preprocessor import run_preprocessing_pipeline
    run_preprocessing_pipeline(cleaned_csv, processed_dir, figures_dir)

    # Step 4: Deep Learning Modeling & Hyperparameter Tuning
    from tuner import run_hyperparameter_tuning
    run_hyperparameter_tuning(processed_dir, reports_dir)

    # Step 5: Evaluation on Held-Out Test Set & Visualizations
    from evaluate import run_evaluation_pipeline
    run_evaluation_pipeline(processed_dir, figures_dir, reports_dir)

    total_time = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"PIPELINE RUN COMPLETED SUCCESSFULLY in {total_time:.2f} seconds!")
    print(f"Figures Saved in:  {figures_dir}")
    print(f"Report Available:  {os.path.join(reports_dir, 'Project_Report.md')}")
    print(f"Champion Model:    {os.path.join(processed_dir, 'best_fraud_detector.keras')}")
    print("=" * 75)

if __name__ == "__main__":
    main()
