"""
tuner.py
Phase 4: Deep Learning Model Training & Hyperparameter Tuning.
Systematically tests multiple neural network architectures, learning rates,
and class imbalance handling strategies to identify the optimal model.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from model import build_fraud_mlp

def run_hyperparameter_tuning(processed_dir: str, reports_dir: str):
    """
    Executes systematic hyperparameter tuning across 4 distinct configurations:
    Evaluates each model strictly on the unmanipulated validation set using PR-AUC & F1-score.
    """
    print("=" * 75)
    print("PHASE 4: DEEP LEARNING MODELING & HYPERPARAMETER TUNING")
    print("=" * 75)

    # 1. Load Preprocessed Data
    print("\n[1] Loading Preprocessed Arrays:")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    X_train_smote = np.load(os.path.join(processed_dir, "X_train_smote.npy"))
    y_train_smote = np.load(os.path.join(processed_dir, "y_train_smote.npy"))

    meta = joblib.load(os.path.join(processed_dir, "preprocessing_metadata.joblib"))
    class_weights = meta['class_weight_dict']
    print(f"    - Training Shape:   {X_train.shape}")
    print(f"    - Validation Shape: {X_val.shape}")
    print(f"    - Test Shape:       {X_test.shape}")
    print(f"    - Class Weights:    {class_weights}")

    # 2. Hyperparameter Search Space
    configs = [
        {
            'name': 'Config 1: Baseline MLP (2 Layers)',
            'hidden_units': [64, 32],
            'dropout_rate': 0.20,
            'learning_rate': 0.001,
            'strategy': 'Class_Weights',
            'batch_size': 2048,
            'epochs': 25
        },
        {
            'name': 'Config 2: Deeper Regularized MLP (3 Layers)',
            'hidden_units': [64, 32, 16],
            'dropout_rate': 0.30,
            'learning_rate': 0.001,
            'strategy': 'Class_Weights',
            'batch_size': 2048,
            'epochs': 25
        },
        {
            'name': 'Config 3: High Capacity MLP (3 Wide Layers)',
            'hidden_units': [128, 64, 32],
            'dropout_rate': 0.30,
            'learning_rate': 0.0005,
            'strategy': 'Class_Weights',
            'batch_size': 2048,
            'epochs': 25
        },
        {
            'name': 'Config 4: SMOTE-Trained MLP (3 Layers)',
            'hidden_units': [64, 32, 16],
            'dropout_rate': 0.30,
            'learning_rate': 0.001,
            'strategy': 'SMOTE_Resampling',
            'batch_size': 2048,
            'epochs': 25
        }
    ]

    tuning_results = []
    best_val_prauc = -1.0
    best_config = None
    best_model = None
    best_history = None

    print("\n[2] Executing Systematic Hyperparameter Grid Search...")

    for idx, cfg in enumerate(configs, 1):
        print(f"\n--- Testing Candidate {idx}/{len(configs)}: {cfg['name']} ---")
        print(f"    Layers: {cfg['hidden_units']} | Dropout: {cfg['dropout_rate']} | LR: {cfg['learning_rate']} | Strategy: {cfg['strategy']}")

        # Set reproducible seeds
        tf.keras.utils.set_random_seed(42 + idx)

        model = build_fraud_mlp(
            input_dim=X_train.shape[1],
            hidden_units=cfg['hidden_units'],
            dropout_rate=cfg['dropout_rate'],
            learning_rate=cfg['learning_rate']
        )

        callbacks = [
            EarlyStopping(monitor='val_pr_auc', mode='max', patience=6, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=3, min_lr=1e-5, verbose=0)
        ]

        # Select training dataset and weights based on strategy
        if cfg['strategy'] == 'Class_Weights':
            train_X, train_y = X_train, y_train
            weights = class_weights
        else:
            train_X, train_y = X_train_smote, y_train_smote
            weights = None # SMOTE already balanced the samples

        history = model.fit(
            train_X, train_y,
            validation_data=(X_val, y_val),
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            class_weight=weights,
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate on unmanipulated validation set
        val_probs = model.predict(X_val, batch_size=4096, verbose=0).flatten()
        val_preds = (val_probs >= 0.50).astype(int)

        val_prauc = average_precision_score(y_val, val_probs)
        val_rocauc = roc_auc_score(y_val, val_probs)
        val_precision = precision_score(y_val, val_preds, zero_division=0)
        val_recall = recall_score(y_val, val_preds, zero_division=0)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)

        epochs_trained = len(history.history['loss'])
        print(f"    -> Results after {epochs_trained} epochs:")
        print(f"       Val PR-AUC:   {val_prauc:.4f}")
        print(f"       Val ROC-AUC:  {val_rocauc:.4f}")
        print(f"       Val Recall:   {val_recall:.4f} (Caught {int(val_recall * 71)}/71 frauds)")
        print(f"       Val Precision:{val_precision:.4f}")
        print(f"       Val F1-Score: {val_f1:.4f}")

        tuning_results.append({
            'Model Configuration': cfg['name'],
            'Architecture': str(cfg['hidden_units']),
            'Dropout': cfg['dropout_rate'],
            'Learning Rate': cfg['learning_rate'],
            'Strategy': cfg['strategy'],
            'Epochs Trained': epochs_trained,
            'Val PR-AUC': round(val_prauc, 4),
            'Val ROC-AUC': round(val_rocauc, 4),
            'Val Recall': round(val_recall, 4),
            'Val Precision': round(val_precision, 4),
            'Val F1-Score': round(val_f1, 4)
        })

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_config = cfg
            best_model = model
            best_history = history.history

    # Convert results to DataFrame
    results_df = pd.DataFrame(tuning_results)
    results_csv = os.path.join(reports_dir, "tuning_comparison_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n[3] Hyperparameter Tuning Comparison Table Saved to: {results_csv}")
    print("\n" + results_df.to_string(index=False))

    # 4. Save Winning Champion Model
    best_model_path = os.path.join(processed_dir, "best_fraud_detector.keras")
    best_model.save(best_model_path)
    print(f"\n[4] Best Performing Model: {best_config['name']} (Val PR-AUC: {best_val_prauc:.4f})")
    print(f"    - Saved champion model to: {best_model_path}")

    # Save training history for learning curve visualization in Phase 5
    history_path = os.path.join(processed_dir, "best_model_history.json")
    with open(history_path, 'w') as f:
        # Convert float32 values to standard float for JSON serialization
        serializable_history = {k: [float(val) for val in v] for k, v in best_history.items()}
        json.dump(serializable_history, f, indent=4)
    print(f"    - Saved training history to: {history_path}")

    # 5. Evaluate Champion Model on Unseen Test Set
    print("\n[5] Final Evaluation of Champion Model on Strictly Unseen Test Set:")
    test_probs = best_model.predict(X_test, batch_size=4096, verbose=0).flatten()
    np.save(os.path.join(processed_dir, "test_probs.npy"), test_probs)

    test_prauc = average_precision_score(y_test, test_probs)
    test_rocauc = roc_auc_score(y_test, test_probs)
    test_preds = (test_probs >= 0.50).astype(int)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    print(f"    - Test Set PR-AUC:    {test_prauc:.4f}")
    print(f"    - Test Set ROC-AUC:   {test_rocauc:.4f}")
    print(f"    - Test Set Recall:    {test_recall:.4f} (Caught {int(test_recall * 71)}/71 frauds)")
    print(f"    - Test Set Precision: {test_precision:.4f}")
    print(f"    - Test Set F1-Score:  {test_f1:.4f}")

    print("=" * 75)
    print("PHASE 4 COMPLETE: CHAMPION DEEP LEARNING MODEL TRAINED AND EVALUATED")
    print("=" * 75)

if __name__ == "__main__":
    p_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\data\processed"
    r_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\reports"
    run_hyperparameter_tuning(p_dir, r_dir)
