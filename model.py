"""
model.py
Deep Learning Model Architecture for Credit Card Fraud Detection.
Multi-Layer Perceptron (MLP) with BatchNormalization and Dropout.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

def build_fraud_mlp(
    input_dim: int = 30,
    hidden_units: list = [64, 32, 16],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> models.Sequential:
    """
    Constructs a Deep Neural Network (MLP) tailored for imbalanced fraud classification.
    
    Architecture:
    - Input: 30 standardized features
    - Dense layers with ReLU activations
    - BatchNormalization to stabilize gradients & accelerate convergence
    - Dropout to prevent co-adaptation and overfitting on the majority class
    - Output layer: 1 neuron with Sigmoid activation (calibrated fraud probability)
    """
    model = models.Sequential(name="Fraud_Detection_MLP")
    
    # Input Layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden Layers
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, use_bias=False, name=f"dense_{i+1}"))
        model.add(layers.BatchNormalization(name=f"batch_norm_{i+1}"))
        model.add(layers.Activation('relu', name=f"relu_{i+1}"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))
            
    # Final Output Layer: Binary Probability
    model.add(layers.Dense(1, activation='sigmoid', name="output_fraud_prob"))
    
    # Compile with Adam optimizer and PR-AUC/ROC-AUC evaluation metrics
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(curve='PR', name='pr_auc'),
            metrics.AUC(curve='ROC', name='roc_auc')
        ]
    )
    
    return model

if __name__ == "__main__":
    test_model = build_fraud_mlp()
    test_model.summary()
