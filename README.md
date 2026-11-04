# Credit Card Fraud Detection Using Exploratory Data Analysis and Deep Learning

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end college data science and deep learning project that identifies fraudulent credit card transactions from highly skewed real-world transaction streams. The project applies **Ben Fry's Seven Stages of Data Visualization**, rigorously tests **discrete and continuous probability distributions (College Rubric CO4)**, implements a **zero-leakage preprocessing pipeline**, and trains a tuned **Deep Multi-Layer Perceptron (MLP)** using TensorFlow/Keras.

---

## 📌 Project Highlights & Real Test Results

All results shown below reflect actual execution on 42,559 unseen test transactions:

| Metric | Evaluated Score | Interpretation & Significance |
| :--- | :---: | :--- |
| **Test PR-AUC (Average Precision)** | **0.7371** | Over **430x** higher than the random baseline ($0.0017$). Primary metric for imbalanced data. |
| **Test ROC-AUC** | **0.9558** | Near-perfect separation between genuine and fraudulent score distributions. |
| **Fraud Recall (Sensitivity)** | **80.28%** | Caught **57 out of 71** fraudulent transactions on the unseen test set. |
| **Precision** | **62.64%** | Only **34 false positives** generated across 42,488 genuine transactions ($0.08\%$ false alarm rate). |
| **Overall F1-Score** | **0.7037** | Balanced harmonic mean of precision and recall. |
| **Overall Accuracy** | **99.89%** | Correct classification rate across all test transactions. |

---

## 🎓 College Assignment Rubric & Course Outcomes (CO) Mapping

| Requirement | Course Outcome | Description & Implementation in Repo |
| :--- | :--- | :--- |
| **1. Dataset with ≥15 features** | **CO1** | Kaggle ULB dataset containing **31 features** (Time, V1–V28 PCA vectors, Amount, Class). |
| **2. Exploratory Data Analysis (EDA)** | **CO2** | Skewness, kurtosis, correlation heatmaps, hourly volume trends, and feature density curves. |
| **3. Missing, Duplicate & Outlier Handling** | **CO1** | Verified 0 missing entries; audited and dropped **1,081 duplicate transactions**; Tukey IQR outlier analysis. |
| **4. Preprocessing & Leakage Prevention** | **CO1** | Stratified 70/15/15 train/val/test split; `RobustScaler` and `SMOTE` fitted **strictly on the training set only**. |
| **5. Statistical & Distribution Analysis** | **CO4** | Discrete **Bernoulli** ($p = 0.00167$) and **Poisson** ($\lambda = 10.28$/hr); Continuous **Log-Normal** and **Gaussian Q-Q**. |
| **6. ≥5 Data Visualizations** | **CO2** | **7 exploratory figures** mapping Ben Fry's 7 stages of visualization (`reports/figures/fig1` to `fig7`). |
| **7. Deep Learning Architecture** | **CO3** | Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout (0.30), and Sigmoid probability output. |
| **8. Hyperparameter Tuning** | **CO3** | Systematic grid search across 4 architectures, learning rates, and balancing strategies (`Class_Weights` vs `SMOTE`). |
| **9. 3–4 Model Evaluation Plots** | **CO5** | **4 evaluation figures**: Learning curves, Confusion Matrix, Precision-Recall curve, and ROC curve (`fig8` to `fig11`). |
| **10. Conclusion, Future Scope & References** | **CO5** | Complete academic report with deployment architecture, limitations, future research, and IEEE citations. |
| **11. Organized GitHub Repository** | **Software Standards** | Clean modular folder structure, requirements, interactive notebooks, and full project report. |

---

## 📁 Repository Directory Structure

```
credit_card_fraud_detection/
├── data/
│   ├── raw/                              # Original Kaggle dataset (creditcard.csv)
│   └── processed/                        # Cleaned CSV, fitted scaler, and numpy arrays
│       ├── creditcard_cleaned.csv        # Cleaned dataset without duplicates (283,726 rows)
│       ├── robust_scaler.joblib          # Fitted RobustScaler (fitted ONLY on training fold)
│       ├── X_train.npy, y_train.npy      # 70% Training partition (198,608 samples)
│       ├── X_val.npy, y_val.npy          # 15% Validation partition (42,559 samples)
│       ├── X_test.npy, y_test.npy        # 15% Unseen test partition (42,559 samples)
│       └── best_fraud_detector.keras     # Trained champion Deep Learning model weights
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb    # Interactive notebook for EDA, cleaning & probability
│   └── 02_deep_learning_and_evaluation.ipynb # Interactive notebook for Deep Learning & evaluation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Phase 1: Ingestion, duplicate audit & outlier inspection
│   ├── eda_visualizer.py                 # Phase 2: Ben Fry's 7 stages & CO4 probability analysis
│   ├── preprocessor.py                   # Phase 3: Zero-leakage scaling, SMOTE & ANOVA importance
│   ├── model.py                          # Phase 4: Keras Multi-Layer Perceptron architecture
│   ├── tuner.py                          # Phase 4: Hyperparameter grid search & model training
│   └── evaluate.py                       # Phase 5: Test set evaluation & publication figures
├── reports/
│   ├── figures/                          # 11 high-resolution figures (PNG 300 DPI)
│   │   ├── fig1_class_distribution.png
│   │   ├── fig2_amount_distribution.png
│   │   ├── fig3_temporal_trends.png
│   │   ├── fig4_correlation_heatmap.png
│   │   ├── fig5_discriminative_pca_distributions.png
│   │   ├── fig6_probability_distributions_co4.png
│   │   ├── fig7_feature_importance.png
│   │   ├── fig8_learning_curves.png
│   │   ├── fig9_confusion_matrix.png
│   │   ├── fig10_pr_curve.png
│   │   └── fig11_roc_curve.png
│   ├── tuning_comparison_results.csv     # Hyperparameter search comparison table
│   ├── test_classification_report.csv    # Final test classification metrics
│   └── Project_Report.md                 # Full academic project report (IEEE format)
├── requirements.txt                      # Project dependencies
└── README.md                             # Repository documentation
```

---

## 📊 Visualizations Overview

All figures are automatically generated and saved in `reports/figures/`:

1. **Figure 1 (`fig1_class_distribution.png`):** Dual linear and log-scale frequency plots showing the 0.167% class imbalance.
2. **Figure 2 (`fig2_amount_distribution.png`):** Continuous KDE density of $\ln(\text{Amount}+1)$ and boxplots comparing genuine vs fraud amounts.
3. **Figure 3 (`fig3_temporal_trends.png`):** 48-hour cyclical volume trends showing nocturnal fraud spikes versus daylight genuine traffic.
4. **Figure 4 (`fig4_correlation_heatmap.png`):** Lower triangular correlation matrix of top PCA components with target `Class`.
5. **Figure 5 (`fig5_discriminative_pca_distributions.png`):** Probability densities of top 6 discriminative features (`V14`, `V17`, `V12`, `V10`, `V11`, `V4`).
6. **Figure 6 (`fig6_probability_distributions_co4.png`):** Discrete Bernoulli PMF, Poisson hourly fraud fit, continuous amount CDF, and normal Q-Q plot for `V14`.
7. **Figure 7 (`fig7_feature_importance.png`):** Top 15 features ranked by ANOVA F-statistic ($F > 20,000$ for `V17` and `V14`).
8. **Figure 8 (`fig8_learning_curves.png`):** Binary cross-entropy loss and PR-AUC training vs validation curves over 25 epochs.
9. **Figure 9 (`fig9_confusion_matrix.png`):** Test confusion matrix showing 57 True Positives and only 34 False Positives.
10. **Figure 10 (`fig10_pr_curve.png`):** Precision-Recall curve with operating threshold marker ($AP = 0.7371$).
11. **Figure 11 (`fig11_roc_curve.png`):** Receiver Operating Characteristic curve ($ROC\text{-}AUC = 0.9558$).

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/creditcard.csv`.

---

## ⚡ Execution Pipeline (Phase by Phase)

You can execute each phase independently or run through the pipeline:

```bash
# Phase 1: Ingest, audit duplicates & clean dataset
python src/data_loader.py

# Phase 2: Run EDA, statistical tests & generate Figures 1-6
python src/eda_visualizer.py

# Phase 3: Zero-leakage scaling, stratified splitting & SMOTE (Figure 7)
python src/preprocessor.py

# Phase 4: Systematic hyperparameter tuning & train Deep Learning model
python src/tuner.py

# Phase 5: Evaluate model on unseen test set & generate Figures 8-11
python src/evaluate.py
```

Or open the interactive Jupyter Notebooks:
```bash
jupyter notebook notebooks/01_eda_and_data_cleaning.ipynb
jupyter notebook notebooks/02_deep_learning_and_evaluation.ipynb
```

---

## 🧠 Model Architecture & Hyperparameter Tuning Findings

### Deep Neural Network (MLP) Specifications
* **Input Layer:** 30 features
* **Dense Layer 1:** 64 neurons, Batch Normalization, ReLU activation, Dropout (0.30)
* **Dense Layer 2:** 32 neurons, Batch Normalization, ReLU activation, Dropout (0.30)
* **Dense Layer 3:** 16 neurons, Batch Normalization, ReLU activation
* **Output Layer:** 1 neuron, Sigmoid activation (calibrated fraud probability)
* **Optimizer:** Adam ($\text{learning rate} = 10^{-3}$)
* **Loss:** Binary Cross-Entropy

### Experimental Comparison Summary
* **Class Weights Strategy (Configs 1–3):** Achieved high recall ($\approx 90\%$) but suffered from very low precision ($\approx 5\%$), producing overwhelming false alarms due to the $599\times$ penalty multiplier.
* **SMOTE Resampling Strategy (Config 4 - Champion):** By synthesizing localized minority instances exclusively in the training fold, the network learned robust decision boundaries, lifting validation precision to **65.91%** and validation PR-AUC to **0.7802**, increasing the F1-Score by more than **$6\times$**.

---

## 📚 Dataset Citation & Acknowledgement
The dataset was collected during a research collaboration of Worldline and the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) on big data mining and fraud detection:
* *Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In IEEE Symposium Series on Computational Intelligence (SSCI), 2015.*
* Dataset URL: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
