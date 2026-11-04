"""
eda_visualizer.py
Phase 2: Exploratory Data Analysis (EDA), Statistical Analysis & Probability Distributions.
College Rubric CO4 (Discrete/Continuous Distributions) & Ben Fry's 7 Visualization Stages.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for reliable server execution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_eda_and_probability_analysis(cleaned_csv_path: str, figures_dir: str):
    """
    Executes comprehensive EDA, statistical analysis, discrete/continuous distribution
    fitting, and exports publication-ready figures mapped to assignment rubric CO4.
    """
    print("=" * 75)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) & PROBABILITY DISTRIBUTIONS")
    print("=" * 75)

    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv(cleaned_csv_path)
    total_records = len(df)
    print(f"Loaded cleaned dataset: {total_records:,} records.")

    # Palette definition: Green for Genuine, Crimson for Fraud
    palette = {0: '#27ae60', 1: '#e74c3c'}
    class_labels = {0: 'Genuine (Class 0)', 1: 'Fraudulent (Class 1)'}

    # -------------------------------------------------------------------------
    # STAGE 4 (MINE) & CO4: STATISTICAL INFERENCE & PROBABILITY DISTRIBUTIONS
    # -------------------------------------------------------------------------
    print("\n--- STATISTICAL & DISTRIBUTION ANALYSIS (CO4 RUBRIC) ---")
    
    # 1. Discrete Distribution: Bernoulli Distribution for Target Variable
    p_fraud = (df['Class'] == 1).mean()
    p_genuine = 1.0 - p_fraud
    bernoulli_var = p_fraud * p_genuine
    print("\n[A] Discrete Probability Distribution: Bernoulli Model of Fraud Class")
    print(f"    - Parameter p (P(Y = 1, Fraud)): {p_fraud:.6f} ({p_fraud*100:.4f}%)")
    print(f"    - Parameter q (P(Y = 0, Genuine)): {p_genuine:.6f} ({p_genuine*100:.4f}%)")
    print(f"    - Expected Value E[Y] = p = {p_fraud:.6f}")
    print(f"    - Theoretical Variance Var(Y) = p*q = {bernoulli_var:.6f}")

    # 2. Discrete Distribution: Poisson Process for Hourly Fraud Arrival
    # Convert Time (seconds) to hours (0 to 48)
    df['Hour'] = (df['Time'] // 3600) % 48
    hourly_fraud_counts = df[df['Class'] == 1].groupby('Hour').size()
    lambda_poisson = hourly_fraud_counts.mean()
    poisson_var = hourly_fraud_counts.var()
    print("\n[B] Discrete Probability Distribution: Poisson Model for Hourly Fraud Arrival")
    print(f"    - Observed Time Horizon: 48 Hours")
    print(f"    - Mean Hourly Fraud Arrival Rate (lambda): {lambda_poisson:.3f} frauds/hour")
    print(f"    - Hourly Fraud Variance: {poisson_var:.3f}")
    print(f"    - Overdispersion Ratio (Var / Mean): {poisson_var / lambda_poisson:.3f}")

    # 3. Continuous Distributions: Amount & Key Feature V14
    print("\n[C] Continuous Distributions Analysis (Skewness & Kurtosis):")
    amount_skew = df['Amount'].skew()
    amount_kurt = df['Amount'].kurtosis()
    v14_skew = df['V14'].skew()
    v14_kurt = df['V14'].kurtosis()
    print(f"    - Transaction Amount: Skewness = {amount_skew:.3f} (Extreme Right Skew), Kurtosis = {amount_kurt:.3f}")
    print(f"    - Feature V14: Skewness = {v14_skew:.3f}, Kurtosis = {v14_kurt:.3f}")

    # Kolmogorov-Smirnov test for normality on V14 for genuine transactions
    v14_genuine = df[df['Class'] == 0]['V14']
    ks_stat, ks_pvalue = stats.kstest((v14_genuine - v14_genuine.mean()) / v14_genuine.std(), 'norm')
    print(f"    - Kolmogorov-Smirnov Normality Test on V14: Stat = {ks_stat:.4f}, p-value = {ks_pvalue:.4e}")

    # Correlation Analysis
    corr_with_class = df.corr()['Class'].drop(['Class', 'Hour']).sort_values()
    print("\n[D] Top Correlated Features with Fraud Class:")
    top_neg_corr = corr_with_class.head(5)
    top_pos_corr = corr_with_class.tail(5)
    print("    - Top 5 Negative Correlations (Higher value => Lower fraud probability):")
    for feat, val in top_neg_corr.items():
        print(f"        {feat:6s}: {val:+.4f}")
    print("    - Top 5 Positive Correlations (Higher value => Higher fraud probability):")
    for feat, val in top_pos_corr.items():
        print(f"        {feat:6s}: {val:+.4f}")

    # -------------------------------------------------------------------------
    # VISUALIZATION PIPELINE (BEN FRY'S 7 STAGES: REPRESENT, REFINE, INTERPRET)
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14})

    # FIGURE 1: Class Imbalance Distribution (Raw Count vs Log10 Scale)
    print("\nGenerating Figure 1: Class Imbalance Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = df['Class'].value_counts()

    # Raw counts
    bars = axes[0].bar(['Genuine (0)', 'Fraud (1)'], counts, color=[palette[0], palette[1]], edgecolor='black', width=0.5)
    axes[0].set_title("Transaction Frequency by Class (Linear Scale)", fontweight='bold')
    axes[0].set_ylabel("Number of Transactions", fontweight='bold')
    axes[0].set_xlabel("Transaction Class", fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f'{h:,}\n({h/total_records*100:.3f}%)',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Log10 scale for visual clarity
    bars_log = axes[1].bar(['Genuine (0)', 'Fraud (1)'], counts, color=[palette[0], palette[1]], edgecolor='black', width=0.5)
    axes[1].set_yscale('log')
    axes[1].set_title("Transaction Frequency by Class (Log10 Scale)", fontweight='bold')
    axes[1].set_ylabel("Number of Transactions (Log10 Scale)", fontweight='bold')
    axes[1].set_xlabel("Transaction Class", fontweight='bold')
    axes[1].annotate("Extreme Class Imbalance\n(0.167% Fraud Rate)", xy=(1, counts[1]),
                     xytext=(0.5, counts[1]*8),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     ha='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fce4ec", ec="red"))
    plt.tight_layout()
    f1_path = os.path.join(figures_dir, "fig1_class_distribution.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {f1_path}")

    # FIGURE 2: Transaction Amount Distribution (Continuous Density & Box Plot)
    print("Generating Figure 2: Transaction Amount Continuous Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log-transformed Amount to handle extreme skewness
    df['Log_Amount'] = np.log1p(df['Amount'])

    sns.kdeplot(data=df[df['Class'] == 0], x='Log_Amount', ax=axes[0], color=palette[0], label='Genuine (Class 0)', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(data=df[df['Class'] == 1], x='Log_Amount', ax=axes[0], color=palette[1], label='Fraud (Class 1)', fill=True, alpha=0.4, linewidth=2)
    axes[0].set_title("Probability Density of Log(Amount + 1)", fontweight='bold')
    axes[0].set_xlabel("Log(Amount in $ + 1)", fontweight='bold')
    axes[0].set_ylabel("Probability Density", fontweight='bold')
    axes[0].legend(title="Transaction Class", frameon=True)

    # Boxplot of Amount (trimmed for visual readability up to $500)
    sns.boxplot(data=df, x='Class', y='Amount', ax=axes[1], palette=[palette[0], palette[1]], showfliers=False, width=0.4)
    axes[1].set_xticklabels(['Genuine (0)', 'Fraud (1)'], fontweight='bold')
    axes[1].set_title("Transaction Amount Distribution without Outliers ($0 - $500)", fontweight='bold')
    axes[1].set_xlabel("Transaction Class", fontweight='bold')
    axes[1].set_ylabel("Transaction Amount ($)", fontweight='bold')

    plt.tight_layout()
    f2_path = os.path.join(figures_dir, "fig2_amount_distribution.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {f2_path}")

    # FIGURE 3: Temporal Trends & Poisson Arrival Analysis
    print("Generating Figure 3: Temporal Transaction Trends...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    hourly_genuine = df[df['Class'] == 0].groupby('Hour').size()
    hourly_fraud = df[df['Class'] == 1].groupby('Hour').size()

    axes[0].plot(hourly_genuine.index, hourly_genuine.values, color=palette[0], marker='o', linewidth=2.5, label='Genuine Volume')
    axes[0].set_title("Hourly Genuine Transaction Volume (48-Hour Cycle)", fontweight='bold')
    axes[0].set_ylabel("Volume (Transactions)", fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc="upper right")

    axes[1].bar(hourly_fraud.index, hourly_fraud.values, color=palette[1], edgecolor='black', alpha=0.75, label='Observed Frauds/Hour')
    axes[1].axhline(lambda_poisson, color='black', linestyle='--', linewidth=2, label=f'Poisson Mean (λ = {lambda_poisson:.2f}/hr)')
    axes[1].set_title("Hourly Fraud Transaction Frequency vs Theoretical Poisson Rate", fontweight='bold')
    axes[1].set_xlabel("Hour of Timeline (0 - 47 Hours)", fontweight='bold')
    axes[1].set_ylabel("Fraud Count", fontweight='bold')
    axes[1].set_xticks(range(0, 48, 2))
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    f3_path = os.path.join(figures_dir, "fig3_temporal_trends.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {f3_path}")

    # FIGURE 4: Feature Correlation Heatmap (Top Predictors)
    print("Generating Figure 4: Correlation Matrix Heatmap...")
    top_features = list(top_neg_corr.index) + list(top_pos_corr.index) + ['Amount', 'Class']
    corr_matrix = df[top_features].corr()

    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.5, vmax=0.5, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f",
                annot_kws={"size": 9, "fontweight": "bold"})
    plt.title("Correlation Matrix of Top Discriminative PCA Features & Amount with Target Class", fontweight='bold', pad=15)
    plt.tight_layout()
    f4_path = os.path.join(figures_dir, "fig4_correlation_heatmap.png")
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {f4_path}")

    # FIGURE 5: Top Discriminative PCA Feature Distributions (Continuous Densities)
    print("Generating Figure 5: Top Discriminative PCA Distributions...")
    discriminative_features = ['V14', 'V17', 'V12', 'V10', 'V11', 'V4']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, feat in enumerate(discriminative_features):
        sns.kdeplot(data=df[df['Class'] == 0], x=feat, ax=axes[idx], color=palette[0], label='Genuine', fill=True, alpha=0.25, linewidth=1.8)
        sns.kdeplot(data=df[df['Class'] == 1], x=feat, ax=axes[idx], color=palette[1], label='Fraud', fill=True, alpha=0.4, linewidth=1.8)
        axes[idx].set_title(f"Density of {feat} (Corr: {corr_with_class[feat]:+.3f})", fontweight='bold')
        axes[idx].set_xlabel(f"{feat} Value", fontweight='bold')
        axes[idx].set_ylabel("Density", fontweight='bold')
        axes[idx].legend(loc='upper right')

    plt.suptitle("Probability Density Functions of Top 6 Discriminative PCA Features", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    f5_path = os.path.join(figures_dir, "fig5_discriminative_pca_distributions.png")
    plt.savefig(f5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {f5_path}")

    # FIGURE 6: Probability Distribution Analysis (Rubric CO4)
    print("Generating Figure 6: Probability Distribution Proofs (CO4 Rubric)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [0, 0]: Discrete Bernoulli PMF
    p_vals = [p_genuine, p_fraud]
    bar_pmf = axes[0, 0].bar([0, 1], p_vals, color=[palette[0], palette[1]], edgecolor='black', width=0.4)
    axes[0, 0].set_title("Discrete Bernoulli PMF: P(Y = y)", fontweight='bold')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['y=0 (Genuine)', 'y=1 (Fraud)'], fontweight='bold')
    axes[0, 0].set_ylabel("Probability P(Y = y)", fontweight='bold')
    for b in bar_pmf:
        h = b.get_height()
        axes[0, 0].annotate(f'{h:.5f}',
                            xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

    # [0, 1]: Discrete Poisson Distribution Fit for Hourly Fraud
    k_range = np.arange(0, 25)
    poisson_pmf = stats.poisson.pmf(k_range, mu=lambda_poisson)
    axes[0, 1].hist(hourly_fraud.values, bins=range(0, 26), density=True, color=palette[1], alpha=0.5, edgecolor='black', label='Empirical Relative Freq')
    axes[0, 1].plot(k_range, poisson_pmf, 'ko--', linewidth=2, markersize=5, label=f'Poisson PMF (λ={lambda_poisson:.2f})')
    axes[0, 1].set_title("Discrete Poisson Model Fitting on Hourly Frauds", fontweight='bold')
    axes[0, 1].set_xlabel("Frauds per Hour (k)", fontweight='bold')
    axes[0, 1].set_ylabel("P(X = k)", fontweight='bold')
    axes[0, 1].legend()

    # [1, 0]: Continuous Cumulative Distribution Function (CDF) for Amount
    sorted_genuine_amt = np.sort(df[df['Class'] == 0]['Amount'])
    sorted_fraud_amt = np.sort(df[df['Class'] == 1]['Amount'])
    cdf_genuine = np.arange(1, len(sorted_genuine_amt) + 1) / len(sorted_genuine_amt)
    cdf_fraud = np.arange(1, len(sorted_fraud_amt) + 1) / len(sorted_fraud_amt)
    axes[1, 0].plot(sorted_genuine_amt, cdf_genuine, color=palette[0], linewidth=2.5, label='Genuine Amount CDF')
    axes[1, 0].plot(sorted_fraud_amt, cdf_fraud, color=palette[1], linewidth=2.5, label='Fraud Amount CDF')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title("Continuous Empirical CDF: Transaction Amount (Log Scale)", fontweight='bold')
    axes[1, 0].set_xlabel("Transaction Amount in $ (Log Scale)", fontweight='bold')
    axes[1, 0].set_ylabel("Cumulative Probability F(x)", fontweight='bold')
    axes[1, 0].legend()

    # [1, 1]: Quantile-Quantile (Q-Q) Plot for Feature V14 (Gaussian Test)
    stats.probplot(df['V14'], dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set_markerfacecolor('#3498db')
    axes[1, 1].get_lines()[0].set_markersize(2)
    axes[1, 1].get_lines()[1].set_color('red')
    axes[1, 1].get_lines()[1].set_linewidth(2)
    axes[1, 1].set_title("Normal Q-Q Plot: Feature V14 vs Standard Normal Distribution", fontweight='bold')
    axes[1, 1].set_xlabel("Theoretical Standard Normal Quantiles", fontweight='bold')
    axes[1, 1].set_ylabel("Sample Quantiles of V14", fontweight='bold')

    plt.suptitle("College Rubric CO4: Discrete & Continuous Probability Distributions Analysis", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    f6_path = os.path.join(figures_dir, "fig6_probability_distributions_co4.png")
    plt.savefig(f6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {f6_path}")

    print("=" * 75)
    print("PHASE 2 COMPLETE: ALL 6 FIGURES AND STATISTICAL TESTS GENERATED")
    print("=" * 75)

if __name__ == "__main__":
    cleaned_csv = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\data\processed\creditcard_cleaned.csv"
    figs_path = r"C:\Users\Admin\.gemini\antigravity\scratch\credit_card_fraud_detection\reports\figures"
    run_eda_and_probability_analysis(cleaned_csv, figs_path)
