"""
notebooks/eda.py
────────────────
EDA script — generates 6 plots and prints key findings.
Run: python notebooks/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 5)

# ── Load ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/raw/creditcard.csv")
print(f"Shape: {df.shape}")

fraud_amt = df[df["Class"] == 1]["Amount"]
legit_amt = df[df["Class"] == 0]["Amount"]
fraud_pct = df["Class"].mean() * 100
v_features = [f"V{i}" for i in range(1, 29)]

# ── Plot 1: Class Imbalance ───────────────────────────────────
print("\nPlot 1: Class imbalance...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = df["Class"].value_counts()
axes[0].bar(["Legitimate", "Fraud"], counts.values,
            color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Transaction Count by Class", fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

axes[1].pie(counts.values,
    labels=["Legitimate (99.83%)", "Fraud (0.17%)"],
    colors=["#2ecc71", "#e74c3c"],
    autopct="%1.3f%%",
    startangle=90,
    explode=(0, 0.1))
axes[1].set_title("Class Distribution", fontweight="bold")

plt.suptitle("Severe Class Imbalance", fontsize=13)
plt.tight_layout()
plt.savefig("data/processed/plot_01_class_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_01_class_imbalance.png")

# ── Plot 2: Amount Analysis ───────────────────────────────────
print("Plot 2: Amount analysis...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(legit_amt, bins=50, alpha=0.6, color="#2ecc71",
             label="Legit", density=True)
axes[0].hist(fraud_amt, bins=50, alpha=0.8, color="#e74c3c",
             label="Fraud", density=True)
axes[0].set_title("Amount Distribution")
axes[0].set_xlabel("Amount ($)")
axes[0].legend()
axes[0].set_xlim(0, 500)

axes[1].boxplot([legit_amt, fraud_amt], labels=["Legit", "Fraud"],
    patch_artist=True, boxprops=dict(facecolor="#2ecc71", alpha=0.7))
axes[1].set_title("Amount Boxplot")
axes[1].set_ylabel("Amount ($)")
axes[1].set_ylim(0, 500)

axes[2].hist(np.log1p(legit_amt), bins=50, alpha=0.6,
             color="#2ecc71", label="Legit", density=True)
axes[2].hist(np.log1p(fraud_amt), bins=50, alpha=0.8,
             color="#e74c3c", label="Fraud", density=True)
axes[2].set_title("Log(Amount+1) Distribution")
axes[2].set_xlabel("log(Amount+1)")
axes[2].legend()

plt.suptitle("Transaction Amount: Fraud vs Legitimate",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/plot_02_amount_analysis.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_02_amount_analysis.png")

# ── Plot 3: Time Analysis ─────────────────────────────────────
print("Plot 3: Time analysis...")
df["hour"] = (df["Time"] // 3600) % 24

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

hourly_all   = df.groupby("hour").size()
hourly_fraud = df[df["Class"] == 1].groupby("hour").size()

axes[0].bar(hourly_all.index, hourly_all.values,
            color="#3498db", alpha=0.7)
axes[0].set_title("All Transactions per Hour")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Count")

axes[1].bar(hourly_fraud.index, hourly_fraud.values,
            color="#e74c3c", alpha=0.8)
axes[1].set_title("Fraud Transactions per Hour")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Count")

plt.suptitle("Temporal Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/plot_03_time_analysis.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_03_time_analysis.png")

# ── Plot 4: Feature Separation ────────────────────────────────
print("Plot 4: Feature separation...")
fraud_means = df[df["Class"] == 1][v_features].mean()
legit_means = df[df["Class"] == 0][v_features].mean()
separation  = (fraud_means - legit_means).abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#e74c3c" if s > separation.median()
          else "#3498db" for s in separation.values]
ax.bar(separation.index, separation.values,
       color=colors, edgecolor="white")
ax.set_title("Feature Separation Power (|Mean Fraud - Mean Legit|)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("PCA Feature")
ax.set_ylabel("Absolute Mean Difference")
ax.axhline(y=separation.median(), color="orange", linestyle="--",
           label=f"Median = {separation.median():.2f}")
ax.legend()
plt.tight_layout()
plt.savefig("data/processed/plot_04_feature_separation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_04_feature_separation.png")

# ── Plot 5: Top Features ──────────────────────────────────────
print("Plot 5: Top feature distributions...")
top_features = separation.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    axes[i].hist(df[df["Class"] == 0][feat], bins=60, alpha=0.5,
                 color="#2ecc71", label="Legit", density=True)
    axes[i].hist(df[df["Class"] == 1][feat], bins=60, alpha=0.7,
                 color="#e74c3c", label="Fraud", density=True)
    axes[i].set_title(f"{feat} Distribution", fontweight="bold")
    axes[i].legend(fontsize=9)

plt.suptitle("Top 6 Features — Fraud vs Legitimate",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/plot_05_top_features.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_05_top_features.png")

# ── Plot 6: Correlation Heatmap ───────────────────────────────
print("Plot 6: Correlation heatmap...")
corr = df[df["Class"] == 1][v_features + ["Amount"]].corr().round(2)

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title("Feature Correlation Matrix (Fraud Transactions)",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/plot_06_correlation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_06_correlation.png")

# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 55)
print("EDA SUMMARY — KEY FINDINGS")
print("=" * 55)
print(f"1. IMBALANCE : {fraud_pct:.3f}% fraud")
print(f"   → Use SMOTE + evaluate with PR-AUC not accuracy")
print(f"2. AMOUNT    : Fraud median ${fraud_amt.median():.2f}"
      f" vs legit ${legit_amt.median():.2f}")
print(f"   → Log-transform Amount to reduce skew")
print(f"3. FEATURES  : Top 3: {list(separation.head(3).index)}")
print(f"   → These will have highest SHAP importance")
print(f"4. TIME      : Fraud occurs across all hours")
print(f"   → Use cyclical sin/cos encoding for hour")
print(f"5. MISSING   : {df.isnull().sum().sum()} missing values")
print(f"   → No imputation needed")
print()
print("All 6 plots saved to data/processed/")
print("EDA complete!")