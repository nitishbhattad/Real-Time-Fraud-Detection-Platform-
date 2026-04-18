"""
🛡️ Fraud Detection Model — Comprehensive Evaluation
=====================================================
Project: Real-Time Fraud Detection Platform
Authors: Nitish Bhattad & Sowmika Dinesh
Course:  DSC 550 — Master's Flagship Project

Run this script from your kafka-fraud-detection directory:
    python model_evaluation.py

It will generate all evaluation plots and save them as PNGs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# Plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================
# 1. PATHS — Update BASE_DIR if your project is elsewhere
# ============================================================

BASE_DIR = "/Users/nitishbhattad/Desktop/kafka-fraud-detection"
PAYSIM_PATH = os.path.join(BASE_DIR, "data", "paysim.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model_paysim_rf.pkl")

features = [
    "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]
target = "isFraud"

# ============================================================
# 2. LOAD DATA & REPRODUCE TRAIN/TEST SPLIT
# ============================================================

print("📄 Loading PaySim dataset...")
df_raw = pd.read_csv(PAYSIM_PATH)
df_raw.columns = df_raw.columns.str.strip()
df_model = df_raw[features + [target]].copy()

df_sampled = df_model.sample(n=200_000, random_state=42)
data = df_sampled

X = data[features].copy()
y = data[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Dataset:  {len(df_raw):,} total rows")
print(f"  Sampled:  {len(data):,} rows")
print(f"  Train:    {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Fraud rate (test): {y_test.mean():.4f} ({y_test.sum()} fraud / {len(y_test)})")

# ============================================================
# 3. LOAD MODEL
# ============================================================

print(f"\n🔄 Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"  Pipeline steps: {[step[0] for step in model.steps]}")
print(f"  Classifier: {model.named_steps['clf']}")

# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

y_proba = model.predict_proba(X_test)[:, 1]

BEST_THRESHOLD = 0.9
y_pred = (y_proba >= BEST_THRESHOLD).astype(int)
y_pred_default = (y_proba >= 0.5).astype(int)

print(f"\n✅ Predictions generated for {len(y_test):,} test samples.")
print(f"  Threshold used: {BEST_THRESHOLD}")
print(f"  Fraud predicted: {y_pred.sum()}  |  Fraud actual: {y_test.sum()}")

# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print(f"CLASSIFICATION REPORT (threshold = {BEST_THRESHOLD})")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score:    {roc_auc_score(y_test, y_proba):.4f}")
print(f"Average Precision (PR-AUC): {average_precision_score(y_test, y_proba):.4f}")

# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

print("\n📊 Generating Confusion Matrix...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp1 = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
disp1.plot(ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title(f"Confusion Matrix (counts)\nThreshold = {BEST_THRESHOLD}", fontsize=14)

cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=["Legit", "Fraud"])
disp2.plot(ax=axes[1], cmap="Oranges", values_format=".2%")
axes[1].set_title(f"Confusion Matrix (normalized)\nThreshold = {BEST_THRESHOLD}", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eval_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")
print(f"  FPR: {fp/(fp+tn):.4f}  |  FNR: {fn/(fn+tp):.4f}")

# ============================================================
# 7. ROC CURVE
# ============================================================

print("\n📊 Generating ROC Curve...")
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"RandomForest (AUC = {auc_score:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")

idx_best = np.argmin(np.abs(thresholds_roc - BEST_THRESHOLD))
ax.scatter(fpr[idx_best], tpr[idx_best], s=150, c="red", zorder=5,
           label=f"Operating Point (thr={BEST_THRESHOLD})")
ax.annotate(f"  FPR={fpr[idx_best]:.4f}\n  TPR={tpr[idx_best]:.4f}",
            xy=(fpr[idx_best], tpr[idx_best]), fontsize=10, color="red")

ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
ax.set_title("ROC Curve — Fraud Detection Model", fontsize=15, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eval_roc_curve.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 8. PRECISION-RECALL CURVE
# ============================================================

print("\n📊 Generating Precision-Recall Curve...")
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall_vals, precision_vals, color="#16A34A", lw=2.5,
        label=f"RandomForest (AP = {ap_score:.4f})")

baseline = y_test.mean()
ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
           label=f"Baseline (prevalence = {baseline:.4f})")

idx_pr = np.argmin(np.abs(thresholds_pr - BEST_THRESHOLD))
ax.scatter(recall_vals[idx_pr], precision_vals[idx_pr], s=150, c="red", zorder=5,
           label=f"Operating Point (thr={BEST_THRESHOLD})")
ax.annotate(f"  P={precision_vals[idx_pr]:.2f}\n  R={recall_vals[idx_pr]:.2f}",
            xy=(recall_vals[idx_pr], precision_vals[idx_pr]), fontsize=10, color="red")

ax.set_xlabel("Recall", fontsize=13)
ax.set_ylabel("Precision", fontsize=13)
ax.set_title("Precision-Recall Curve — Fraud Detection Model", fontsize=15, fontweight="bold")
ax.legend(loc="upper right", fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eval_pr_curve.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 9. THRESHOLD ANALYSIS
# ============================================================

print("\n📊 Generating Threshold Analysis...")
thresholds = np.arange(0.1, 1.0, 0.05)
rows = []

for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_thr, average="binary", zero_division=0
    )
    rows.append({"threshold": thr, "precision": p, "recall": r, "f1": f1})

thr_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thr_df["threshold"], thr_df["precision"], "o-", label="Precision", color="#2563EB", lw=2)
ax.plot(thr_df["threshold"], thr_df["recall"], "s-", label="Recall", color="#DC2626", lw=2)
ax.plot(thr_df["threshold"], thr_df["f1"], "^-", label="F1 Score", color="#16A34A", lw=2.5)

best_idx = thr_df["f1"].idxmax()
ax.axvline(x=thr_df.loc[best_idx, "threshold"], color="gray", linestyle="--", alpha=0.7)
ax.scatter(thr_df.loc[best_idx, "threshold"], thr_df.loc[best_idx, "f1"],
           s=200, c="gold", edgecolors="black", zorder=5,
           label=f"Best F1 @ {thr_df.loc[best_idx, 'threshold']:.2f}")

ax.set_xlabel("Decision Threshold", fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("Precision / Recall / F1 vs. Threshold", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim([0.05, 0.95])
ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eval_threshold_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\nFull threshold scan:")
print(thr_df.to_string(index=False))

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\n📊 Generating Feature Importance...")
preprocessor = model.named_steps["preprocess"]
clf = model.named_steps["clf"]

cat_encoder = preprocessor.named_transformers_["cat"]
cat_names = cat_encoder.get_feature_names_out(["type"]).tolist()
num_names = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
all_feature_names = cat_names + num_names

importances = clf.feature_importances_

feat_imp = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2563EB" if v >= feat_imp["importance"].quantile(0.75) else "#93C5FD"
          for v in feat_imp["importance"]]
ax.barh(feat_imp["feature"], feat_imp["importance"], color=colors, edgecolor="white")
ax.set_xlabel("Importance (Gini)", fontsize=13)
ax.set_title("Feature Importance — RandomForest", fontsize=15, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eval_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\nFeature importance ranking:")
print(feat_imp.sort_values("importance", ascending=False).to_string(index=False))

# ============================================================
# 11. SCORE DISTRIBUTION
# ============================================================

print("\n📊 Generating Score Distribution...")
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(y_proba[y_test == 0], bins=50, alpha=0.6, color="#3B82F6",
        label="Legit", density=True, edgecolor="white")
ax.hist(y_proba[y_test == 1], bins=50, alpha=0.7, color="#EF4444",
        label="Fraud", density=True, edgecolor="white")

ax.axvline(x=BEST_THRESHOLD, color="black", linestyle="--", lw=2,
           label=f"Threshold = {BEST_THRESHOLD}")

ax.set_xlabel("ML Fraud Probability Score", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Score Distribution — Fraud vs. Legitimate", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eval_score_distribution.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 12. HYBRID DETECTION COMPARISON
# ============================================================

print("\n📊 Generating Hybrid Detection Comparison...")
HIGH_AMOUNT_THRESHOLD = 400_000.0
ZERO_PATTERN_MIN_AMOUNT = 50_000.0

X_test_rules = X_test.copy()
X_test_rules["high_amount_flag"] = (X_test_rules["amount"] > HIGH_AMOUNT_THRESHOLD).astype(int)
X_test_rules["zero_balance_flag"] = (
    X_test_rules["type"].isin(["TRANSFER", "CASH_OUT"])
    & (X_test_rules["oldbalanceOrg"] == 0.0)
    & (X_test_rules["newbalanceOrig"] == 0.0)
    & (X_test_rules["amount"] > ZERO_PATTERN_MIN_AMOUNT)
).astype(int)

rule_pred = ((X_test_rules["high_amount_flag"] == 1) | (X_test_rules["zero_balance_flag"] == 1)).astype(int)
ml_pred = y_pred
hybrid_pred = ((ml_pred == 1) | (rule_pred == 1)).astype(int)

print("\n" + "=" * 65)
print("COMPARISON: ML-Only vs. Rules-Only vs. Hybrid")
print("=" * 65)

methods = ["ML-Only", "Rules-Only", "Hybrid"]
preds_list = [ml_pred, rule_pred, hybrid_pred]
metrics_data = []

for name, preds in zip(methods, preds_list):
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    print(f"\n{name}:")
    print(f"  Precision: {p:.4f}  |  Recall: {r:.4f}  |  F1: {f1:.4f}")
    print(f"  Flagged: {preds.sum()}  |  Actual Fraud: {y_test.sum()}")
    metrics_data.append({"Method": name, "Precision": p, "Recall": r, "F1": f1})

metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, metrics_df["Precision"], width, label="Precision", color="#3B82F6")
bars2 = ax.bar(x, metrics_df["Recall"], width, label="Recall", color="#EF4444")
bars3 = ax.bar(x + width, metrics_df["F1"], width, label="F1 Score", color="#16A34A")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("Detection Strategy Comparison\nML-Only vs. Rules-Only vs. Hybrid", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim([0, 1.15])

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eval_hybrid_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 13. SUMMARY TABLE
# ============================================================

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

summary = pd.DataFrame([
    ["ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}"],
    ["Average Precision (PR-AUC)", f"{average_precision_score(y_test, y_proba):.4f}"],
    ["Best Threshold", f"{BEST_THRESHOLD}"],
    ["Accuracy", f"{accuracy_score(y_test, y_pred):.4f}"],
    ["Precision (Fraud)", f"{tp/(tp+fp):.4f}"],
    ["Recall (Fraud)", f"{tp/(tp+fn):.4f}"],
    ["F1 Score (Fraud)", f"{f1_score(y_test, y_pred):.4f}"],
    ["True Positives", f"{tp}"],
    ["False Positives", f"{fp}"],
    ["False Negatives (missed)", f"{fn}"],
    ["True Negatives", f"{tn:,}"],
], columns=["Metric", "Value"])

print("\n" + "=" * 50)
print("📊 MODEL EVALUATION SUMMARY")
print("=" * 50)
print(summary.to_string(index=False))

summary.to_csv(os.path.join(BASE_DIR, "eval_summary_metrics.csv"), index=False)
print(f"\n✅ Summary saved to {BASE_DIR}/eval_summary_metrics.csv")

print("\n" + "=" * 50)
print("✅ ALL EVALUATION COMPLETE!")
print("=" * 50)
print(f"\nAll plots saved to: {BASE_DIR}/")
print("Files generated:")
for f in [
    "eval_confusion_matrix.png",
    "eval_roc_curve.png",
    "eval_pr_curve.png",
    "eval_threshold_analysis.png",
    "eval_feature_importance.png",
    "eval_score_distribution.png",
    "eval_hybrid_comparison.png",
    "eval_summary_metrics.csv",
]:
    print(f"  ✓ {f}")
