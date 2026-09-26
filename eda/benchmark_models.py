"""
Phase 5 Model Benchmarking & Comparison
Amazon ML Challenge 2026: Business Entity Resolution

Compares supervised matching models on the approved Phase 4 pipeline (Cap 100):
- Model A: Logistic Regression baseline (imputed, scaled, balanced class weights)
- Model B: Gradient-Boosted Trees (XGBoost 3.2.0 unweighted, conservative tabular hyperparameters)
- Model C1: XGBoost with mild class weighting (scale_pos_weight = sqrt(neg/pos))
- Model C2: XGBoost with full balanced class weighting (scale_pos_weight = neg/pos)
- Model C3: Random Forest baseline (balanced class weights)

Evaluates:
- Training & validation runtimes
- Pair-level ROC-AUC, PR-AUC, LogLoss, Brier score
- S1-level Macro-F0.5, Macro Precision, Macro Recall across validation S1 entities
- Compares class weighting impact on entity-level matching
"""

import sys
import os
import io
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import EntityMatcherModel
from src.evaluate import evaluate_predictions_df, compute_pair_diagnostics, evaluate_by_group
from src.feature_schema import FEATURE_NAMES

output_dir = "output"
parquet_path = os.path.join(output_dir, "eval_features_cap150.parquet")
gt_path = os.path.join(output_dir, "eval_ground_truth.pkl")

print("=" * 80)
print("PHASE 5: SUPERVISED MODEL BENCHMARK & COMPARISON")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. LOAD DATASET & METADATA
# ------------------------------------------------------------------------------
print("\n[1/5] Loading features and ground-truth metadata...")
if not os.path.exists(parquet_path):
    raise FileNotFoundError(f"Feature dataset not found at {parquet_path}. Run prepare_training_data.py first.")

df_all = pd.read_parquet(parquet_path)
with open(gt_path, "rb") as f:
    meta = pickle.load(f)

gt_dict = meta["gt_dict"]
s1_list_all = meta["s1_list"]

# Filter to approved Priority Cap = 100
df_cap100 = df_all[df_all["blocking_rank_order"] <= 100].copy()
print(f"  Total pairs at Cap 100: {len(df_cap100):,} across {df_cap100['source1_entity_id'].nunique():,} S1 entities")

# Split into train and val
df_train = df_cap100[df_cap100["split_group"] == "train"].copy()
df_val = df_cap100[df_cap100["split_group"] == "val"].copy()

val_s1_list = sorted(list(df_val["source1_entity_id"].unique()))

# Add any validation S1 that had 0 candidate pairs (if any)
all_val_s1 = set(s1 for s1 in s1_list_all if s1 not in set(df_train["source1_entity_id"].unique()))
val_s1_list = sorted(list(all_val_s1))

# ------------------------------------------------------------------------------
# 2. MEASURE EXACT CLASS IMBALANCE
# ------------------------------------------------------------------------------
print("\n[2/5] Exact Training Class Imbalance Analysis:")
train_pos = int(df_train["match_label"].sum())
train_neg = int(len(df_train) - train_pos)
imbalance_ratio = train_neg / train_pos if train_pos > 0 else 0
pos_rate = train_pos / len(df_train) * 100.0

val_pos = int(df_val["match_label"].sum())
val_neg = int(len(df_val) - val_pos)
val_imbalance = val_neg / val_pos if val_pos > 0 else 0
val_pos_rate = val_pos / len(df_val) * 100.0

print(f"  Training Set   : {len(df_train):,} pairs ({df_train['source1_entity_id'].nunique():,} S1)")
print(f"    - Positives  : {train_pos:,} ({pos_rate:.2f}%)")
print(f"    - Negatives  : {train_neg:,} ({100 - pos_rate:.2f}%)")
print(f"    - Imbalance  : Exactly {imbalance_ratio:.2f} negatives per positive ({imbalance_ratio:.1f}:1)")

print(f"  Validation Set : {len(df_val):,} pairs ({len(val_s1_list):,} S1)")
print(f"    - Positives  : {val_pos:,} ({val_pos_rate:.2f}%)")
print(f"    - Negatives  : {val_neg:,} ({100 - val_pos_rate:.2f}%)")
print(f"    - Imbalance  : Exactly {val_imbalance:.2f} negatives per positive ({val_imbalance:.1f}:1)")

X_train = df_train[list(FEATURE_NAMES)].values
y_train = df_train["match_label"].values

X_val = df_val[list(FEATURE_NAMES)].values
y_val = df_val["match_label"].values

# ------------------------------------------------------------------------------
# 3. DEFINE MODELS TO EVALUATE
# ------------------------------------------------------------------------------
print("\n[3/5] Defining models and hyperparameter configurations...")

sqrt_weight = float(np.sqrt(imbalance_ratio))
full_weight = float(imbalance_ratio)

models_config = {
    "Model A (Logistic Regression)": {
        "model_type": "logistic_regression",
        "params": {"C": 1.0, "class_weight": "balanced", "max_iter": 500},
    },
    "Model B (XGBoost Default)": {
        "model_type": "xgboost",
        "params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
        },
    },
    "Model C1 (XGBoost Mild Weight)": {
        "model_type": "xgboost_weighted",
        "params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": sqrt_weight,
        },
    },
    "Model C2 (XGBoost Full Weight)": {
        "model_type": "xgboost_weighted",
        "params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": full_weight,
        },
    },
    "Model C3 (Random Forest)": {
        "model_type": "random_forest",
        "params": {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
        },
    },
}

# ------------------------------------------------------------------------------
# 4. TRAIN AND EVALUATE EACH MODEL
# ------------------------------------------------------------------------------
print("\n[4/5] Training models and computing pair-level & S1-level metrics...")

results = []
val_predictions = df_val[["source1_entity_id", "candidate_entity_id", "match_label", "blocking_rank_order"]].copy()

best_model_obj = None
best_f05 = -1.0
best_model_name = ""

for name, cfg in models_config.items():
    print(f"\n--- Training {name} ---")
    model = EntityMatcherModel(
        model_type=cfg["model_type"],
        params=cfg["params"],
        random_state=42,
        feature_names=list(FEATURE_NAMES),
    )

    t0_fit = time.time()
    if cfg["model_type"] in ("xgboost", "xgboost_weighted"):
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=30, verbose=False)
    else:
        model.fit(X_train, y_train)
    fit_time = time.time() - t0_fit

    t0_val = time.time()
    val_probs = model.predict_proba(X_val)
    val_time = time.time() - t0_val

    # Store predicted probabilities
    col_name = f"prob_{name.split()[1].lower()}"
    val_predictions[col_name] = val_probs

    # Pair-level diagnostic metrics
    pair_diag = compute_pair_diagnostics(y_val, val_probs, threshold=0.5)

    # Temporary dataframe for S1-level evaluation
    df_eval_temp = df_val[["source1_entity_id", "candidate_entity_id"]].copy()
    df_eval_temp["score"] = val_probs

    # Search for best preliminary threshold between 0.1 and 0.9 (step 0.05)
    best_thresh = 0.5
    best_s1_f05 = -1.0
    best_s1_res = None

    for thresh in np.arange(0.10, 0.95, 0.05):
        s1_res = evaluate_predictions_df(
            df_eval_temp,
            gt_dict,
            threshold=thresh,
            s1_list=val_s1_list,
        )
        if s1_res["macro_f05"] > best_s1_f05:
            best_s1_f05 = s1_res["macro_f05"]
            best_thresh = thresh
            best_s1_res = s1_res

    print(f"  Training Time   : {fit_time:.2f}s")
    print(f"  Validation Time : {val_time:.2f}s ({len(df_val)/val_time:,.0f} pairs/sec)")
    print(f"  Pair ROC-AUC    : {pair_diag['roc_auc']:.4f}")
    print(f"  Pair PR-AUC     : {pair_diag['pr_auc']:.4f}")
    print(f"  Pair Log-Loss   : {pair_diag['log_loss']:.4f}")
    print(f"  Pair Brier Score: {pair_diag['brier_score']:.4f}")
    print(f"  Optimal Thresh  : {best_thresh:.2f}")
    print(f"  S1 Macro-F0.5   : {best_s1_res['macro_f05']:.4f} (P={best_s1_res['macro_precision']:.4f}, R={best_s1_res['macro_recall']:.4f})")
    print(f"  Pred Matches    : {best_s1_res['total_pred_matches']:,} (Avg {best_s1_res['avg_pred_matches']:.2f}/S1, {best_s1_res['empty_pred_pct']:.1f}% empty)")

    results.append({
        "Model": name,
        "Train Time (s)": round(fit_time, 2),
        "Val Time (s)": round(val_time, 2),
        "ROC-AUC": round(pair_diag["roc_auc"], 4),
        "PR-AUC": round(pair_diag["pr_auc"], 4),
        "Log-Loss": round(pair_diag["log_loss"], 4),
        "Brier": round(pair_diag["brier_score"], 4),
        "Best Thresh": round(best_thresh, 2),
        "S1 Macro-F0.5": round(best_s1_res["macro_f05"], 4),
        "S1 Macro-P": round(best_s1_res["macro_precision"], 4),
        "S1 Macro-R": round(best_s1_res["macro_recall"], 4),
        "Avg Matches/S1": round(best_s1_res["avg_pred_matches"], 2),
        "% S1 Empty": round(best_s1_res["empty_pred_pct"], 1),
    })

    if best_s1_res["macro_f05"] > best_f05:
        best_f05 = best_s1_res["macro_f05"]
        best_model_obj = model
        best_model_name = name

# ------------------------------------------------------------------------------
# 5. SUMMARY COMPARISON & SAVE ARTIFACTS
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 5 MODEL COMPARISON SUMMARY")
print("=" * 80)
df_summary = pd.DataFrame(results)
print(df_summary.to_string(index=False))

# Save validation predictions
preds_path = os.path.join(output_dir, "val_predictions_cap100.parquet")
val_predictions.to_parquet(preds_path, index=False)
print(f"\nSaved validation predictions to: {preds_path}")

# Save best model
best_model_path = os.path.join(output_dir, "best_matcher_model.pkl")
best_model_obj.save(best_model_path)
print(f"Saved best model ({best_model_name}) to: {best_model_path}")

print(f"\n[WINNER] Best Model: {best_model_name} with S1 Macro-F0.5 = {best_f05:.4f}")
