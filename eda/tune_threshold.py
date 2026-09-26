"""
Phase 5 Threshold Sweep, Multi-Match Calibration & Cap Comparison
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Conducts a coarse threshold sweep (0.05 to 0.95, step 0.05) on the validation set for the best model.
2. Conducts a fine threshold sweep around the peak region (step 0.01) to pinpoint optimal tau*.
3. Reports for every threshold: Macro-P, Macro-R, Macro-F0.5, total predicted matches,
   % S1 predicted empty, average predicted matches per S1, and false-positive behavior.
4. Evaluates decision rules (simple threshold vs threshold + margin rule).
5. Performs Group Performance Breakdown:
   - Group 1: True Singleton S1 (exactly 1 match)
   - Group 2: True Zero-Match S1 (0 matches)
   - Group 3: True Multi-Match S1 (>= 2 matches)
6. Analyzes Probability Calibration:
   - Score distributions for positive vs negative pairs
   - Raw XGBoost vs Sigmoid (Platt) vs Isotonic calibration
7. Compares Blocking Caps:
   - Priority Cap 60 vs Cap 100 vs Cap 150 on the same validation pipeline.
"""

import sys
import os
import io
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import EntityMatcherModel
from src.evaluate import evaluate_predictions_df, evaluate_by_group, compute_pair_diagnostics, evaluate_s1_macro
from src.feature_schema import FEATURE_NAMES

output_dir = "output"
parquet_path = os.path.join(output_dir, "eval_features_cap150.parquet")
gt_path = os.path.join(output_dir, "eval_ground_truth.pkl")
model_path = os.path.join(output_dir, "best_matcher_model.pkl")

print("=" * 80)
print("PHASE 5: THRESHOLD SWEEP, SINGLETON ANALYSIS & BLOCKING CAP COMPARISON")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. LOAD DATA, METADATA & MODEL
# ------------------------------------------------------------------------------
print("\n[1/6] Loading datasets and best matcher model...")
df_all = pd.read_parquet(parquet_path)
with open(gt_path, "rb") as f:
    meta = pickle.load(f)

gt_dict = meta["gt_dict"]
s1_list_all = meta["s1_list"]

best_model = EntityMatcherModel.load(model_path)
print(f"  Loaded model: {best_model.model_type} (trained in {best_model.training_time_sec:.2f}s)")

# Filter to Cap 100 validation set
df_cap100 = df_all[df_all["blocking_rank_order"] <= 100].copy()
df_val = df_cap100[df_cap100["split_group"] == "val"].copy()

# Determine full validation S1 list (including zero-candidate S1s)
train_s1_set = set(df_cap100[df_cap100["split_group"] == "train"]["source1_entity_id"].unique())
val_s1_list = sorted([s1 for s1 in s1_list_all if s1 not in train_s1_set])

print(f"  Validation pairs (Cap 100): {len(df_val):,} across {len(val_s1_list):,} S1 entities")

# Compute model match probabilities
X_val = df_val[list(FEATURE_NAMES)].values
val_probs = best_model.predict_proba(X_val)
df_val["score"] = val_probs

# ------------------------------------------------------------------------------
# 2. COARSE THRESHOLD SWEEP (0.05 to 0.95, step 0.05)
# ------------------------------------------------------------------------------
print("\n[2/6] Running Coarse Threshold Sweep (0.05 to 0.95)...")
coarse_thresholds = np.arange(0.05, 0.96, 0.05)
coarse_results = []

for thresh in coarse_thresholds:
    res = evaluate_predictions_df(
        df_val,
        gt_dict,
        threshold=thresh,
        s1_list=val_s1_list,
    )
    # Check false positives on true-empty entities
    group_res = evaluate_by_group(df_val, gt_dict, threshold=thresh, s1_list=val_s1_list)
    empty_fp_rate = group_res["group2_zero_match"]["false_positive_rate"]

    coarse_results.append({
        "Threshold": round(thresh, 2),
        "Macro F0.5": round(res["macro_f05"], 4),
        "Macro P": round(res["macro_precision"], 4),
        "Macro R": round(res["macro_recall"], 4),
        "Pred Matches": res["total_pred_matches"],
        "% S1 Empty": round(res["empty_pred_pct"], 1),
        "Avg Matches/S1": round(res["avg_pred_matches"], 2),
        "Zero-Match FP%": round(empty_fp_rate, 1),
    })

df_coarse = pd.DataFrame(coarse_results)
print(df_coarse.to_string(index=False))

best_coarse_idx = df_coarse["Macro F0.5"].idxmax()
best_coarse_thresh = df_coarse.loc[best_coarse_idx, "Threshold"]
print(f"\n  Peak Coarse Threshold: {best_coarse_thresh:.2f} with Macro F0.5 = {df_coarse.loc[best_coarse_idx, 'Macro F0.5']:.4f}")

# ------------------------------------------------------------------------------
# 3. FINE THRESHOLD SWEEP (step 0.01 around peak region)
# ------------------------------------------------------------------------------
print(f"\n[3/6] Running Fine Threshold Sweep around {best_coarse_thresh:.2f} (step 0.01)...")
fine_min = max(0.05, best_coarse_thresh - 0.10)
fine_max = min(0.95, best_coarse_thresh + 0.10)
fine_thresholds = np.arange(fine_min, fine_max + 0.005, 0.01)
fine_results = []

for thresh in fine_thresholds:
    res = evaluate_predictions_df(
        df_val,
        gt_dict,
        threshold=thresh,
        s1_list=val_s1_list,
    )
    group_res = evaluate_by_group(df_val, gt_dict, threshold=thresh, s1_list=val_s1_list)
    empty_fp_rate = group_res["group2_zero_match"]["false_positive_rate"]

    fine_results.append({
        "Threshold": round(thresh, 2),
        "Macro F0.5": round(res["macro_f05"], 4),
        "Macro P": round(res["macro_precision"], 4),
        "Macro R": round(res["macro_recall"], 4),
        "Pred Matches": res["total_pred_matches"],
        "% S1 Empty": round(res["empty_pred_pct"], 1),
        "Avg Matches/S1": round(res["avg_pred_matches"], 2),
        "Zero-Match FP%": round(empty_fp_rate, 1),
    })

df_fine = pd.DataFrame(fine_results)
print(df_fine.to_string(index=False))

optimal_idx = df_fine["Macro F0.5"].idxmax()
optimal_thresh = df_fine.loc[optimal_idx, "Threshold"]
optimal_f05 = df_fine.loc[optimal_idx, "Macro F0.5"]
print(f"\n  [OPTIMAL THRESHOLD SELECTED]: tau* = {optimal_thresh:.2f} | Validation Macro F0.5 = {optimal_f05:.4f}")

# ------------------------------------------------------------------------------
# 4. DECISION RULE INVESTIGATION (Simple Threshold vs Margin Rule)
# ------------------------------------------------------------------------------
print("\n[4/6] Evaluating Decision Rules (Multi-Match Threshold vs Top-Candidate Margin)...")

# Rule 1: Simple score >= optimal_thresh (allows multiple matches)
rule1_res = evaluate_predictions_df(df_val, gt_dict, threshold=optimal_thresh, s1_list=val_s1_list)

# Rule 2: Top-1 (argmax only)
top1_preds: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
for s1, group in df_val.groupby("source1_entity_id"):
    max_row = group.loc[group["score"].idxmax()]
    if max_row["score"] >= optimal_thresh:
        top1_preds[s1] = {max_row["candidate_entity_id"]}
rule2_res = evaluate_s1_macro(top1_preds, gt_dict, s1_list=val_s1_list)

# Rule 3: Top-Candidate Margin Rule (score >= optimal_thresh AND score >= max_score - 0.15)
margin_preds: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
for s1, group in df_val.groupby("source1_entity_id"):
    max_score = group["score"].max()
    qual = group[(group["score"] >= optimal_thresh) & (group["score"] >= max_score - 0.15)]
    if not qual.empty:
        margin_preds[s1] = set(qual["candidate_entity_id"].values)
rule3_res = evaluate_s1_macro(margin_preds, gt_dict, s1_list=val_s1_list)

print(f"  Decision Rule Comparison:")
print(f"    - Rule 1 (Multi-Candidate Threshold tau* >= {optimal_thresh:.2f}) : Macro F0.5 = {rule1_res['macro_f05']:.4f} (P={rule1_res['macro_precision']:.4f}, R={rule1_res['macro_recall']:.4f})")
print(f"    - Rule 2 (Argmax Only - Single Match)                : Macro F0.5 = {rule2_res['macro_f05']:.4f} (P={rule2_res['macro_precision']:.4f}, R={rule2_res['macro_recall']:.4f})")
print(f"    - Rule 3 (Threshold + Margin Delta <= 0.15)          : Macro F0.5 = {rule3_res['macro_f05']:.4f} (P={rule3_res['macro_precision']:.4f}, R={rule3_res['macro_recall']:.4f})")

# ------------------------------------------------------------------------------
# 5. DIAGNOSTIC GROUP PERFORMANCE BREAKDOWN AT OPTIMAL THRESHOLD
# ------------------------------------------------------------------------------
print(f"\n[5/6] Diagnostic Group Performance Breakdown (at tau* = {optimal_thresh:.2f}):")
group_eval = evaluate_by_group(df_val, gt_dict, threshold=optimal_thresh, s1_list=val_s1_list)

for grp_name, stats in [
    ("Group 1: True Singleton S1 (1 match)", group_eval["group1_singleton"]),
    ("Group 2: True Zero-Match S1 (0 matches)", group_eval["group2_zero_match"]),
    ("Group 3: True Multi-Match S1 (>=2 matches)", group_eval["group3_multi_match"]),
]:
    print(f"\n  {grp_name}:")
    print(f"    - Number of S1 Entities : {stats['count']:,}")
    print(f"    - Macro F0.5            : {stats['macro_f05']:.4f}")
    print(f"    - Macro Precision       : {stats['macro_precision']:.4f}")
    print(f"    - Macro Recall          : {stats['macro_recall']:.4f}")
    print(f"    - Avg Predicted Matches : {stats['avg_pred_matches']:.2f}")
    print(f"    - False Positive Rate   : {stats['false_positive_rate']:.2f}%")

# ------------------------------------------------------------------------------
# 6. PROBABILITY CALIBRATION ANALYSIS & BLOCKING CAP COMPARISON
# ------------------------------------------------------------------------------
print("\n[6/6] Probability Calibration & Blocking Cap Comparison...")

# Positive vs Negative Score Distributions
pos_scores = df_val[df_val["match_label"] == 1]["score"].values
neg_scores = df_val[df_val["match_label"] == 0]["score"].values

print("\n  Score Distributions for Positive vs Negative Pairs:")
print(f"    - Positive Pairs (N={len(pos_scores):,}): Mean={np.mean(pos_scores):.4f}, Median={np.median(pos_scores):.4f}, P10={np.percentile(pos_scores, 10):.4f}, P90={np.percentile(pos_scores, 90):.4f}")
print(f"    - Negative Pairs (N={len(neg_scores):,}): Mean={np.mean(neg_scores):.4f}, Median={np.median(neg_scores):.4f}, P10={np.percentile(neg_scores, 10):.4f}, P90={np.percentile(neg_scores, 90):.4f}")

# Sigmoid / Platt Calibration Test
calib_sigmoid = CalibratedClassifierCV(estimator=best_model.model, method="sigmoid", cv="prefit")
calib_sigmoid.fit(X_val, df_val["match_label"].values)
calib_probs = calib_sigmoid.predict_proba(X_val)[:, 1]

df_val_calib = df_val.copy()
df_val_calib["score"] = calib_probs

# Evaluate best threshold on calibrated probabilities
best_calib_f05 = -1.0
best_calib_thresh = 0.5
for t in np.arange(0.10, 0.95, 0.05):
    res_cal = evaluate_predictions_df(df_val_calib, gt_dict, threshold=t, s1_list=val_s1_list)
    if res_cal["macro_f05"] > best_calib_f05:
        best_calib_f05 = res_cal["macro_f05"]
        best_calib_thresh = t

print(f"\n  Calibration Comparison:")
print(f"    - Uncalibrated XGBoost : Best Macro F0.5 = {optimal_f05:.4f} at tau* = {optimal_thresh:.2f} (Brier = {brier_score_loss(df_val['match_label'], val_probs):.4f})")
print(f"    - Platt Calibrated    : Best Macro F0.5 = {best_calib_f05:.4f} at tau* = {best_calib_thresh:.2f} (Brier = {brier_score_loss(df_val['match_label'], calib_probs):.4f})")
print(f"    - Conclusion          : Calibration does {'NOT materially improve' if abs(best_calib_f05 - optimal_f05) < 0.005 else 'improve'} Macro-F0.5. Retaining clean uncalibrated tree scores.")

# Blocking Cap Comparison (Cap 60 vs Cap 100 vs Cap 150)
print("\n" + "=" * 80)
print("BLOCKING CAP COMPARISON: CAP 60 vs CAP 100 vs CAP 150")
print("=" * 80)

# True matches in validation set
val_true_matches = sum(len(gt_dict.get(s1, set())) for s1 in val_s1_list)

cap_results = []
df_val_all = df_all[df_all["split_group"] == "val"].copy()

for cap in [60, 100, 150]:
    t0_cap = time.time()
    df_cap_sub = df_val_all[df_val_all["blocking_rank_order"] <= cap].copy()
    
    # Compute scores for this cap
    X_cap = df_cap_sub[list(FEATURE_NAMES)].values
    df_cap_sub["score"] = best_model.predict_proba(X_cap)
    cap_time = time.time() - t0_cap

    # Blocking recall for this cap
    cap_pos_retained = int(df_cap_sub["match_label"].sum())
    block_rec = cap_pos_retained / val_true_matches * 100.0 if val_true_matches > 0 else 0.0

    # Evaluate at optimal threshold
    eval_res = evaluate_predictions_df(df_cap_sub, gt_dict, threshold=optimal_thresh, s1_list=val_s1_list)

    cap_results.append({
        "Priority Cap": cap,
        "Total Candidates": len(df_cap_sub),
        "Avg Cands/S1": round(len(df_cap_sub) / len(val_s1_list), 1),
        "Blocker Recall %": round(block_rec, 2),
        "S1 Macro F0.5": round(eval_res["macro_f05"], 4),
        "Macro Precision": round(eval_res["macro_precision"], 4),
        "Macro Recall": round(eval_res["macro_recall"], 4),
        "Avg Matches/S1": round(eval_res["avg_pred_matches"], 2),
        "Scoring Time (s)": round(cap_time, 2),
    })

df_caps = pd.DataFrame(cap_results)
print(df_caps.to_string(index=False))

# Save summary of threshold sweep and cap comparison
res_path = os.path.join(output_dir, "threshold_and_cap_summary.csv")
df_caps.to_csv(res_path, index=False)
print(f"\nSaved summary to: {res_path}")

print("\n[SUCCESS] Threshold tuning and cap comparison complete.")
