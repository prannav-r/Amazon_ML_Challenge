"""
Phase 5.1: Untouched S1 Holdout Validation & Cap Verification
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Creates a 3-way S1-level stratified split (Train ~60%, Dev ~20%, Untouched Holdout ~20%) with zero leakage.
2. Reports comprehensive split statistics (S1 count, singletons, zero-matches, multi-matches, pairs, positives, zero-candidate S1s).
3. Trains Model C1 (XGBoost Mild Weight) on the training split.
4. On the Development Set:
   - Performs independent threshold searches for Cap 100 and Cap 150.
   - Selects the final configuration (Cap and threshold) based strictly on Dev S1 Macro-F0.5.
5. Freezes the configuration.
6. Evaluates EXACTLY ONCE on the completely untouched Final Holdout set.
7. Reports unbiased holdout metrics, group breakdowns, source/country breakdowns, and side-by-side comparison.
"""

import sys
import os
import io
import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import EntityMatcherModel
from src.evaluate import evaluate_predictions_df, evaluate_by_group, evaluate_s1_macro, compute_pair_diagnostics
from src.feature_schema import FEATURE_NAMES

output_dir = "output"
parquet_path = os.path.join(output_dir, "eval_features_cap150.parquet")
gt_path = os.path.join(output_dir, "eval_ground_truth.pkl")

print("=" * 80)
print("PHASE 5.1: UNTOUCHED S1 HOLDOUT VALIDATION & CAP VERIFICATION")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. LOAD DATA & BUILD 3-WAY S1-LEVEL STRATIFIED SPLIT
# ------------------------------------------------------------------------------
print("\n[1/5] Loading data and constructing 3-way S1-level stratified split...")
df_all = pd.read_parquet(parquet_path)
with open(gt_path, "rb") as f:
    meta = pickle.load(f)

gt_dict = meta["gt_dict"]
s1_list_all = meta["s1_list"]
s1_lookup = meta["s1_lookup"]
cand_lookup = meta["cand_lookup"]

# Stratify by Country (US/India) and Match Count (Zero, Singleton, Multi)
strata = defaultdict(list)
for s1 in s1_list_all:
    c = s1_lookup[s1]["country"]
    num_matches = len(gt_dict.get(s1, set()))
    if num_matches == 0:
        m_type = "zero"
    elif num_matches == 1:
        m_type = "single"
    else:
        m_type = "multi"
    strata[(c, m_type)].append(s1)

train_s1 = []
dev_s1 = []
holdout_s1 = []

rng = np.random.RandomState(42)
for (c, m_type), ids in sorted(strata.items()):
    shuffled = rng.permutation(ids)
    n = len(shuffled)
    n_train = int(n * 0.60)
    n_dev = int(n * 0.20)
    train_s1.extend(shuffled[:n_train])
    dev_s1.extend(shuffled[n_train:n_train + n_dev])
    holdout_s1.extend(shuffled[n_train + n_dev:])

# Verify zero leakage
assert len(set(train_s1) & set(dev_s1)) == 0, "Leakage between Train and Dev!"
assert len(set(train_s1) & set(holdout_s1)) == 0, "Leakage between Train and Holdout!"
assert len(set(dev_s1) & set(holdout_s1)) == 0, "Leakage between Dev and Holdout!"
assert len(train_s1) + len(dev_s1) + len(holdout_s1) == len(s1_list_all), "Total S1 count mismatch!"

train_s1_set = set(train_s1)
dev_s1_set = set(dev_s1)
holdout_s1_set = set(holdout_s1)

# Assign split column
split_arr = np.array([
    "train" if s1 in train_s1_set else ("dev" if s1 in dev_s1_set else "holdout")
    for s1 in df_all["source1_entity_id"].values
])
df_all["split_3way"] = split_arr

# ------------------------------------------------------------------------------
# 2. REPORT SPLIT CHARACTERISTICS
# ------------------------------------------------------------------------------
def get_split_stats(s1_sublist: List[str], split_name: str, cap: int = 150) -> Dict[str, Any]:
    s1_set = set(s1_sublist)
    df_sub = df_all[(df_all["source1_entity_id"].isin(s1_set)) & (df_all["blocking_rank_order"] <= cap)]
    
    s1_with_cands = set(df_sub["source1_entity_id"].unique())
    zero_cand_s1 = len(s1_set - s1_with_cands)
    
    total_pos = int(df_sub["match_label"].sum())
    total_neg = len(df_sub) - total_pos
    imbalance = total_neg / total_pos if total_pos > 0 else 0
    
    us_cnt = sum(1 for s in s1_sublist if s1_lookup[s]["country"] == "US")
    in_cnt = sum(1 for s in s1_sublist if s1_lookup[s]["country"] == "India")
    zero_m_cnt = sum(1 for s in s1_sublist if len(gt_dict.get(s, set())) == 0)
    single_m_cnt = sum(1 for s in s1_sublist if len(gt_dict.get(s, set())) == 1)
    multi_m_cnt = sum(1 for s in s1_sublist if len(gt_dict.get(s, set())) > 1)
    
    return {
        "Split": split_name,
        "Total S1": len(s1_sublist),
        "US S1": us_cnt,
        "India S1": in_cnt,
        "Zero-Match S1": zero_m_cnt,
        "Singleton S1": single_m_cnt,
        "Multi-Match S1": multi_m_cnt,
        "Zero-Cand S1": zero_cand_s1,
        "Total Pairs (Cap 150)": len(df_sub),
        "Pos Pairs": total_pos,
        "Neg Pairs": total_neg,
        "Imbalance Ratio": round(imbalance, 2),
    }

stats_train = get_split_stats(train_s1, "Training (60%)", cap=150)
stats_dev = get_split_stats(dev_s1, "Development (20%)", cap=150)
stats_holdout = get_split_stats(holdout_s1, "Untouched Holdout (20%)", cap=150)

df_split_report = pd.DataFrame([stats_train, stats_dev, stats_holdout])
print("\n--- 3-WAY S1-LEVEL STRATIFIED DATASET BREAKDOWN ---")
print(df_split_report.to_string(index=False))

# ------------------------------------------------------------------------------
# 3. TRAIN WINNING MODEL C1 ON NEW TRAINING SPLIT
# ------------------------------------------------------------------------------
print("\n[2/5] Training Model C1 (XGBoost Mild Weight) on Training Split...")
df_train = df_all[(df_all["split_3way"] == "train") & (df_all["blocking_rank_order"] <= 150)].copy()

train_pos = int(df_train["match_label"].sum())
train_neg = int(len(df_train) - train_pos)
imbalance_train = train_neg / train_pos
mild_scale_weight = float(np.sqrt(imbalance_train))

print(f"  Training Pairs: {len(df_train):,} ({train_pos:,} pos, {train_neg:,} neg | Imbalance: {imbalance_train:.2f}:1)")
print(f"  Setting scale_pos_weight = sqrt({imbalance_train:.2f}) = {mild_scale_weight:.2f}")

X_train = df_train[list(FEATURE_NAMES)].values
y_train = df_train["match_label"].values

model_c1 = EntityMatcherModel(
    model_type="xgboost_weighted",
    params={
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": mild_scale_weight,
    },
    random_state=42,
    feature_names=list(FEATURE_NAMES),
)

t0_train = time.time()
model_c1.fit(X_train, y_train)
t_train = time.time() - t0_train
print(f"  Model trained in {t_train:.2f} seconds.")

# ------------------------------------------------------------------------------
# 4. DEVELOPMENT SET: REPRODUCE CAP 100 vs CAP 150 WITH INDEPENDENT THRESHOLD TUNING
# ------------------------------------------------------------------------------
print("\n[3/5] Development Set Evaluation & Independent Threshold Searches...")

df_dev_all = df_all[df_all["split_3way"] == "dev"].copy()
X_dev_all = df_dev_all[list(FEATURE_NAMES)].values
df_dev_all["score"] = model_c1.predict_proba(X_dev_all)

dev_true_matches = sum(len(gt_dict.get(s1, set())) for s1 in dev_s1)

dev_cap_summary = []
best_dev_configs = {}

for cap in [100, 150]:
    df_dev_cap = df_dev_all[df_dev_all["blocking_rank_order"] <= cap].copy()
    
    # Measure blocker recall on Dev
    dev_pos_retained = int(df_dev_cap["match_label"].sum())
    block_rec = dev_pos_retained / dev_true_matches * 100.0 if dev_true_matches > 0 else 0
    
    print(f"\n--- Independent Threshold Search on Dev Set for Cap {cap} ---")
    print(f"  Dev Pairs: {len(df_dev_cap):,} across {len(dev_s1):,} S1s | Blocker Recall: {block_rec:.2f}%")
    
    # Coarse sweep
    coarse_best_f05 = -1.0
    coarse_best_thresh = 0.5
    for t in np.arange(0.10, 0.95, 0.05):
        res = evaluate_predictions_df(df_dev_cap, gt_dict, threshold=t, s1_list=dev_s1)
        if res["macro_f05"] > coarse_best_f05:
            coarse_best_f05 = res["macro_f05"]
            coarse_best_thresh = t
            
    # Fine sweep around coarse peak
    fine_best_f05 = -1.0
    fine_best_thresh = coarse_best_thresh
    fine_best_res = None
    for t in np.arange(max(0.10, coarse_best_thresh - 0.10), min(0.95, coarse_best_thresh + 0.105), 0.01):
        res = evaluate_predictions_df(df_dev_cap, gt_dict, threshold=t, s1_list=dev_s1)
        if res["macro_f05"] > fine_best_f05:
            fine_best_f05 = res["macro_f05"]
            fine_best_thresh = t
            fine_best_res = res
            
    best_dev_configs[cap] = {
        "threshold": round(fine_best_thresh, 2),
        "results": fine_best_res,
        "blocker_recall": block_rec,
        "total_pairs": len(df_dev_cap),
    }
    
    print(f"  Optimal Threshold for Cap {cap}: tau* = {fine_best_thresh:.2f}")
    print(f"  Dev Macro F0.5 : {fine_best_res['macro_f05']:.4f} (P={fine_best_res['macro_precision']:.4f}, R={fine_best_res['macro_recall']:.4f})")
    print(f"  Predicted Matches: {fine_best_res['total_pred_matches']} (Avg {fine_best_res['avg_pred_matches']:.2f}/S1, {fine_best_res['empty_pred_pct']:.1f}% empty)")
    
    dev_cap_summary.append({
        "Cap": cap,
        "Total Candidates": len(df_dev_cap),
        "Avg Cands/S1": round(len(df_dev_cap) / len(dev_s1), 1),
        "Blocker Recall %": round(block_rec, 2),
        "Indep Tuned tau*": round(fine_best_thresh, 2),
        "Dev Macro F0.5": round(fine_best_res["macro_f05"], 4),
        "Dev Macro P": round(fine_best_res["macro_precision"], 4),
        "Dev Macro R": round(fine_best_res["macro_recall"], 4),
        "Avg Matches/S1": round(fine_best_res["avg_pred_matches"], 2),
        "% S1 Empty": round(fine_best_res["empty_pred_pct"], 1),
    })

print("\n--- DEVELOPMENT CAP COMPARISON SUMMARY ---")
df_dev_cap_summary = pd.DataFrame(dev_cap_summary)
print(df_dev_cap_summary.to_string(index=False))

# Select winning configuration strictly based on Dev Macro F0.5
if best_dev_configs[150]["results"]["macro_f05"] >= best_dev_configs[100]["results"]["macro_f05"]:
    selected_cap = 150
else:
    selected_cap = 100

frozen_threshold = best_dev_configs[selected_cap]["threshold"]
frozen_dev_f05 = best_dev_configs[selected_cap]["results"]["macro_f05"]

print(f"\n" + "=" * 80)
print(f"FROZEN CONFIGURATION (SELECTED ON DEV SET):")
print(f"  - Model: Model C1 (XGBoost Mild Weight, scale_pos_weight={mild_scale_weight:.2f})")
print(f"  - Priority Cap: {selected_cap}")
print(f"  - Decision Threshold: tau* = {frozen_threshold:.2f}")
print(f"  - Development Macro-F0.5: {frozen_dev_f05:.4f}")
print("=" * 80)

# ------------------------------------------------------------------------------
# 5. EVALUATE EXACTLY ONCE ON COMPLETELY UNTOUCHED FINAL S1 HOLDOUT
# ------------------------------------------------------------------------------
print("\n[4/5] Evaluating Frozen Configuration EXACTLY ONCE on Untouched Final Holdout...")

df_holdout_all = df_all[df_all["split_3way"] == "holdout"].copy()
df_holdout_cap = df_holdout_all[df_holdout_all["blocking_rank_order"] <= selected_cap].copy()

# Score holdout candidates
X_holdout = df_holdout_cap[list(FEATURE_NAMES)].values
t0_score = time.time()
df_holdout_cap["score"] = model_c1.predict_proba(X_holdout)
score_time = time.time() - t0_score

# Compute blocker recall on Holdout
holdout_true_matches = sum(len(gt_dict.get(s1, set())) for s1 in holdout_s1)
holdout_pos_retained = int(df_holdout_cap["match_label"].sum())
holdout_block_rec = holdout_pos_retained / holdout_true_matches * 100.0 if holdout_true_matches > 0 else 0

# Overall Holdout Evaluation at frozen threshold
holdout_eval = evaluate_predictions_df(
    df_holdout_cap,
    gt_dict,
    threshold=frozen_threshold,
    s1_list=holdout_s1,
)

# Diagnostic group evaluation
holdout_groups = evaluate_by_group(
    df_holdout_cap,
    gt_dict,
    threshold=frozen_threshold,
    s1_list=holdout_s1,
)

# Country-specific evaluation on Holdout
holdout_us_s1 = [s for s in holdout_s1 if s1_lookup[s]["country"] == "US"]
holdout_in_s1 = [s for s in holdout_s1 if s1_lookup[s]["country"] == "India"]

us_df = df_holdout_cap[df_holdout_cap["source1_entity_id"].isin(set(holdout_us_s1))]
in_df = df_holdout_cap[df_holdout_cap["source1_entity_id"].isin(set(holdout_in_s1))]

us_eval = evaluate_predictions_df(us_df, gt_dict, threshold=frozen_threshold, s1_list=holdout_us_s1)
in_eval = evaluate_predictions_df(in_df, gt_dict, threshold=frozen_threshold, s1_list=holdout_in_s1)

# Count zero-candidate S1 entities in holdout
holdout_s1_with_cands = set(df_holdout_cap["source1_entity_id"].unique())
zero_cand_holdout = len(set(holdout_s1) - holdout_s1_with_cands)

print("\n" + "=" * 80)
print("UNTOUCHED FINAL HOLDOUT VALIDATION RESULTS (EXACTLY ONCE)")
print("=" * 80)
print(f"  Total Holdout S1 Entities    : {len(holdout_s1):,}")
print(f"  Holdout Candidate Pairs      : {len(df_holdout_cap):,} (Avg {len(df_holdout_cap)/len(holdout_s1):.1f}/S1)")
print(f"  Holdout Blocker Recall       : {holdout_block_rec:.2f}% ({holdout_pos_retained:,} / {holdout_true_matches:,} true matches)")
print(f"  Holdout Zero-Candidate S1s   : {zero_cand_holdout} entities")
print(f"  Scoring Throughput           : {len(df_holdout_cap)/score_time:,.0f} pairs/sec ({score_time:.3f}s)")

print("\n--- PRIMARY COMPETITION METRICS ON UNTOUCHED HOLDOUT ---")
print(f"  >> Macro F0.5 : {holdout_eval['macro_f05']:.4f} <<")
print(f"  >> Macro Precision : {holdout_eval['macro_precision']:.4f} <<")
print(f"  >> Macro Recall    : {holdout_eval['macro_recall']:.4f} <<")
print(f"  Total Predicted Matches : {holdout_eval['total_pred_matches']:,}")
print(f"  Average Matches / S1    : {holdout_eval['avg_pred_matches']:.2f}")
print(f"  % S1 Predicted Empty    : {holdout_eval['empty_pred_pct']:.1f}%")

print("\n--- DIAGNOSTIC GROUP PERFORMANCE ON UNTOUCHED HOLDOUT ---")
for grp_label, grp_data in [
    ("Group 1: True Singletons", holdout_groups["group1_singleton"]),
    ("Group 2: True Zero-Match Entities", holdout_groups["group2_zero_match"]),
    ("Group 3: True Multi-Match Entities", holdout_groups["group3_multi_match"]),
]:
    print(f"  {grp_label} (N={grp_data['count']:,}):")
    print(f"    Macro F0.5: {grp_data['macro_f05']:.4f} | Precision: {grp_data['macro_precision']:.4f} | Recall: {grp_data['macro_recall']:.4f}")
    print(f"    Avg Matches: {grp_data['avg_pred_matches']:.2f} | False Positive Rate: {grp_data['false_positive_rate']:.2f}%")

print("\n--- COUNTRY BREAKDOWN ON UNTOUCHED HOLDOUT ---")
print(f"  US S1 Entities (N={len(holdout_us_s1):,}):")
print(f"    Macro F0.5: {us_eval['macro_f05']:.4f} | Precision: {us_eval['macro_precision']:.4f} | Recall: {us_eval['macro_recall']:.4f} | Avg Matches: {us_eval['avg_pred_matches']:.2f} | % Empty: {us_eval['empty_pred_pct']:.1f}%")
print(f"  India S1 Entities (N={len(holdout_in_s1):,}):")
print(f"    Macro F0.5: {in_eval['macro_f05']:.4f} | Precision: {in_eval['macro_precision']:.4f} | Recall: {in_eval['macro_recall']:.4f} | Avg Matches: {in_eval['avg_pred_matches']:.2f} | % Empty: {in_eval['empty_pred_pct']:.1f}%")

# ------------------------------------------------------------------------------
# 6. SIDE-BY-SIDE DEVELOPMENT vs FINAL HOLDOUT COMPARISON
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SIDE-BY-SIDE: DEVELOPMENT SET vs UNTOUCHED FINAL HOLDOUT")
print("=" * 80)

dev_win = best_dev_configs[selected_cap]["results"]

comparison_rows = [
    {
        "Metric": "Macro F0.5",
        "Development Set": round(dev_win["macro_f05"], 4),
        "Final Holdout Set": round(holdout_eval["macro_f05"], 4),
        "Delta (Holdout - Dev)": round(holdout_eval["macro_f05"] - dev_win["macro_f05"], 4),
    },
    {
        "Metric": "Macro Precision",
        "Development Set": round(dev_win["macro_precision"], 4),
        "Final Holdout Set": round(holdout_eval["macro_precision"], 4),
        "Delta (Holdout - Dev)": round(holdout_eval["macro_precision"] - dev_win["macro_precision"], 4),
    },
    {
        "Metric": "Macro Recall",
        "Development Set": round(dev_win["macro_recall"], 4),
        "Final Holdout Set": round(holdout_eval["macro_recall"], 4),
        "Delta (Holdout - Dev)": round(holdout_eval["macro_recall"] - dev_win["macro_recall"], 4),
    },
    {
        "Metric": "Avg Matches / S1",
        "Development Set": round(dev_win["avg_pred_matches"], 2),
        "Final Holdout Set": round(holdout_eval["avg_pred_matches"], 2),
        "Delta (Holdout - Dev)": round(holdout_eval["avg_pred_matches"] - dev_win["avg_pred_matches"], 2),
    },
    {
        "Metric": "% S1 Predicted Empty",
        "Development Set": round(dev_win["empty_pred_pct"], 1),
        "Final Holdout Set": round(holdout_eval["empty_pred_pct"], 1),
        "Delta (Holdout - Dev)": round(holdout_eval["empty_pred_pct"] - dev_win["empty_pred_pct"], 1),
    },
    {
        "Metric": "True Zero FP Rate",
        "Development Set": round(evaluate_by_group(df_dev_all[df_dev_all['blocking_rank_order']<=selected_cap], gt_dict, threshold=frozen_threshold, s1_list=dev_s1)["group2_zero_match"]["false_positive_rate"], 1),
        "Final Holdout Set": round(holdout_groups["group2_zero_match"]["false_positive_rate"], 1),
        "Delta (Holdout - Dev)": round(holdout_groups["group2_zero_match"]["false_positive_rate"] - evaluate_by_group(df_dev_all[df_dev_all['blocking_rank_order']<=selected_cap], gt_dict, threshold=frozen_threshold, s1_list=dev_s1)["group2_zero_match"]["false_positive_rate"], 1),
    },
]

df_comp = pd.DataFrame(comparison_rows)
print(df_comp.to_string(index=False))

# Also compare Cap 100 on Holdout for full transparency
df_holdout_cap100 = df_holdout_all[df_holdout_all["blocking_rank_order"] <= 100].copy()
X_h100 = df_holdout_cap100[list(FEATURE_NAMES)].values
df_holdout_cap100["score"] = model_c1.predict_proba(X_h100)
holdout_cap100_eval = evaluate_predictions_df(df_holdout_cap100, gt_dict, threshold=best_dev_configs[100]["threshold"], s1_list=holdout_s1)

print("\n--- HOLDOUT VERIFICATION OF CAP 100 vs CAP 150 ---")
print(f"  Holdout at Cap 100 (tau*={best_dev_configs[100]['threshold']:.2f}): Macro F0.5 = {holdout_cap100_eval['macro_f05']:.4f} (P={holdout_cap100_eval['macro_precision']:.4f}, R={holdout_cap100_eval['macro_recall']:.4f})")
print(f"  Holdout at Cap 150 (tau*={frozen_threshold:.2f}): Macro F0.5 = {holdout_eval['macro_f05']:.4f} (P={holdout_eval['macro_precision']:.4f}, R={holdout_eval['macro_recall']:.4f})")
print(f"  Holdout Cap 150 Gain: +{holdout_eval['macro_f05'] - holdout_cap100_eval['macro_f05']:.4f} Macro F0.5")

# Save summary artifacts
summary_payload = {
    "split_report": df_split_report.to_dict(orient="records"),
    "dev_cap_summary": df_dev_cap_summary.to_dict(orient="records"),
    "frozen_config": {
        "model": "Model C1 (XGBoost Mild Weight)",
        "scale_pos_weight": mild_scale_weight,
        "selected_cap": selected_cap,
        "frozen_threshold": frozen_threshold,
    },
    "holdout_overall": holdout_eval,
    "holdout_groups": holdout_groups,
    "holdout_us": us_eval,
    "holdout_india": in_eval,
    "comparison": comparison_rows,
}

with open(os.path.join(output_dir, "phase5_1_holdout_summary.pkl"), "wb") as f:
    pickle.dump(summary_payload, f)

print(f"\nSaved summary artifacts to: output/phase5_1_holdout_summary.pkl")
print("\n[SUCCESS] Phase 5.1 Untouched S1 Holdout Validation Complete.")
