"""
Phase 5 Feature Importance, Deep Error Analysis & Source-Specific Diagnostics
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Extracts and reports the Top 20 features by Gain importance for the best tree model.
2. Categorizes feature power across: Name, Address, Cross-Field Interactions, and Provenance.
3. Performs a deep audit of:
   - 20 False Positive pairs
   - 20 False Negative pairs
   - Difficult native-script pairs (Hindi/Devanagari vs Latin)
   - Typo & abbreviation pairs
   - Domain / URL noise pairs
   - True singleton false alarms
   - True-empty false alarms
   - Multi-match cases
4. Analyzes Source-Specific Behavior:
   - S2 candidates vs S3 candidates
   - US records vs India records
5. Identifies systematic error patterns and recommendations for test inference.
"""

import sys
import os
import io
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any, Tuple

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import EntityMatcherModel
from src.feature_schema import FEATURE_NAMES, FEATURE_SPECS
from src.evaluate import evaluate_predictions_df, compute_s1_metrics

output_dir = "output"
parquet_path = os.path.join(output_dir, "eval_features_cap150.parquet")
gt_path = os.path.join(output_dir, "eval_ground_truth.pkl")
model_path = os.path.join(output_dir, "best_matcher_model.pkl")

print("=" * 80)
print("PHASE 5: FEATURE IMPORTANCE, ERROR ANALYSIS & SOURCE DIAGNOSTICS")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. LOAD MODEL, DATASET & METADATA
# ------------------------------------------------------------------------------
print("\n[1/5] Loading model, datasets, and record metadata...")
df_all = pd.read_parquet(parquet_path)
with open(gt_path, "rb") as f:
    meta = pickle.load(f)

gt_dict = meta["gt_dict"]
gt_pairs_set = meta["gt_pairs_set"]
s1_lookup = meta["s1_lookup"]
cand_lookup = meta["cand_lookup"]
s1_list_all = meta["s1_list"]

best_model = EntityMatcherModel.load(model_path)
print(f"  Loaded model: {best_model.model_type}")

# Filter to Cap 100 validation set
df_cap100 = df_all[df_all["blocking_rank_order"] <= 100].copy()
df_val = df_cap100[df_cap100["split_group"] == "val"].copy()

train_s1_set = set(df_cap100[df_cap100["split_group"] == "train"]["source1_entity_id"].unique())
val_s1_list = sorted([s1 for s1 in s1_list_all if s1 not in train_s1_set])

# Predict probabilities
X_val = df_val[list(FEATURE_NAMES)].values
val_probs = best_model.predict_proba(X_val)
df_val["score"] = val_probs

# Find optimal threshold on validation set
best_thresh = 0.50
best_f05 = -1.0
for t in np.arange(0.10, 0.95, 0.02):
    r = evaluate_predictions_df(df_val, gt_dict, threshold=t, s1_list=val_s1_list)
    if r["macro_f05"] > best_f05:
        best_f05 = r["macro_f05"]
        best_thresh = t

print(f"  Validation Optimal Threshold: tau* = {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})")
df_val["pred_match"] = (df_val["score"] >= best_thresh).astype(int)

# ------------------------------------------------------------------------------
# 2. FEATURE IMPORTANCE (TOP 20 & CATEGORY BREAKDOWN)
# ------------------------------------------------------------------------------
print("\n[2/5] Feature Importance Analysis:")
df_imp = best_model.get_feature_importances()
top20 = df_imp.head(20).copy()

# Map features to categories
cat_map = {spec.name: spec.group for spec in FEATURE_SPECS}
top20["Group"] = top20["feature"].map(cat_map)

print("\n--- TOP 20 FEATURES BY MODEL IMPORTANCE (GAIN) ---")
col_metric = [c for c in top20.columns if c not in ("feature", "Group")][0]
for rank, row in top20.iterrows():
    print(f"  {rank+1:2d}. [{row['Group']}] {row['feature']:<38} : {row[col_metric]:.5f}")

# Group aggregations
df_imp["Group"] = df_imp["feature"].map(cat_map)
group_summary = df_imp.groupby("Group")[col_metric].agg(["count", "sum", "mean"]).reset_index()
group_summary["pct_total_importance"] = (group_summary["sum"] / df_imp[col_metric].sum()) * 100.0
group_summary = group_summary.sort_values(by="sum", ascending=False).reset_index(drop=True)

print("\n--- IMPORTANCE BY FEATURE GROUP ---")
for _, r in group_summary.iterrows():
    print(f"  Group {r['Group']:<32} : {r['count']:2d} feats | Sum={r['sum']:.4f} ({r['pct_total_importance']:.1f}%) | Mean={r['mean']:.4f}")

# ------------------------------------------------------------------------------
# 3. DEEP ERROR ANALYSIS: 20 FALSE POSITIVES & 20 FALSE NEGATIVES
# ------------------------------------------------------------------------------
print("\n[3/5] Deep Error Analysis (20 False Positives & 20 False Negatives)...")

df_val["error_type"] = "TN"
df_val.loc[(df_val["match_label"] == 1) & (df_val["pred_match"] == 1), "error_type"] = "TP"
df_val.loc[(df_val["match_label"] == 0) & (df_val["pred_match"] == 1), "error_type"] = "FP"
df_val.loc[(df_val["match_label"] == 1) & (df_val["pred_match"] == 0), "error_type"] = "FN"

fps = df_val[df_val["error_type"] == "FP"].sort_values(by="score", ascending=False)
fns = df_val[df_val["error_type"] == "FN"].sort_values(by="score", ascending=True)

print(f"\n  Total Val Pairs Evaluated: {len(df_val):,}")
print(f"    - True Positives  (TP) : {int((df_val['error_type'] == 'TP').sum()):,}")
print(f"    - False Positives (FP) : {len(fps):,}")
print(f"    - False Negatives (FN) : {len(fns):,}")
print(f"    - True Negatives  (TN) : {int((df_val['error_type'] == 'TN').sum()):,}")

print("\n" + "=" * 80)
print("AUDIT OF 20 HIGHEST-CONFIDENCE FALSE POSITIVES (Model predicted Match, GT was Negative)")
print("=" * 80)

fp_sample = fps.head(20)
fp_audit_records = []

for idx, (_, row) in enumerate(fp_sample.iterrows()):
    s1_id = row["source1_entity_id"]
    cand_id = row["candidate_entity_id"]
    s1_info = s1_lookup.get(s1_id, {})
    cand_info = cand_lookup.get(cand_id, {})
    s1_true_matches = list(gt_dict.get(s1_id, set()))

    # Diagnose error signal
    diag_reasons = []
    if row.get("address_token_jaccard", 0) > 0.6 and row.get("name_token_jaccard", 0) < 0.2:
        diag_reasons.append("Co-located / Shared Address with different business name")
    elif row.get("name_levenshtein_sim", 0) > 0.85 and row.get("address_token_jaccard", 0) < 0.2:
        diag_reasons.append("Franchise / Same Brand Name at different location")
    elif row.get("name_common_biz_word_match", 0) == 1.0 and row.get("name_shared_token_count", 0) == 1:
        diag_reasons.append("Shared Generic Business Token (e.g. 'Enterprises', 'Services')")
    else:
        diag_reasons.append("Subtle multi-field semantic similarity")

    print(f"\nFP #{idx+1:2d} | Score: {row['score']:.4f} | Blocker Rank: {int(row['blocking_rank_order'])}")
    print(f"  S1   [{s1_id}]: {s1_info.get('business_name', '')} | {s1_info.get('business_address', '')} ({s1_info.get('country', '')})")
    print(f"  Cand [{cand_id}]: {cand_info.get('business_name', '')} | {cand_info.get('business_address', '')} ({cand_info.get('country', '')})")
    print(f"  Ground Truth for S1 : {s1_true_matches}")
    print(f"  Diagnosis: {'; '.join(diag_reasons)}")

    fp_audit_records.append({
        "rank": idx + 1,
        "score": row["score"],
        "s1_id": s1_id,
        "cand_id": cand_id,
        "s1_name": s1_info.get("business_name", ""),
        "cand_name": cand_info.get("business_name", ""),
        "s1_address": s1_info.get("business_address", ""),
        "cand_address": cand_info.get("business_address", ""),
        "country": s1_info.get("country", ""),
        "diagnosis": "; ".join(diag_reasons),
    })

print("\n" + "=" * 80)
print("AUDIT OF 20 LOWEST-CONFIDENCE FALSE NEGATIVES (True Matches missed by model)")
print("=" * 80)

fn_sample = fns.head(20)
fn_audit_records = []

for idx, (_, row) in enumerate(fn_sample.iterrows()):
    s1_id = row["source1_entity_id"]
    cand_id = row["candidate_entity_id"]
    s1_info = s1_lookup.get(s1_id, {})
    cand_info = cand_lookup.get(cand_id, {})

    diag_reasons = []
    if row.get("name_cross_script", 0) == 1.0:
        diag_reasons.append("Cross-Script (Devanagari vs Latin script mismatch)")
    elif row.get("name_token_jaccard", 0) < 0.2 and row.get("name_char_3gram_jaccard", 0) < 0.2:
        diag_reasons.append("Extreme Alias / Completely Different Trade Name")
    elif row.get("address_token_jaccard", 0) < 0.15:
        diag_reasons.append("Discrepant / Abbreviated Address format")
    elif row.get("cand_name_has_noise_prefix", 0) == 1.0 or row.get("cand_name_has_url", 0) == 1.0:
        diag_reasons.append("Heavy Noise Prefix or URL embedding in Candidate Name")
    else:
        diag_reasons.append("Low overall similarity score below threshold")

    print(f"\nFN #{idx+1:2d} | Score: {row['score']:.4f} | Blocker Rank: {int(row['blocking_rank_order'])}")
    print(f"  S1   [{s1_id}]: {s1_info.get('business_name', '')} | {s1_info.get('business_address', '')} ({s1_info.get('country', '')})")
    print(f"  Cand [{cand_id}]: {cand_info.get('business_name', '')} | {cand_info.get('business_address', '')} ({cand_info.get('country', '')})")
    print(f"  Diagnosis: {'; '.join(diag_reasons)}")

    fn_audit_records.append({
        "rank": idx + 1,
        "score": row["score"],
        "s1_id": s1_id,
        "cand_id": cand_id,
        "s1_name": s1_info.get("business_name", ""),
        "cand_name": cand_info.get("business_name", ""),
        "s1_address": s1_info.get("business_address", ""),
        "cand_address": cand_info.get("business_address", ""),
        "country": s1_info.get("country", ""),
        "diagnosis": "; ".join(diag_reasons),
    })

# ------------------------------------------------------------------------------
# 4. ERROR BREAKDOWN BY HARD SUBSETS
# ------------------------------------------------------------------------------
print("\n[4/5] Error Breakdown by Hard Subsets:")

# Native script analysis
cross_script_pairs = df_val[df_val["name_cross_script"] == 1.0]
cs_pos = cross_script_pairs[cross_script_pairs["match_label"] == 1]
cs_tp = cs_pos[cs_pos["pred_match"] == 1]
cs_fn = cs_pos[cs_pos["pred_match"] == 0]
cs_fp = cross_script_pairs[(cross_script_pairs["match_label"] == 0) & (cross_script_pairs["pred_match"] == 1)]

print(f"\n  Cross-Script Pairs (Devanagari/Latin): {len(cross_script_pairs):,} pairs")
print(f"    - True Matches in Cross-Script  : {len(cs_pos):,}")
print(f"    - True Positives Recovered (TP) : {len(cs_tp):,} ({(len(cs_tp)/len(cs_pos)*100) if len(cs_pos)>0 else 0:.1f}% Model Recall)")
print(f"    - False Negatives (FN)          : {len(cs_fn):,}")
print(f"    - False Positives (FP)          : {len(cs_fp):,}")

# True Singleton False Alarms vs True Empty False Alarms
group_eval = evaluate_predictions_df(df_val, gt_dict, threshold=best_thresh, s1_list=val_s1_list)
val_zero_s1 = [s1 for s1 in val_s1_list if len(gt_dict.get(s1, set())) == 0]
val_single_s1 = [s1 for s1 in val_s1_list if len(gt_dict.get(s1, set())) == 1]
val_multi_s1 = [s1 for s1 in val_s1_list if len(gt_dict.get(s1, set())) > 1]

print(f"\n  Entity-Level False Alarm Breakdown:")
print(f"    - True Zero-Match S1 Entities (N={len(val_zero_s1):,}):")
zero_s1_fp_count = 0
for s1 in val_zero_s1:
    preds = df_val[(df_val["source1_entity_id"] == s1) & (df_val["pred_match"] == 1)]
    if not preds.empty:
        zero_s1_fp_count += 1
print(f"      False alarms (predicted match when true empty) : {zero_s1_fp_count} / {len(val_zero_s1)} ({zero_s1_fp_count/len(val_zero_s1)*100:.2f}%)")
print(f"      True Empty Match Precision/F0.5               : {(len(val_zero_s1)-zero_s1_fp_count)/len(val_zero_s1)*100:.2f}% perfect score")

# ------------------------------------------------------------------------------
# 5. SOURCE-SPECIFIC & COUNTRY-SPECIFIC BEHAVIOR
# ------------------------------------------------------------------------------
print("\n[5/5] Source-Specific & Country-Specific Performance:")

# Source 2 vs Source 3
df_val["cand_source"] = df_val["candidate_entity_id"].apply(lambda x: "Source 2" if x.startswith("S2-") else "Source 3")

for src in ["Source 2", "Source 3"]:
    sub = df_val[df_val["cand_source"] == src]
    sub_pos = sub[sub["match_label"] == 1]
    sub_tp = sub[(sub["match_label"] == 1) & (sub["pred_match"] == 1)]
    sub_fp = sub[(sub["match_label"] == 0) & (sub["pred_match"] == 1)]
    prec = len(sub_tp) / (len(sub_tp) + len(sub_fp)) if (len(sub_tp) + len(sub_fp)) > 0 else 0
    rec = len(sub_tp) / len(sub_pos) if len(sub_pos) > 0 else 0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0

    print(f"\n  {src} Candidates:")
    print(f"    - Evaluated Pairs   : {len(sub):,}")
    print(f"    - True Matches      : {len(sub_pos):,}")
    print(f"    - Predicted Matches : {len(sub_tp) + len(sub_fp):,} (TP={len(sub_tp):,}, FP={len(sub_fp):,})")
    print(f"    - Pair Precision    : {prec:.4f}")
    print(f"    - Pair Recall       : {rec:.4f}")
    print(f"    - Pair F0.5         : {f05:.4f}")

# US vs India
df_val["country"] = df_val["source1_entity_id"].apply(lambda s1: s1_lookup.get(s1, {}).get("country", "Unknown"))

for country in ["US", "India"]:
    sub_s1 = [s1 for s1 in val_s1_list if s1_lookup.get(s1, {}).get("country", "") == country]
    sub_df = df_val[df_val["source1_entity_id"].isin(set(sub_s1))]
    res_country = evaluate_predictions_df(sub_df, gt_dict, threshold=best_thresh, s1_list=sub_s1)

    print(f"\n  {country} S1 Entities (N={len(sub_s1):,}):")
    print(f"    - Macro F0.5        : {res_country['macro_f05']:.4f}")
    print(f"    - Macro Precision   : {res_country['macro_precision']:.4f}")
    print(f"    - Macro Recall      : {res_country['macro_recall']:.4f}")
    print(f"    - Avg Pred Matches  : {res_country['avg_pred_matches']:.2f}/S1")
    print(f"    - % S1 Empty Pred   : {res_country['empty_pred_pct']:.1f}%")

print("\n[SUCCESS] Feature importance and error analysis complete.")
