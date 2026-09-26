"""
Phase 4 Feature Engineering Benchmark, Distribution Analysis & Diagnostics
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Generates candidates on a representative evaluation dataset (5,000 S1: 3k US, 2k India)
   using the approved Cap 100 blocker.
2. Attaches ground-truth matching labels (label=1 for true matches, 0 for negatives).
3. Performs S1-level grouped train/validation splitting (80% train / 20% val).
4. Extracts all 65 pairwise features with runtime and memory profiling.
5. Computes complete feature statistics (type, missing rate, min, max, mean, median, unique values,
   positive-class mean, negative-class mean).
6. Computes individual feature discriminative power (AUC and Cohen's d separation).
7. Analyzes feature correlations and redundancy.
8. Analyzes performance across subsets (Source 2 vs Source 3, Native-script vs Latin).
"""

import sys
import os
import io
import time
import duckdb
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import CandidateBlocker
from src.features import PairwiseFeatureExtractor

base_train = "student_resource/dataset/train"
con = duckdb.connect()

print("=" * 80)
print("PHASE 4: PAIRWISE FEATURE ENGINEERING BENCHMARK & ANALYSIS")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. GENERATE CANDIDATE DATASET USING APPROVED BLOCKER (CAP 100)
# ------------------------------------------------------------------------------
print("\n[1/6] Sampling 5,000 S1 records and generating candidate pairs (Cap 100)...")

con.execute(f"""
CREATE TEMP TABLE eval_s1 AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'US'
LIMIT 3000;
""")

con.execute(f"""
INSERT INTO eval_s1
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'India'
LIMIT 2000;
""")

con.execute(f"""
CREATE TEMP TABLE all_candidates AS
SELECT entity_id, business_name, business_address, country, 'S2' as src
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN (SELECT DISTINCT country FROM eval_s1)
UNION ALL
SELECT entity_id, business_name, business_address, country, 'S3' as src
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN (SELECT DISTINCT country FROM eval_s1);
""")

con.execute(f"""
CREATE TEMP TABLE eval_gt AS
SELECT 
    g.source1_entity_id as s1_id,
    unnest(string_split(g.matched_entity_ids, ',')) as true_match_id
FROM read_csv('{base_train}/train_ground_truth.tsv', delim='\\t', header=true, quote='', all_varchar=true) g
JOIN eval_s1 s ON g.source1_entity_id = s.entity_id
WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != '';
""")

blocker = CandidateBlocker(con)
t0 = time.time()
block_res = blocker.generate_candidates(
    s1_table_or_path="eval_s1",
    cand_table_or_path="all_candidates",
    output_table="eval_candidates",
    max_candidates_per_s1=100
)
block_time = time.time() - t0

total_pairs = block_res["total_candidates"]
print(f"  Blocking Runtime : {block_time:.2f}s")
print(f"  Candidates Built : {total_pairs:,} candidate pairs across 5,000 S1")

# Fetch ground-truth set
gt_rows = con.execute("SELECT s1_id, true_match_id FROM eval_gt").fetchall()
gt_set = set(gt_rows)
total_gt_available = len(gt_set)

# Retrieve candidate pairs with provenance
df_candidates = con.execute("SELECT * FROM eval_candidates").fetchdf()

# ------------------------------------------------------------------------------
# 2. ATTACH GROUND TRUTH LABELS & ANALYZE CLASS IMBALANCE
# ------------------------------------------------------------------------------
print("\n[2/6] Attaching ground truth labels and analyzing negative/positive ratio...")
extractor = PairwiseFeatureExtractor()
df_labeled = extractor.attach_ground_truth_labels(df_candidates, gt_set)

pos_count = int(df_labeled["match_label"].sum())
neg_count = len(df_labeled) - pos_count
pos_rate = (pos_count / len(df_labeled)) * 100.0
neg_pos_ratio = neg_count / pos_count if pos_count > 0 else 0

print(f"  Total Candidate Pairs : {len(df_labeled):,}")
print(f"  Positive Pairs (y=1)  : {pos_count:,} ({pos_rate:.2f}%)")
print(f"  Negative Pairs (y=0)  : {neg_count:,} ({100.0 - pos_rate:.2f}%)")
print(f"  Negative/Positive Ratio: {neg_pos_ratio:.1f} : 1")
print(f"  True Matches Retained  : {pos_count:,} / {total_gt_available:,} ({(pos_count/total_gt_available)*100:.2f}% Blocker Recall)")

# Per-S1 candidate distribution
s1_cands = df_labeled.groupby("source1_entity_id").size()
s1_pos = df_labeled.groupby("source1_entity_id")["match_label"].sum()
s1_singletons = (s1_pos == 1).sum()
s1_multi = (s1_pos > 1).sum()
s1_zero = (s1_pos == 0).sum()

print(f"  Candidates per S1      : Mean={s1_cands.mean():.1f}, Median={s1_cands.median():.1f}, P95={np.percentile(s1_cands, 95):.1f}, Max={s1_cands.max()}")
print(f"  S1 Match Breakdown     : Zero={s1_zero:,}, Singleton (1 match)={s1_singletons:,}, Multi-match (>1)={s1_multi:,}")

# ------------------------------------------------------------------------------
# 3. S1-LEVEL GROUPED TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------------------------
print("\n[3/6] Performing S1-level grouped train/val split (80% train / 20% val)...")
df_split = extractor.assign_s1_group_split(df_labeled, val_ratio=0.20, seed=42)

train_mask = df_split["split_group"] == "train"
val_mask = df_split["split_group"] == "val"

train_s1 = df_split[train_mask]["source1_entity_id"].nunique()
val_s1 = df_split[val_mask]["source1_entity_id"].nunique()
train_pairs = train_mask.sum()
val_pairs = val_mask.sum()
train_pos = df_split[train_mask]["match_label"].sum()
val_pos = df_split[val_mask]["match_label"].sum()

print(f"  Train Split : {train_s1:,} S1 entities | {train_pairs:,} candidate pairs | {train_pos:,} positives ({train_pos/train_pairs*100:.2f}%)")
print(f"  Val Split   : {val_s1:,} S1 entities | {val_pairs:,} candidate pairs | {val_pos:,} positives ({val_pos/val_pairs*100:.2f}%)")

# Verify zero leakage
leakage_check = set(df_split[train_mask]["source1_entity_id"]) & set(df_split[val_mask]["source1_entity_id"])
assert len(leakage_check) == 0, "S1 leakage detected across train and val!"
print("  [PASSED] Strict zero-leakage guarantee verified (0 shared S1 entities).")

# ------------------------------------------------------------------------------
# 4. EXTRACT ALL 65 PAIRWISE FEATURES & PROFILE PERFORMANCE
# ------------------------------------------------------------------------------
print("\n[4/6] Extracting all 65 pairwise features for candidate pairs...")

# Build fast dictionary lookup for S1 and Candidates
s1_rows = con.execute("SELECT entity_id, country, business_name, business_address FROM eval_s1").fetchall()
s1_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in s1_rows}

# Only fetch candidate records that appear in candidate pairs to optimize RAM
cand_ids_needed = tuple(df_split["candidate_entity_id"].unique())
cand_rows = con.execute(f"""
SELECT entity_id, country, business_name, business_address 
FROM all_candidates 
WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM eval_candidates);
""").fetchall()
cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in cand_rows}

t_feat_start = time.time()
df_features = extractor.extract_features(df_split, s1_lookup, cand_lookup)
feat_time = time.time() - t_feat_start
pairs_per_sec = len(df_features) / feat_time if feat_time > 0 else 0
mem_mb = df_features.memory_usage(deep=True).sum() / (1024 * 1024)

print(f"  Feature Extraction Time : {feat_time:.2f}s ({pairs_per_sec:,.0f} pairs/sec)")
print(f"  Feature Matrix Memory   : {mem_mb:.2f} MB ({len(df_features):,} rows x {df_features.shape[1]} cols)")
print(f"  Total Features Extracted: {len(extractor.get_feature_names())} features")

# Append match label and split group to feature matrix for analysis
df_features["match_label"] = df_split["match_label"].values
df_features["split_group"] = df_split["split_group"].values

# ------------------------------------------------------------------------------
# 5. FEATURE DISTRIBUTION & DISCRIMINATIVE POWER ANALYSIS
# ------------------------------------------------------------------------------
print("\n[5/6] Computing feature distribution statistics and positive vs negative separation...")

feature_cols = extractor.get_feature_names()
y = df_features["match_label"].values
pos_idx = (y == 1)
neg_idx = (y == 0)

stats_rows = []

from sklearn.metrics import roc_auc_score

for col in feature_cols:
    vals = df_features[col].values
    dtype = str(vals.dtype)
    missing_rate = np.isnan(vals).mean() * 100.0
    
    val_min = float(np.nanmin(vals))
    val_max = float(np.nanmax(vals))
    val_mean = float(np.nanmean(vals))
    val_med = float(np.nanmedian(vals))
    n_unique = len(np.unique(vals[~np.isnan(vals)]))
    
    pos_vals = vals[pos_idx]
    neg_vals = vals[neg_idx]
    
    pos_mean = float(np.nanmean(pos_vals)) if len(pos_vals) > 0 else 0.0
    pos_med = float(np.nanmedian(pos_vals)) if len(pos_vals) > 0 else 0.0
    neg_mean = float(np.nanmean(neg_vals)) if len(neg_vals) > 0 else 0.0
    neg_med = float(np.nanmedian(neg_vals)) if len(neg_vals) > 0 else 0.0
    
    # Separation: Cohen's d = (pos_mean - neg_mean) / pooled_std
    p_std = np.nanstd(pos_vals)
    n_std = np.nanstd(neg_vals)
    pooled_std = np.sqrt(0.5 * (p_std**2 + n_std**2)) if (p_std + n_std) > 0 else 1.0
    cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0.0
    
    # Individual feature AUC
    try:
        if n_unique > 1:
            auc = roc_auc_score(y, vals)
            if auc < 0.5:
                auc = 1.0 - auc  # Align orientation
        else:
            auc = 0.5
    except Exception:
        auc = 0.5
        
    stats_rows.append({
        "feature": col,
        "type": dtype,
        "missing_pct": missing_rate,
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
        "median": val_med,
        "unique": n_unique,
        "pos_mean": pos_mean,
        "pos_med": pos_med,
        "neg_mean": neg_mean,
        "neg_med": neg_med,
        "cohens_d": cohens_d,
        "auc": auc
    })

df_stats = pd.DataFrame(stats_rows)

print("\n--- TOP 25 MOST DISCRIMINATIVE INDIVIDUAL FEATURES (Ranked by AUC) ---")
print(f"{'Feature Name':36s} | {'AUC':6s} | {'Cohen d':8s} | {'Pos Mean':9s} | {'Neg Mean':9s} | {'Unique':6s}")
print("-" * 88)
top_features = df_stats.sort_values(by="auc", ascending=False).head(25)
for _, r in top_features.iterrows():
    print(f"{r['feature']:36s} | {r['auc']:6.4f} | {r['cohens_d']:8.3f} | {r['pos_mean']:9.4f} | {r['neg_mean']:9.4f} | {int(r['unique']):6d}")

# ------------------------------------------------------------------------------
# 6. CORRELATION & REDUNDANCY AUDIT
# ------------------------------------------------------------------------------
print("\n[6/6] Auditing feature correlations and collinearity...")
corr_matrix = df_features[feature_cols].corr().abs()

# Find feature pairs with Pearson r > 0.85
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        c1 = feature_cols[i]
        c2 = feature_cols[j]
        r_val = corr_matrix.loc[c1, c2]
        if r_val >= 0.85:
            high_corr_pairs.append((c1, c2, r_val))

high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
print(f"\nTotal highly correlated pairs (r >= 0.85): {len(high_corr_pairs)}")
print(f"{'Feature 1':36s} <--> {'Feature 2':36s} | {'Correlation':11s}")
print("-" * 92)
for f1, f2, r_val in high_corr_pairs[:15]:
    print(f"{f1:36s} <--> {f2:36s} | {r_val:11.4f}")

# Subgroup Diagnostic: Source 2 vs Source 3 performance
print("\n--- Diagnostic: Discriminative Power across Source 2 vs Source 3 ---")
is_s2 = df_features["candidate_is_source2"] == 1.0
is_s3 = df_features["candidate_is_source3"] == 1.0

auc_s2_name = roc_auc_score(y[is_s2], df_features.loc[is_s2, "name_levenshtein_sim"])
auc_s3_name = roc_auc_score(y[is_s3], df_features.loc[is_s3, "name_levenshtein_sim"])
auc_s2_addr = roc_auc_score(y[is_s2], df_features.loc[is_s2, "address_token_jaccard"])
auc_s3_addr = roc_auc_score(y[is_s3], df_features.loc[is_s3, "address_token_jaccard"])

print(f"  Source 2: Name Levenshtein AUC = {auc_s2_name:.4f} | Address Token Jaccard AUC = {auc_s2_addr:.4f}")
print(f"  Source 3: Name Levenshtein AUC = {auc_s3_name:.4f} | Address Token Jaccard AUC = {auc_s3_addr:.4f}")

# Cross-script diagnostic
is_cross = df_features["name_cross_script"] == 1.0
cross_pos = int(df_features.loc[is_cross, "match_label"].sum())
cross_tot = int(is_cross.sum())
print(f"\n--- Diagnostic: Cross-Script Pairs in Candidate Set ---")
print(f"  Total Cross-Script Pairs : {cross_tot:,}")
print(f"  True Matches in Subgroup : {cross_pos:,} ({cross_pos/cross_tot*100:.2f}%)")
if cross_tot > 0 and cross_pos > 0:
    auc_cross_addr = roc_auc_score(y[is_cross], df_features.loc[is_cross, "address_token_jaccard"])
    auc_cross_num = roc_auc_score(y[is_cross], df_features.loc[is_cross, "address_building_num_match"])
    print(f"  Cross-Script Address Jaccard AUC = {auc_cross_addr:.4f} | Building Number AUC = {auc_cross_num:.4f}")

print("\n" + "=" * 80)
print("PHASE 4 BENCHMARK & DISTRIBUTION ANALYSIS COMPLETE")
print("=" * 80)
