"""
Phase 5 Training & Validation Dataset Preparation
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Samples 5,000 representative S1 records (3,000 US, 2,000 India).
2. Generates candidate pairs with Priority Cap = 150 using CandidateBlocker.
3. Retrieves ground truth matches and attaches binary match labels (label=1 for true match, 0 for negative).
4. Performs S1-level grouped train/val split (80% train / 20% val, seed=42, 0 entity leakage).
5. Extracts all 65 pairwise features using PairwiseFeatureExtractor.
6. Saves the dataset to 'output/eval_features_cap150.parquet' and saves ground-truth mappings to 'output/eval_ground_truth.pkl'.
"""

import sys
import os
import io
import time
import pickle
import duckdb
import numpy as np
import pandas as pd

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import CandidateBlocker
from src.features import PairwiseFeatureExtractor

base_train = "student_resource/dataset/train"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

con = duckdb.connect()

print("=" * 80)
print("PHASE 5: DATASET PREPARATION & FEATURE EXTRACTION (CAP 150)")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. SAMPLE S1 AND CANDIDATES
# ------------------------------------------------------------------------------
print("\n[1/5] Sampling 5,000 S1 records (3k US, 2k India) and candidate tables...")

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

# Build complete ground-truth mapping for all 5,000 S1s (including zero-match S1s)
s1_list = [r[0] for r in con.execute("SELECT entity_id FROM eval_s1 ORDER BY entity_id").fetchall()]
gt_rows = con.execute("SELECT s1_id, true_match_id FROM eval_gt").fetchall()

gt_dict = {s1: set() for s1 in s1_list}
for s1_id, match_id in gt_rows:
    if match_id:
        gt_dict[s1_id].add(match_id)

gt_pairs_set = set(gt_rows)

print(f"  Total S1 Entities           : {len(s1_list):,}")
print(f"  Total Ground-Truth Pairs    : {len(gt_pairs_set):,}")
print(f"  Zero-Match S1 Entities      : {sum(1 for s in s1_list if len(gt_dict[s]) == 0):,}")
print(f"  Singleton S1 Entities       : {sum(1 for s in s1_list if len(gt_dict[s]) == 1):,}")
print(f"  Multi-Match S1 Entities     : {sum(1 for s in s1_list if len(gt_dict[s]) > 1):,}")

# ------------------------------------------------------------------------------
# 2. RUN BLOCKER UP TO CAP 150
# ------------------------------------------------------------------------------
print("\n[2/5] Running Multi-Channel Blocker with Priority Cap = 150...")
blocker = CandidateBlocker(con)
t0 = time.time()
block_res = blocker.generate_candidates(
    s1_table_or_path="eval_s1",
    cand_table_or_path="all_candidates",
    output_table="eval_candidates",
    max_candidates_per_s1=150
)
block_time = time.time() - t0
total_candidates = block_res["total_candidates"]
print(f"  Blocking Runtime : {block_time:.2f}s")
print(f"  Candidates Built : {total_candidates:,} candidate pairs across 5,000 S1 (avg {total_candidates/len(s1_list):.1f}/S1)")

df_candidates = con.execute("SELECT * FROM eval_candidates").fetchdf()

# ------------------------------------------------------------------------------
# 3. ATTACH GROUND-TRUTH LABELS & ASSIGN S1-GROUP SPLIT
# ------------------------------------------------------------------------------
print("\n[3/5] Attaching ground truth labels and S1-level grouped train/val split (80/20)...")
extractor = PairwiseFeatureExtractor()
df_labeled = extractor.attach_ground_truth_labels(df_candidates, gt_pairs_set)
df_split = extractor.assign_s1_group_split(df_labeled, val_ratio=0.20, seed=42)

train_mask = df_split["split_group"] == "train"
val_mask = df_split["split_group"] == "val"

train_s1 = df_split[train_mask]["source1_entity_id"].nunique()
val_s1 = df_split[val_mask]["source1_entity_id"].nunique()
train_pairs = train_mask.sum()
val_pairs = val_mask.sum()
train_pos = df_split[train_mask]["match_label"].sum()
val_pos = df_split[val_mask]["match_label"].sum()

print(f"  Train Split : {train_s1:,} S1 entities | {train_pairs:,} pairs | {train_pos:,} pos ({(train_pos/train_pairs)*100:.2f}%) | Neg/Pos: {(train_pairs-train_pos)/train_pos:.1f}:1")
print(f"  Val Split   : {val_s1:,} S1 entities | {val_pairs:,} pairs | {val_pos:,} pos ({(val_pos/val_pairs)*100:.2f}%) | Neg/Pos: {(val_pairs-val_pos)/val_pos:.1f}:1")

# ------------------------------------------------------------------------------
# 4. EXTRACT ALL 65 PAIRWISE FEATURES
# ------------------------------------------------------------------------------
print("\n[4/5] Extracting canonical 65 features...")
s1_rows = con.execute("SELECT entity_id, country, business_name, business_address FROM eval_s1").fetchall()
s1_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in s1_rows}

cand_rows = con.execute(f"""
SELECT entity_id, country, business_name, business_address 
FROM all_candidates 
WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM eval_candidates);
""").fetchall()
cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in cand_rows}

t_feat_start = time.time()
df_features = extractor.extract_features(df_split, s1_lookup, cand_lookup)
feat_time = time.time() - t_feat_start

# Re-attach match_label and split_group
df_features["match_label"] = df_split["match_label"].values
df_features["split_group"] = df_split["split_group"].values

throughput = len(df_features) / feat_time if feat_time > 0 else 0
print(f"  Extraction Runtime : {feat_time:.2f}s ({throughput:,.0f} pairs/sec)")
print(f"  Feature Matrix     : {df_features.shape[0]:,} rows x {df_features.shape[1]} columns")

# ------------------------------------------------------------------------------
# 5. SAVE DATASETS & LOOKUPS
# ------------------------------------------------------------------------------
print("\n[5/5] Saving feature parquet and lookup metadata...")
parquet_path = os.path.join(output_dir, "eval_features_cap150.parquet")
df_features.to_parquet(parquet_path, index=False)
print(f"  Saved features to: {parquet_path} ({os.path.getsize(parquet_path)/(1024*1024):.2f} MB)")

gt_path = os.path.join(output_dir, "eval_ground_truth.pkl")
with open(gt_path, "wb") as f:
    pickle.dump({
        "gt_dict": gt_dict,
        "gt_pairs_set": gt_pairs_set,
        "s1_list": s1_list,
        "s1_lookup": s1_lookup,
        "cand_lookup": cand_lookup,
    }, f)
print(f"  Saved metadata to: {gt_path}")

print("\n[SUCCESS] Dataset preparation complete.")
