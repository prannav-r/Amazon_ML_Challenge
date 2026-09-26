"""
Phase 4 Feature Distribution & Discriminative Separation Analysis Script
Amazon ML Challenge 2026: Business Entity Resolution
"""

import sys
import os
import io
import time
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import CandidateBlocker
from src.features import PairwiseFeatureExtractor

base_train = "student_resource/dataset/train"
con = duckdb.connect()

print("=" * 80)
print("FEATURE DISTRIBUTION & DISCRIMINATIVE SEPARATION ANALYSIS")
print("=" * 80)

# Setup 5,000 S1 sample
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
block_res = blocker.generate_candidates(
    s1_table_or_path="eval_s1",
    cand_table_or_path="all_candidates",
    output_table="eval_candidates",
    max_candidates_per_s1=100
)

gt_rows = con.execute("SELECT s1_id, true_match_id FROM eval_gt").fetchall()
gt_set = set(gt_rows)

df_candidates = con.execute("SELECT * FROM eval_candidates").fetchdf()

extractor = PairwiseFeatureExtractor()
df_labeled = extractor.attach_ground_truth_labels(df_candidates, gt_set)
df_split = extractor.assign_s1_group_split(df_labeled, val_ratio=0.20, seed=42)

s1_rows = con.execute("SELECT entity_id, country, business_name, business_address FROM eval_s1").fetchall()
s1_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in s1_rows}

cand_rows = con.execute(f"""
SELECT entity_id, country, business_name, business_address 
FROM all_candidates 
WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM eval_candidates);
""").fetchall()
cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in cand_rows}

df_features = extractor.extract_features(df_split, s1_lookup, cand_lookup)
df_features["match_label"] = df_split["match_label"].values

feature_cols = extractor.get_feature_names()
y = df_features["match_label"].values
pos_idx = (y == 1)
neg_idx = (y == 0)

stats_rows = []
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
    
    p_std = np.nanstd(pos_vals)
    n_std = np.nanstd(neg_vals)
    pooled_std = np.sqrt(0.5 * (p_std**2 + n_std**2)) if (p_std + n_std) > 0 else 1.0
    cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0.0
    
    try:
        if n_unique > 1:
            auc = roc_auc_score(y, vals)
            if auc < 0.5:
                auc = 1.0 - auc
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
print("\n--- ALL 65 FEATURES SORTED BY DISCRIMINATIVE AUC ---")
print(f"{'Feature':36s} | {'AUC':6s} | {'Cohen d':8s} | {'Pos Mean':9s} | {'Neg Mean':9s} | {'Missing %':9s}")
print("-" * 88)
for _, r in df_stats.sort_values(by="auc", ascending=False).iterrows():
    print(f"{r['feature']:36s} | {r['auc']:6.4f} | {r['cohens_d']:8.3f} | {r['pos_mean']:9.4f} | {r['neg_mean']:9.4f} | {r['missing_pct']:8.2f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
