"""
Phase 4 Benchmark Features Script
Amazon ML Challenge 2026: Business Entity Resolution

Benchmarks candidate generation, label attachment, S1-grouped splitting,
and pairwise feature extraction runtime & memory profiling.
"""

import sys
import os
import io
import time
import duckdb
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import CandidateBlocker
from src.features import PairwiseFeatureExtractor

base_train = "student_resource/dataset/train"
con = duckdb.connect()

print("=" * 80)
print("BENCHMARK: PAIRWISE FEATURE ENGINEERING RUNTIME & MEMORY PROFILING")
print("=" * 80)

# Setup sample
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
print(f"Blocking completed in {block_time:.2f}s: {block_res['total_candidates']:,} candidates generated.")

gt_rows = con.execute("SELECT s1_id, true_match_id FROM eval_gt").fetchall()
gt_set = set(gt_rows)

df_candidates = con.execute("SELECT * FROM eval_candidates").fetchdf()

extractor = PairwiseFeatureExtractor()
t0 = time.time()
df_labeled = extractor.attach_ground_truth_labels(df_candidates, gt_set)
df_split = extractor.assign_s1_group_split(df_labeled, val_ratio=0.20, seed=42)
split_time = time.time() - t0
print(f"Label attachment & S1-group split completed in {split_time:.2f}s.")

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
pairs_per_sec = len(df_features) / feat_time if feat_time > 0 else 0
mem_mb = df_features.memory_usage(deep=True).sum() / (1024 * 1024)

print(f"\nFeature Extraction Performance:")
print(f"  Total Candidate Pairs : {len(df_features):,}")
print(f"  Feature Dimensions    : {df_features.shape[1]} columns (2 IDs + {len(extractor.get_feature_names())} features)")
print(f"  Extraction Runtime    : {feat_time:.2f}s")
print(f"  Extraction Throughput : {pairs_per_sec:,.0f} pairs / second")
print(f"  RAM Usage             : {mem_mb:.2f} MB")
print("=" * 80)
