"""
Verify Candidate Blocker and candidate_pairs.tsv Compliance
Amazon ML Challenge: Business Entity Resolution
"""

import duckdb
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add repo root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import CandidateBlocker

con = duckdb.connect()
blocker = CandidateBlocker(con)

base_train = "student_resource/dataset/train"

print("=" * 80)
print("VERIFYING REVISED CANDIDATE BLOCKER AND TSV INTEGRITY")
print("=" * 80)

# Setup 1,000 S1 sample for fast integration test
con.execute(f"""
CREATE TEMP TABLE test_s1 AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
LIMIT 1000;
""")

con.execute(f"""
CREATE TEMP TABLE test_cands AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN (SELECT DISTINCT country FROM test_s1)
UNION ALL
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN (SELECT DISTINCT country FROM test_s1);
""")

con.execute(f"""
CREATE TEMP TABLE test_gt AS
SELECT 
    g.source1_entity_id as s1_id,
    unnest(string_split(g.matched_entity_ids, ',')) as true_match_id
FROM read_csv('{base_train}/train_ground_truth.tsv', delim='\\t', header=true, quote='', all_varchar=true) g
JOIN test_s1 s ON g.source1_entity_id = s.entity_id
WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != '';
""")

total_gt = con.execute("SELECT count(*) FROM test_gt").fetchone()[0]
print(f"Sample S1: 1,000 | Ground Truth Pairs: {total_gt:,}")

# Run candidate generation with Cap 100
res = blocker.generate_candidates(
    s1_table_or_path="test_s1",
    cand_table_or_path="test_cands",
    output_table="verified_candidates",
    max_candidates_per_s1=100
)
print(f"Candidate generation completed in {res['runtime_seconds']:.2f}s: {res['total_candidates']:,} candidates generated.")

# Verify ground truth retained
hits = con.execute("""
SELECT count(*) FROM test_gt g
JOIN verified_candidates c ON g.s1_id = c.source1_entity_id AND g.true_match_id = c.candidate_entity_id
""").fetchone()[0]
print(f"True Match Hits: {hits:,} / {total_gt:,} ({hits/total_gt*100:.2f}%)")

# Test export
test_tsv = "output/test_candidate_pairs.tsv"
exp_res = blocker.export_candidate_pairs_tsv(
    candidate_table="verified_candidates",
    output_tsv_path=test_tsv,
    s1_table_or_path="test_s1",
    cand_table_or_path="test_cands"
)
print("\nExport Results:", exp_res)

# Read exported TSV and run verification assertions
exported_rows = con.execute(f"SELECT * FROM read_csv('{test_tsv}', delim='\\t', header=true, quote='', all_varchar=true)").fetchall()
print(f"Total exported rows read from TSV: {len(exported_rows):,}")

# 1. Check duplicate candidate pairs
dup_count = con.execute(f"""
SELECT count(*) FROM (
    SELECT source1_entity_id, candidate_entity_id, count(*)
    FROM read_csv('{test_tsv}', delim='\\t', header=true, quote='', all_varchar=true)
    GROUP BY source1_entity_id, candidate_entity_id
    HAVING count(*) > 1
)
""").fetchone()[0]
assert dup_count == 0, f"Found {dup_count} duplicate candidate pairs in TSV!"
print("[PASSED] Zero duplicate pairs in TSV.")

# 2. Check S1 IDs format
invalid_s1 = con.execute(f"""
SELECT count(*) FROM read_csv('{test_tsv}', delim='\\t', header=true, quote='', all_varchar=true)
WHERE source1_entity_id NOT LIKE 'S1-%'
""").fetchone()[0]
assert invalid_s1 == 0, f"Found {invalid_s1} invalid S1 IDs!"
print("[PASSED] All S1 IDs strictly follow 'S1-*' format.")

# 3. Check candidate IDs format (only S2 or S3)
invalid_cand = con.execute(f"""
SELECT count(*) FROM read_csv('{test_tsv}', delim='\\t', header=true, quote='', all_varchar=true)
WHERE candidate_entity_id NOT LIKE 'S2-%' AND candidate_entity_id NOT LIKE 'S3-%'
""").fetchone()[0]
assert invalid_cand == 0, f"Found {invalid_cand} non-S2/S3 candidate IDs!"
print("[PASSED] All candidate IDs are strictly from S2 or S3.")

# 4. Check country equality constraint
mismatched_countries = con.execute(f"""
WITH tsv_data AS (
    SELECT source1_entity_id as s1_id, candidate_entity_id as c_id
    FROM read_csv('{test_tsv}', delim='\\t', header=true, quote='', all_varchar=true)
)
SELECT count(*)
FROM tsv_data t
JOIN test_s1 s ON t.s1_id = s.entity_id
JOIN test_cands c ON t.c_id = c.entity_id
WHERE s.country != c.country;
""").fetchone()[0]
assert mismatched_countries == 0, f"Found {mismatched_countries} country-mismatched candidate pairs!"
print("[PASSED] 100% of candidate pairs satisfy candidate.country == s1.country.")

# 5. Check all retained blocker pairs are present in TSV
retained_in_blocker = con.execute("SELECT count(*) FROM verified_candidates").fetchone()[0]
retained_in_tsv = len(exported_rows)
assert retained_in_tsv == retained_in_blocker, f"Mismatch: Blocker has {retained_in_blocker}, TSV has {retained_in_tsv}"
print(f"[PASSED] Exactly 100% of retained blocker candidates ({retained_in_tsv:,}) are present in TSV.")

# Clean up test output
if os.path.exists(test_tsv):
    os.remove(test_tsv)

print("\n" + "=" * 80)
print("ALL CANDIDATE_PAIRS.TSV INTEGRITY CHECKS PASSED PERFECTLY")
print("=" * 80)
