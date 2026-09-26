import duckdb
import sys
import io
import time
import re
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Benchmarking Candidate Generation / Blocking Rules ---")

# Let's take 2,000 Source 1 entities with their ground truth matches
# Sample 1,000 US and 1,000 India entities
con.execute(f"""
CREATE TEMP TABLE eval_s1 AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'US'
LIMIT 1000;
""")

con.execute(f"""
INSERT INTO eval_s1
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'India'
LIMIT 1000;
""")

# Get ground truth for these 2000 entities
con.execute(f"""
CREATE TEMP TABLE eval_gt AS
SELECT 
    g.source1_entity_id as s1_id,
    unnest(string_split(g.matched_entity_ids, ',')) as matched_id
FROM read_csv('{base_train}/train_ground_truth.tsv', delim='\\t', header=true, quote='', all_varchar=true) g
JOIN eval_s1 s ON g.source1_entity_id = s.entity_id
WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != '';
""")

total_true_pairs = con.execute("SELECT count(*) FROM eval_gt").fetchone()[0]
total_eval_s1 = con.execute("SELECT count(*) FROM eval_s1").fetchone()[0]
print(f"Evaluation set: {total_eval_s1:,} S1 entities, {total_true_pairs:,} true positive match pairs.")

# Now test blocking on a subset of S2/S3 (e.g., all S2/S3 for the cities or states of these entities, or 200k S2/S3 records)
# But even better: let's test directly against the full S2/S3 database in DuckDB using SQL queries!
# DuckDB can join 2000 S1 records against all 10M S2/S3 records in seconds using indexes or hash joins!
print("\nLoading full S2 and S3 into a unified candidates table...")
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE all_s2_s3 AS
SELECT entity_id, business_name, business_address, country, 'S2' as src
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
UNION ALL
SELECT entity_id, business_name, business_address, country, 'S3' as src
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true);
""")
print(f"Loaded 10.3M candidate records in {time.time()-t0:.2f}s")

# Let's test blocking rule 1: Normalized alphanumeric name match (Country + normalized name)
print("\n--- Rule 1: Country + Normalized Name Match ---")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE r1_candidates AS
SELECT 
    s.entity_id as s1_id,
    c.entity_id as candidate_id
FROM eval_s1 s
JOIN all_s2_s3 c ON s.country = c.country
WHERE regexp_replace(lower(s.business_name), '[^a-z0-9]', '', 'g') = 
      regexp_replace(lower(c.business_name), '[^a-z0-9]', '', 'g')
  AND length(regexp_replace(lower(s.business_name), '[^a-z0-9]', '', 'g')) > 3;
""")
r1_count = con.execute("SELECT count(*) FROM r1_candidates").fetchone()[0]
r1_hits = con.execute("""
SELECT count(*) FROM eval_gt g 
JOIN r1_candidates c ON g.s1_id = c.s1_id AND g.matched_id = c.candidate_id
""").fetchone()[0]
print(f"Rule 1 generated {r1_count:,} candidate pairs ({r1_count/total_eval_s1:.1f} per S1 entity) in {time.time()-t0:.2f}s")
print(f"Rule 1 Recall: {r1_hits:,} / {total_true_pairs:,} ({r1_hits/total_true_pairs*100:.2f}%)")

# Let's test Rule 2: Blocking on First Significant Token of Name + Country
print("\n--- Rule 2: Country + First Significant Token of Name (>=4 chars) ---")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE r2_candidates AS
SELECT 
    s.entity_id as s1_id,
    c.entity_id as candidate_id
FROM eval_s1 s
JOIN all_s2_s3 c ON s.country = c.country
WHERE split_part(regexp_replace(lower(trim(s.business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ', 1) = 
      split_part(regexp_replace(lower(trim(c.business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ', 1)
  AND length(split_part(regexp_replace(lower(trim(s.business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ', 1)) >= 4;
""")
r2_count = con.execute("SELECT count(*) FROM r2_candidates").fetchone()[0]
r2_hits = con.execute("""
SELECT count(*) FROM eval_gt g 
JOIN r2_candidates c ON g.s1_id = c.s1_id AND g.matched_id = c.candidate_id
""").fetchone()[0]
print(f"Rule 2 generated {r2_count:,} candidate pairs ({r2_count/total_eval_s1:.1f} per S1 entity) in {time.time()-t0:.2f}s")
print(f"Rule 2 Recall: {r2_hits:,} / {total_true_pairs:,} ({r2_hits/total_true_pairs*100:.2f}%)")

# Let's test Rule 3: Blocking on Address Number + First Significant Name Token
print("\n--- Rule 3: Country + Address Digits (Street No / Plot / PIN) + First 3 chars of Name ---")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE r3_candidates AS
SELECT 
    s.entity_id as s1_id,
    c.entity_id as candidate_id
FROM eval_s1 s
JOIN all_s2_s3 c ON s.country = c.country
WHERE regexp_extract(s.business_address, '[0-9]{2,6}') = regexp_extract(c.business_address, '[0-9]{2,6}')
  AND regexp_extract(s.business_address, '[0-9]{2,6}') != ''
  AND substring(lower(regexp_replace(s.business_name, '[^a-z0-9]', '', 'g')), 1, 3) =
      substring(lower(regexp_replace(c.business_name, '[^a-z0-9]', '', 'g')), 1, 3);
""")
r3_count = con.execute("SELECT count(*) FROM r3_candidates").fetchone()[0]
r3_hits = con.execute("""
SELECT count(*) FROM eval_gt g 
JOIN r3_candidates c ON g.s1_id = c.s1_id AND g.matched_id = c.candidate_id
""").fetchone()[0]
print(f"Rule 3 generated {r3_count:,} candidate pairs ({r3_count/total_eval_s1:.1f} per S1 entity) in {time.time()-t0:.2f}s")
print(f"Rule 3 Recall: {r3_hits:,} / {total_true_pairs:,} ({r3_hits/total_true_pairs*100:.2f}%)")

# Union of candidates
con.execute("""
CREATE TEMP TABLE union_candidates AS
SELECT s1_id, candidate_id FROM r1_candidates
UNION
SELECT s1_id, candidate_id FROM r2_candidates
UNION
SELECT s1_id, candidate_id FROM r3_candidates;
""")
u_count = con.execute("SELECT count(*) FROM union_candidates").fetchone()[0]
u_hits = con.execute("""
SELECT count(*) FROM eval_gt g 
JOIN union_candidates c ON g.s1_id = c.s1_id AND g.matched_id = c.candidate_id
""").fetchone()[0]
print(f"\nUnion of Rules 1 + 2 + 3:")
print(f"Total candidates: {u_count:,} ({u_count/total_eval_s1:.1f} per S1 entity)")
print(f"Combined Recall: {u_hits:,} / {total_true_pairs:,} ({u_hits/total_true_pairs*100:.2f}%)")
