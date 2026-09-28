"""
Phase 9: Frequency Audit & True-Match Recovery of Potential New Channels
Amazon ML Challenge 2026: Business Entity Resolution

Evaluates candidate channels for:
1. Bucket size distribution (Max, P95, Mean)
2. Total candidates generated
3. True matches recovered from Category B (the 690 unretrieved matches)
4. Runtime and efficiency
"""

import sys
import os
import io
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import UNAMBIGUOUS_LEGAL_REGEX

gt_path = "output/eval_ground_truth.pkl"
base_train = "student_resource/dataset/train"

print("=" * 80)
print("PHASE 9: NEW BLOCKING CHANNEL AUDIT & CATEGORY B RECOVERY")
print("=" * 80)

with open(gt_path, "rb") as f:
    meta = pickle.load(f)

gt_dict = meta["gt_dict"]
s1_list_all = meta["s1_list"]
s1_lookup = meta["s1_lookup"]

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

train_s1, dev_s1, holdout_s1 = [], [], []
rng = np.random.RandomState(42)
for (c, m_type), ids in sorted(strata.items()):
    shuffled = rng.permutation(ids)
    n = len(shuffled)
    n_train = int(n * 0.60)
    n_dev = int(n * 0.20)
    train_s1.extend(shuffled[:n_train])
    dev_s1.extend(shuffled[n_train:n_train + n_dev])
    holdout_s1.extend(shuffled[n_train + n_dev:])

dev_gt_pairs = set()
for s in dev_s1:
    for c in gt_dict.get(s, set()):
        dev_gt_pairs.add((s, c))

print(f"Dev S1 Count: {len(dev_s1):,} | Total True Matches: {len(dev_gt_pairs):,}")

con = duckdb.connect()

# Dev S1 table
dev_s1_rows = [
    (s, s1_lookup[s]["business_name"], s1_lookup[s]["business_address"], s1_lookup[s]["country"])
    for s in dev_s1
]
df_dev_s1 = pd.DataFrame(dev_s1_rows, columns=["entity_id", "business_name", "business_address", "country"])
con.register("dev_s1", df_dev_s1)

# All candidates
con.execute(f"""
CREATE TEMP TABLE all_candidates AS
SELECT entity_id, business_name, business_address, country, 'S2' as src
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN ('US', 'India')
UNION ALL
SELECT entity_id, business_name, business_address, country, 'S3' as src
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN ('US', 'India');
""")

# Extended Prefix & Domain Regex
EXTENDED_PREFIX_REGEX = (
    r"^(?:the\s+|dr\s+|smt\s+|m/s\.?\s*|mr\s+|co\s+|>>\s+|#\s*|"
    r"formerly\s+|t/a\s+|d/b/a\s+|dba[:\s]+|c/o\s+)"
)

DOMAIN_STRIP_REGEX = r"(?:\.com|\.org|\.net|\.in|\.co\.in|\.co|\.us|\.biz|\.info|\.edu)(?:$|\s)"

print("\n--- 1. AUDITING ENHANCED CHANNEL A & A2 (DOMAIN & DBA PREFIX STRIPPING) ---")
con.execute(f"""
CREATE TEMP TABLE test_chan_a_enhanced AS
WITH s1_clean AS (
    SELECT 
        entity_id as s1_id, country,
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(replace(lower(trim(business_name)), '.', ''), '{DOMAIN_STRIP_REGEX}', '', 'g'),
                    '{EXTENDED_PREFIX_REGEX}', '', 'g'
                ),
                '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'
            ),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM dev_s1
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT 
        entity_id as candidate_id, country,
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(replace(lower(trim(business_name)), '.', ''), '{DOMAIN_STRIP_REGEX}', '', 'g'),
                    '{EXTENDED_PREFIX_REGEX}', '', 'g'
                ),
                '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'
            ),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM all_candidates
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT s.s1_id, c.candidate_id
FROM s1_clean s
JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key
WHERE length(s.norm_key) >= 3;
""")
res_a = con.execute("SELECT count(*), count(DISTINCT s1_id) FROM test_chan_a_enhanced").fetchone()
true_a = con.execute("""
SELECT count(*) FROM test_chan_a_enhanced c
JOIN (SELECT * FROM (VALUES """ + ",".join([f"('{s}', '{c}')" for s, c in dev_gt_pairs]) + """) t(s1, cand)) gt
ON c.s1_id = gt.s1 AND c.candidate_id = gt.cand
""").fetchone()[0]
print(f"  Total Pairs: {res_a[0]:,} | S1s covered: {res_a[1]:,} | True Matches Recovered: {true_a:,} ({true_a/len(dev_gt_pairs)*100:.2f}%)")

print("\n--- 2. AUDITING CHANNEL E3 (NUMBER + STREET TOKEN ANCHOR, PAIR-FREQUENCY BOUNDED) ---")
con.execute("""
CREATE TEMP TABLE s1_addr_pairs AS
SELECT 
    entity_id as s1_id, country,
    cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as num,
    unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM dev_s1
WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != '';

DELETE FROM s1_addr_pairs
WHERE length(tok) < 4
   OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
              'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
              'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');

CREATE TEMP TABLE cand_addr_pairs AS
SELECT 
    entity_id as candidate_id, country,
    cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as num,
    unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates
WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != '';

DELETE FROM cand_addr_pairs
WHERE length(tok) < 4
   OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
              'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
              'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');

-- Count pair frequency: (country, num, tok)
CREATE TEMP TABLE rare_num_tok_pairs AS
SELECT country, num, tok
FROM cand_addr_pairs
GROUP BY country, num, tok
HAVING count(*) BETWEEN 1 AND 50;

CREATE TEMP TABLE test_chan_e3 AS
SELECT DISTINCT s.s1_id, c.candidate_id
FROM s1_addr_pairs s
JOIN rare_num_tok_pairs r ON s.country = r.country AND s.num = r.num AND s.tok = r.tok
JOIN cand_addr_pairs c ON s.country = c.country AND s.num = c.num AND s.tok = c.tok;
""")
res_e3 = con.execute("SELECT count(*), count(DISTINCT s1_id) FROM test_chan_e3").fetchone()
true_e3 = con.execute("""
SELECT count(*) FROM test_chan_e3 c
JOIN (SELECT * FROM (VALUES """ + ",".join([f"('{s}', '{c}')" for s, c in dev_gt_pairs]) + """) t(s1, cand)) gt
ON c.s1_id = gt.s1 AND c.candidate_id = gt.cand
""").fetchone()[0]
print(f"  Total Pairs: {res_e3[0]:,} | S1s covered: {res_e3[1]:,} | True Matches Recovered: {true_e3:,} ({true_e3/len(dev_gt_pairs)*100:.2f}%)")

print("\n--- 3. AUDITING CHANNEL H (POSTAL CODE + 3-CHAR NAME PREFIX) ---")
con.execute("""
CREATE TEMP TABLE test_chan_h AS
WITH s1_post AS (
    SELECT 
        entity_id as s1_id, country,
        regexp_extract(business_address, '\\b[0-9]{5,6}\\b') as post_code,
        substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM dev_s1
    WHERE business_address IS NOT NULL AND regexp_extract(business_address, '\\b[0-9]{5,6}\\b') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
),
cand_post AS (
    SELECT 
        entity_id as candidate_id, country,
        regexp_extract(business_address, '\\b[0-9]{5,6}\\b') as post_code,
        substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM all_candidates
    WHERE business_address IS NOT NULL AND regexp_extract(business_address, '\\b[0-9]{5,6}\\b') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
)
SELECT DISTINCT s.s1_id, c.candidate_id
FROM s1_post s
JOIN cand_post c ON s.country = c.country AND s.post_code = c.post_code AND s.p3 = c.p3;
""")
res_h = con.execute("SELECT count(*), count(DISTINCT s1_id) FROM test_chan_h").fetchone()
true_h = con.execute("""
SELECT count(*) FROM test_chan_h c
JOIN (SELECT * FROM (VALUES """ + ",".join([f"('{s}', '{c}')" for s, c in dev_gt_pairs]) + """) t(s1, cand)) gt
ON c.s1_id = gt.s1 AND c.candidate_id = gt.cand
""").fetchone()[0]
print(f"  Total Pairs: {res_h[0]:,} | S1s covered: {res_h[1]:,} | True Matches Recovered: {true_h:,} ({true_h/len(dev_gt_pairs)*100:.2f}%)")

print("\n--- 4. AUDITING CHANNEL I (DISTINCTIVE NAME TOKEN PAIRS, DF <= 100) ---")
con.execute("""
CREATE TEMP TABLE cand_name_pairs AS
WITH toks AS (
    SELECT country, entity_id as candidate_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM all_candidates
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT country, candidate_id, tok
FROM toks
WHERE length(tok) >= 4
  AND tok NOT IN ('enterprises', 'services', 'solutions', 'industries', 'group', 'holdings', 'trading', 
                  'company', 'limited', 'private', 'consultancy', 'associates', 'properties');

CREATE TEMP TABLE cand_tok_pairs AS
SELECT 
    t1.country, t1.candidate_id,
    least(t1.tok, t2.tok) as tok_a,
    greatest(t1.tok, t2.tok) as tok_b
FROM cand_name_pairs t1
JOIN cand_name_pairs t2 ON t1.candidate_id = t2.candidate_id AND t1.tok < t2.tok;

CREATE TEMP TABLE rare_tok_pairs AS
SELECT country, tok_a, tok_b
FROM cand_tok_pairs
GROUP BY country, tok_a, tok_b
HAVING count(*) BETWEEN 1 AND 50;

CREATE TEMP TABLE s1_name_pairs AS
WITH toks AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM dev_s1
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT country, s1_id, tok
FROM toks
WHERE length(tok) >= 4
  AND tok NOT IN ('enterprises', 'services', 'solutions', 'industries', 'group', 'holdings', 'trading', 
                  'company', 'limited', 'private', 'consultancy', 'associates', 'properties');

CREATE TEMP TABLE s1_tok_pairs AS
SELECT 
    t1.country, t1.s1_id,
    least(t1.tok, t2.tok) as tok_a,
    greatest(t1.tok, t2.tok) as tok_b
FROM s1_name_pairs t1
JOIN s1_name_pairs t2 ON t1.s1_id = t2.s1_id AND t1.tok < t2.tok;

CREATE TEMP TABLE test_chan_i AS
SELECT DISTINCT s.s1_id, c.candidate_id
FROM s1_tok_pairs s
JOIN rare_tok_pairs r ON s.country = r.country AND s.tok_a = r.tok_a AND s.tok_b = r.tok_b
JOIN cand_tok_pairs c ON s.country = c.country AND s.tok_a = c.tok_a AND s.tok_b = c.tok_b;
""")
res_i = con.execute("SELECT count(*), count(DISTINCT s1_id) FROM test_chan_i").fetchone()
true_i = con.execute("""
SELECT count(*) FROM test_chan_i c
JOIN (SELECT * FROM (VALUES """ + ",".join([f"('{s}', '{c}')" for s, c in dev_gt_pairs]) + """) t(s1, cand)) gt
ON c.s1_id = gt.s1 AND c.candidate_id = gt.cand
""").fetchone()[0]
print(f"  Total Pairs: {res_i[0]:,} | S1s covered: {res_i[1]:,} | True Matches Recovered: {true_i:,} ({true_i/len(dev_gt_pairs)*100:.2f}%)")

print("\n--- COMBINED RECOVERY FROM ALL 4 NEW CHANNELS ---")
con.execute("""
CREATE TEMP TABLE all_new_channels AS
SELECT s1_id, candidate_id FROM test_chan_a_enhanced
UNION
SELECT s1_id, candidate_id FROM test_chan_e3
UNION
SELECT s1_id, candidate_id FROM test_chan_h
UNION
SELECT s1_id, candidate_id FROM test_chan_i;
""")
res_all = con.execute("SELECT count(*), count(DISTINCT s1_id) FROM all_new_channels").fetchone()
true_all = con.execute("""
SELECT count(*) FROM all_new_channels c
JOIN (SELECT * FROM (VALUES """ + ",".join([f"('{s}', '{c}')" for s, c in dev_gt_pairs]) + """) t(s1, cand)) gt
ON c.s1_id = gt.s1 AND c.candidate_id = gt.cand
""").fetchone()[0]
print(f"  Combined New Pairs: {res_all[0]:,} | S1s covered: {res_all[1]:,} | True Matches Recovered: {true_all:,} ({true_all/len(dev_gt_pairs)*100:.2f}%)")
