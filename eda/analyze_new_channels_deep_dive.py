"""
Phase 10: Deep-Dive Analysis of New Blocking Channels
Amazon ML Challenge 2026: Business Entity Resolution

Quantifies the unique and joint contributions of:
- Enhanced Channel A & A2 (domain stripping & trade prefix stripping)
- Channel E3 (address number + street token)
- Channel H (postal code + 3-char name prefix)
- Channel I (distinctive name token pairs)
vs the baseline channels (A, A2, B, C, D, E, E2, G).
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import UNAMBIGUOUS_LEGAL_REGEX, PREFIX_NOISE_REGEX
from src.blocking_v2 import EXTENDED_PREFIX_REGEX, DOMAIN_STRIP_REGEX, ADDRESS_STOPWORDS_SQL, BUSINESS_STOPWORDS_SQL

gt_path = "output/eval_ground_truth.pkl"
base_train = "student_resource/dataset/train"

print("=" * 80)
print("PHASE 10: UNIQUE CHANNEL RECOVERY & OVERLAP ANALYSIS")
print("=" * 80)

# Load ground truth and dev split
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

total_dev_true = len(dev_gt_pairs)
print(f"Dev S1 Count: {len(dev_s1):,} | Total True Matches: {total_dev_true:,}")

con = duckdb.connect()

# Setup S1 table
dev_s1_rows = [
    (s, s1_lookup[s]["business_name"], s1_lookup[s]["business_address"], s1_lookup[s]["country"])
    for s in dev_s1
]
df_dev_s1 = pd.DataFrame(dev_s1_rows, columns=["entity_id", "business_name", "business_address", "country"])
con.register("dev_s1_tbl", df_dev_s1)

# Setup candidate pool
t0 = time.time()
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
print(f"Loaded candidates in {time.time()-t0:.2f}s.")

# ------------------------------------------------------------------------------
# GENERATE INDIVIDUAL CHANNELS
# ------------------------------------------------------------------------------
print("\nGenerating candidate pairs for all channels individually...")

# Original A
con.execute(f"""
CREATE TEMP TABLE chan_a_orig AS
WITH s1_clean AS (
    SELECT entity_id as s1_id, country,
           regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM dev_s1_tbl WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT entity_id as candidate_id, country,
           regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_clean s JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key WHERE length(s.norm_key) >= 3;
""")

# Enhanced A & A2
domain_sub = f"regexp_replace(replace(lower(trim(business_name)), '.', ''), '{DOMAIN_STRIP_REGEX}', '', 'g')"
con.execute(f"""
CREATE TEMP TABLE chan_a_enh AS
WITH s1_clean AS (
    SELECT entity_id as s1_id, country,
           regexp_replace(regexp_replace({domain_sub}, '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM dev_s1_tbl WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT entity_id as candidate_id, country,
           regexp_replace(regexp_replace({domain_sub}, '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_clean s JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key WHERE length(s.norm_key) >= 3;
""")

con.execute(f"""
CREATE TEMP TABLE chan_a2_enh AS
WITH s1_clean AS (
    SELECT entity_id as s1_id, country,
           regexp_replace(regexp_replace(regexp_replace({domain_sub}, '{EXTENDED_PREFIX_REGEX}', '', 'g'), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM dev_s1_tbl WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT entity_id as candidate_id, country,
           regexp_replace(regexp_replace(regexp_replace({domain_sub}, '{EXTENDED_PREFIX_REGEX}', '', 'g'), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_clean s JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key WHERE length(s.norm_key) >= 3;
""")

# Baseline channels B, C, D, E, E2, G
con.execute("""
CREATE TEMP TABLE _cand_name_toks AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != '';
DELETE FROM _cand_name_toks WHERE length(tok) < 4;

CREATE TEMP TABLE _rare_name_toks AS
SELECT country, tok FROM _cand_name_toks GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 200;

CREATE TEMP TABLE chan_b AS
WITH s1_tokens AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM dev_s1_tbl WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_tokens s JOIN _rare_name_toks r ON s.country = r.country AND s.tok = r.tok JOIN _cand_name_toks c ON s.country = c.country AND s.tok = c.tok;
""")

# Channel E
con.execute("""
CREATE TEMP TABLE chan_e AS
WITH s1_num AS (
    SELECT entity_id as s1_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as addr_num,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM dev_s1_tbl WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != '' AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
),
cand_num AS (
    SELECT entity_id as candidate_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as addr_num,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM all_candidates WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != '' AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_num s JOIN cand_num c ON s.country = c.country AND s.addr_num = c.addr_num AND s.p3 = c.p3 WHERE length(s.addr_num) > 0;
""")

# Channel D & E2
con.execute(f"""
CREATE TEMP TABLE _cand_addr_toks AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_address IS NOT NULL AND trim(business_address) != '';
DELETE FROM _cand_addr_toks WHERE length(tok) < 5 OR tok IN ({ADDRESS_STOPWORDS_SQL});

CREATE TEMP TABLE _rare_addr_toks AS
SELECT country, tok FROM _cand_addr_toks GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 150;

CREATE TEMP TABLE chan_d AS
WITH s1_tokens AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM dev_s1_tbl WHERE business_address IS NOT NULL AND trim(business_address) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_tokens s JOIN _rare_addr_toks r ON s.country = r.country AND s.tok = r.tok JOIN _cand_addr_toks c ON s.country = c.country AND s.tok = c.tok;

CREATE TEMP TABLE chan_e2 AS
WITH s1_addr_anchor AS (
    SELECT entity_id as s1_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
           unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM dev_s1_tbl WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
),
cand_addr_anchor AS (
    SELECT c.entity_id as candidate_id, c.country,
           cast(ltrim(regexp_extract(c.business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
           cat.tok
    FROM all_candidates c JOIN _cand_addr_toks cat ON c.entity_id = cat.candidate_id
    WHERE c.business_address IS NOT NULL AND regexp_extract(c.business_address, '[0-9]{{1,6}}') != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_addr_anchor s JOIN _rare_addr_toks r ON s.country = r.country AND s.tok = r.tok JOIN cand_addr_anchor c ON s.country = c.country AND s.tok = c.tok AND s.num = c.num WHERE length(s.num) > 0;
""")

# Channel G
con.execute("""
CREATE TEMP TABLE _cand_del_keys AS
SELECT country, entity_id as candidate_id, substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
FROM all_candidates WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7;

CREATE TEMP TABLE _rare_del_keys AS
SELECT country, del_key FROM _cand_del_keys GROUP BY country, del_key HAVING count(*) BETWEEN 2 AND 150;

CREATE TEMP TABLE chan_g AS
WITH s1_del AS (
    SELECT entity_id as s1_id, country, substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
    FROM dev_s1_tbl WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_del s JOIN _rare_del_keys r ON s.country = r.country AND s.del_key = r.del_key JOIN _cand_del_keys c ON s.country = c.country AND s.del_key = c.del_key;
""")

# Channel C
con.execute(f"""
CREATE TEMP TABLE chan_c AS
WITH s1_ngrams AS (
    SELECT entity_id as s1_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
    FROM dev_s1_tbl WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
),
cand_ngrams AS (
    SELECT entity_id as candidate_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
    FROM all_candidates WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_ngrams s JOIN cand_ngrams c ON s.country = c.country AND s.p4 = c.p4 AND s.s4 = c.s4;
""")

# New Channel E3
con.execute(f"""
CREATE TEMP TABLE _cand_num_street AS
SELECT entity_id as candidate_id, country,
       cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
       unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != '';
DELETE FROM _cand_num_street WHERE length(tok) < 4 OR tok IN ({ADDRESS_STOPWORDS_SQL});

CREATE TEMP TABLE _rare_num_street AS
SELECT country, num, tok FROM _cand_num_street GROUP BY country, num, tok HAVING count(*) BETWEEN 1 AND 50;

CREATE TEMP TABLE chan_e3 AS
WITH s1_num_street AS (
    SELECT entity_id as s1_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
           unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM dev_s1_tbl WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_num_street s JOIN _rare_num_street r ON s.country = r.country AND s.num = r.num AND s.tok = r.tok JOIN _cand_num_street c ON s.country = c.country AND s.num = c.num AND s.tok = c.tok;
""")

# New Channel H
con.execute("""
CREATE TEMP TABLE chan_h AS
WITH s1_post AS (
    SELECT entity_id as s1_id, country,
           regexp_extract(business_address, '\\b[0-9]{5,6}\\b') as post_code,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM dev_s1_tbl WHERE business_address IS NOT NULL AND regexp_extract(business_address, '\\b[0-9]{5,6}\\b') != '' AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
),
cand_post AS (
    SELECT entity_id as candidate_id, country,
           regexp_extract(business_address, '\\b[0-9]{5,6}\\b') as post_code,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM all_candidates WHERE business_address IS NOT NULL AND regexp_extract(business_address, '\\b[0-9]{5,6}\\b') != '' AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
)
SELECT DISTINCT s.s1_id, c.candidate_id FROM s1_post s JOIN cand_post c ON s.country = c.country AND s.post_code = c.post_code AND s.p3 = c.p3;
""")

# New Channel I
con.execute(f"""
CREATE TEMP TABLE _cand_name_pairs AS
WITH toks AS (
    SELECT country, entity_id as candidate_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT country, candidate_id, tok FROM toks WHERE length(tok) >= 4 AND tok NOT IN ({BUSINESS_STOPWORDS_SQL});

CREATE TEMP TABLE _cand_tok_pairs AS
SELECT t1.country, t1.candidate_id, least(t1.tok, t2.tok) as tok_a, greatest(t1.tok, t2.tok) as tok_b
FROM _cand_name_pairs t1 JOIN _cand_name_pairs t2 ON t1.candidate_id = t2.candidate_id AND t1.tok < t2.tok;

CREATE TEMP TABLE _rare_tok_pairs AS
SELECT country, tok_a, tok_b FROM _cand_tok_pairs GROUP BY country, tok_a, tok_b HAVING count(*) BETWEEN 1 AND 50;

CREATE TEMP TABLE _s1_name_pairs AS
WITH toks AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM dev_s1_tbl WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT country, s1_id, tok FROM toks WHERE length(tok) >= 4 AND tok NOT IN ({BUSINESS_STOPWORDS_SQL});

CREATE TEMP TABLE _s1_tok_pairs AS
SELECT t1.country, t1.s1_id, least(t1.tok, t2.tok) as tok_a, greatest(t1.tok, t2.tok) as tok_b
FROM _s1_name_pairs t1 JOIN _s1_name_pairs t2 ON t1.s1_id = t2.s1_id AND t1.tok < t2.tok;

CREATE TEMP TABLE chan_i AS
SELECT DISTINCT s.s1_id, c.candidate_id FROM _s1_tok_pairs s JOIN _rare_tok_pairs r ON s.country = r.country AND s.tok_a = r.tok_a AND s.tok_b = r.tok_b JOIN _cand_tok_pairs c ON s.country = c.country AND s.tok_a = c.tok_a AND s.tok_b = c.tok_b;
""")

print("All individual channel tables constructed.")

# ------------------------------------------------------------------------------
# CONVERT TO SETS & PERFORM OVERLAP & UNIQUENESS ANALYSIS
# ------------------------------------------------------------------------------
channel_tables = {
    "Chan A (Orig)": "chan_a_orig",
    "Chan A (Enh)": "chan_a_enh",
    "Chan A2 (Enh)": "chan_a2_enh",
    "Chan B": "chan_b",
    "Chan C": "chan_c",
    "Chan D": "chan_d",
    "Chan E": "chan_e",
    "Chan E2": "chan_e2",
    "Chan G": "chan_g",
    "Chan E3 (New)": "chan_e3",
    "Chan H (New)": "chan_h",
    "Chan I (New)": "chan_i",
}

channel_pairs = {}
channel_true = {}

for name, tbl in channel_tables.items():
    pairs = set(con.execute(f"SELECT s1_id, candidate_id FROM {tbl}").fetchall())
    channel_pairs[name] = pairs
    channel_true[name] = pairs & dev_gt_pairs

# Baseline channels union
baseline_chans = ["Chan A (Orig)", "Chan B", "Chan C", "Chan D", "Chan E", "Chan E2", "Chan G"]
baseline_pairs = set()
baseline_true = set()
for c in baseline_chans:
    baseline_pairs |= channel_pairs[c]
    baseline_true |= channel_true[c]

all_pairs_union = set()
all_true_union = set()
for c, p in channel_pairs.items():
    all_pairs_union |= p
    all_true_union |= channel_true[c]

print("\n" + "=" * 105)
print("INDIVIDUAL CHANNEL STATS & UNIQUE TRUE MATCH CONTRIBUTION")
print("=" * 105)
print(f"{'Channel':<16} | {'Pairs':<10} | {'Avg/S1':<8} | {'True Recovered':<16} | {'Purity':<8} | {'Unique in All':<15} | {'Unique vs Baseline':<20}")
print("-" * 105)

new_channels = ["Chan A (Enh)", "Chan A2 (Enh)", "Chan E3 (New)", "Chan H (New)", "Chan I (New)"]

for name, tbl in channel_tables.items():
    p = channel_pairs[name]
    t = channel_true[name]
    n_pairs = len(p)
    avg_s1 = n_pairs / len(dev_s1)
    n_true = len(t)
    purity = n_true / n_pairs * 100.0 if n_pairs > 0 else 0.0
    
    # Unique across ALL 12 channels
    other_true = set()
    for other_name, other_t in channel_true.items():
        if other_name != name:
            other_true |= other_t
    unique_all = len(t - other_true)
    
    # Unique vs BASELINE channels
    unique_vs_base = len(t - baseline_true)
    
    print(f"{name:<16} | {n_pairs:<10,} | {avg_s1:<8.1f} | {n_true:<6,} ({n_true/total_dev_true*100:5.2f}%) | {purity:<6.2f}% | {unique_all:<15,} | {unique_vs_base:<20,}")

print("-" * 105)
print(f"{'Baseline Union':<16} | {len(baseline_pairs):<10,} | {len(baseline_pairs)/len(dev_s1):<8.1f} | {len(baseline_true):<6,} ({len(baseline_true)/total_dev_true*100:5.2f}%) | {len(baseline_true)/len(baseline_pairs)*100:<6.2f}% | {'-':<15} | {'-':<20}")
print(f"{'Enhanced Union':<16} | {len(all_pairs_union):<10,} | {len(all_pairs_union)/len(dev_s1):<8.1f} | {len(all_true_union):<6,} ({len(all_true_union)/total_dev_true*100:5.2f}%) | {len(all_true_union)/len(all_pairs_union)*100:<6.2f}% | {'-':<15} | {len(all_true_union - baseline_true):<20,}")

# Breakdown of matches recovered ONLY by new channels
recovered_only_by_new = all_true_union - baseline_true
print(f"\nTotal True Matches recovered ONLY by new channels (Category B breakthrough): {len(recovered_only_by_new):,} ({len(recovered_only_by_new)/total_dev_true*100:.2f}%)")

# How many exclusively by E3, H, I, or combinations
excl_e3 = len(channel_true["Chan E3 (New)"] - baseline_true - channel_true["Chan H (New)"] - channel_true["Chan I (New)"] - channel_true["Chan A (Enh)"] - channel_true["Chan A2 (Enh)"])
excl_h = len(channel_true["Chan H (New)"] - baseline_true - channel_true["Chan E3 (New)"] - channel_true["Chan I (New)"] - channel_true["Chan A (Enh)"] - channel_true["Chan A2 (Enh)"])
excl_i = len(channel_true["Chan I (New)"] - baseline_true - channel_true["Chan E3 (New)"] - channel_true["Chan H (New)"] - channel_true["Chan A (Enh)"] - channel_true["Chan A2 (Enh)"])
excl_enh_name = len((channel_true["Chan A (Enh)"] | channel_true["Chan A2 (Enh)"]) - baseline_true - channel_true["Chan E3 (New)"] - channel_true["Chan H (New)"] - channel_true["Chan I (New)"])
multi_new = len(recovered_only_by_new) - excl_e3 - excl_h - excl_i - excl_enh_name

print(f"  - Recovered ONLY by Channel E3 (Address Number + Street) : {excl_e3:,} matches")
print(f"  - Recovered ONLY by Channel I (Distinctive Token Pairs)    : {excl_i:,} matches")
print(f"  - Recovered ONLY by Channel H (Postal + Name Prefix)       : {excl_h:,} matches")
print(f"  - Recovered ONLY by Enhanced A / A2 (Domain/Trade Prefix)  : {excl_enh_name:,} matches")
print(f"  - Recovered by MULTIPLE new channels                       : {multi_new:,} matches")
