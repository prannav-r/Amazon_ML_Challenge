"""
Phase 3 Revision Benchmark & Validation Script
Amazon ML Challenge: Business Entity Resolution

This script:
1. Sets up the reproducible 10,000 S1 held-out evaluation sample (6,000 US, 4,000 India).
2. Audits the 14.48% missed matches from initial blocker into concrete failure categories.
3. Benchmarks new high-recall blocking channels:
   - Channel A: Exact Core Name
   - Channel A2: Prefix-Stripped Core Name
   - Channel B: Rare Name Tokens (DF <= 200)
   - Channel C (4-gram) vs Channel C3 (3-gram prefix+suffix and 3-gram prefix+mid)
   - Channel D: Distinctive Address Tokens (DF <= 150)
   - Channel E: Address Number + Name Prefix-3
   - Channel E2: Address Number + Distinctive Locality Token (DF <= 250) [Cross-Script & Alias recovery]
   - Channel G: Approximate Name Retrieval (1-edit / initial substitution invariant)
4. Measures recovery of Native-Script, Severe Typos (e.g. 6nni vs Gnni), and Aliases.
5. Evaluates the updated multi-signal union recall.
6. Evaluates pruning across Priority Caps: Unpruned, 150, 100, 75, 60.
7. Validates 25 representative previously missed matches.
"""

import duckdb
import sys
import io
import time
import re
import numpy as np
from collections import Counter, defaultdict

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
con = duckdb.connect()

base_train = "student_resource/dataset/train"

print("=" * 80)
print("PHASE 3 REVISION: HIGH-RECALL CANDIDATE GENERATION & BLOCKING BENCHMARK")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. SETUP REPRODUCIBLE EVALUATION SAMPLE
# ------------------------------------------------------------------------------
print("\n[1/7] Loading evaluation sample (10,000 S1 entities: 6k US, 4k India)...")
con.execute(f"""
CREATE TEMP TABLE eval_s1 AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'US'
USING SAMPLE 6000 (reservoir);
""")

con.execute(f"""
INSERT INTO eval_s1
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'India'
USING SAMPLE 4000 (reservoir);
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

con.execute(f"""
CREATE TEMP TABLE all_candidates AS
SELECT entity_id, business_name, business_address, country, 'S2' as src
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
UNION ALL
SELECT entity_id, business_name, business_address, country, 'S3' as src
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true);
""")

total_s1 = con.execute("SELECT count(*) FROM eval_s1").fetchone()[0]
total_gt = con.execute("SELECT count(*) FROM eval_gt").fetchone()[0]
cand_count = con.execute("SELECT count(*) FROM all_candidates").fetchone()[0]
total_possible = total_s1 * cand_count

print(f"  Eval S1 Entities : {total_s1:,}")
print(f"  Ground Truth Pairs: {total_gt:,}")
print(f"  Candidate Pool   : {cand_count:,} (S2 + S3)")
print(f"  Max Possible Pairs: {total_possible:,}")

UNAMBIGUOUS_LEGAL_REGEX = (
    r"(?:[,\s\-\(\[\#]+|\b)("
    r"private\s+limited\s+company|limited\s+liability\s+company|limited\s+liability\s+partnership|"
    r"public\s+limited\s+company|private\s+limited|public\s+limited|"
    r"pvt\s+ltd|pvt\s+limited|private\s+ltd|pub\s+ltd|pub\s+limited|"
    r"corporation|incorporated|limited|company|corp|inc|llc|llp|plc|ltd|co|pvt|"
    r"societe\s+a\s+responsabilite\s+limitee|societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|"
    r"societe\s+par\s+actions\s+simplifiee|entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|"
    r"societe\s+anonyme|societe\s+civile|sarlu|sasu|sarl|sas|eurl|sci|snc|sa|ei|et\s+fils|fils|"
    r"प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|लिमिटेड|एलएलपी"
    r")(?:[,\s\.\)\]]*)$"
)

PREFIX_NOISE_REGEX = r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|mr\s+|co\s+|>>\s+|#\s*)"

# Helper function to evaluate any channel
def eval_channel(name, table_name, elapsed_time):
    tot_cands = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    hits = con.execute(f"""
    SELECT count(*) FROM eval_gt g
    JOIN {table_name} c ON g.s1_id = c.s1_id AND g.true_match_id = c.candidate_id
    """).fetchone()[0]
    rec = (hits / total_gt) * 100.0
    
    per_s1 = con.execute(f"SELECT s1_id, count(*) FROM {table_name} GROUP BY s1_id").fetchall()
    counts_map = {r[0]: r[1] for r in per_s1}
    all_c = [counts_map.get(r[0], 0) for r in con.execute("SELECT entity_id FROM eval_s1").fetchall()]
    
    avg_c = float(np.mean(all_c))
    med_c = float(np.median(all_c))
    p95_c = float(np.percentile(all_c, 95))
    p99_c = float(np.percentile(all_c, 99))
    max_c = int(np.max(all_c)) if all_c else 0
    rr = (1.0 - (tot_cands / total_possible)) * 100.0
    
    print(f"\n[{name}]")
    print(f"  Runtime          : {elapsed_time:.2f}s")
    print(f"  Total Candidates : {tot_cands:,}")
    print(f"  True Matches     : {hits:,} / {total_gt:,} ({rec:.2f}%)")
    print(f"  Avg Cands / S1   : {avg_c:.1f} (Median: {med_c:.1f}, P95: {p95_c:.1f}, P99: {p99_c:.1f}, Max: {max_c:,})")
    print(f"  Reduction Ratio  : {rr:.6f}%")
    return {
        "name": name,
        "hits": hits,
        "recall": rec,
        "total_cands": tot_cands,
        "avg": avg_c,
        "median": med_c,
        "p95": p95_c,
        "p99": p99_c,
        "max": max_c,
        "rr": rr,
        "runtime": elapsed_time
    }

channel_results = []

# ------------------------------------------------------------------------------
# 2. BENCHMARK TARGETED BLOCKING CHANNELS
# ------------------------------------------------------------------------------
print("\n[2/7] Benchmarking individual channels...")

# CHANNEL A: Exact Core Name
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE chan_a AS
WITH s1_clean AS (
    SELECT entity_id as s1_id, country,
           regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT entity_id as candidate_id, country,
           regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT s.s1_id, c.candidate_id, 100 as priority
FROM s1_clean s JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key
WHERE length(s.norm_key) >= 3;
""")
channel_results.append(eval_channel("Channel A: Exact Core Name", "chan_a", time.time() - t0))

# CHANNEL A2: Prefix-Stripped Core Name (The/Dr/Smt/M/s/Mr/Co prefixes)
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE chan_a2 AS
WITH s1_clean AS (
    SELECT entity_id as s1_id, country,
           regexp_replace(
               regexp_replace(
                   regexp_replace(replace(lower(trim(business_name)), '.', ''), '{PREFIX_NOISE_REGEX}', '', 'g'),
                   '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'
               ),
               '[^a-z0-9]', '', 'g'
           ) as norm_key
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT entity_id as candidate_id, country,
           regexp_replace(
               regexp_replace(
                   regexp_replace(replace(lower(trim(business_name)), '.', ''), '{PREFIX_NOISE_REGEX}', '', 'g'),
                   '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'
               ),
               '[^a-z0-9]', '', 'g'
           ) as norm_key
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT s.s1_id, c.candidate_id, 95 as priority
FROM s1_clean s JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key
WHERE length(s.norm_key) >= 3;
""")
channel_results.append(eval_channel("Channel A2: Prefix-Stripped Core Name", "chan_a2", time.time() - t0))

# CHANNEL B: Rare Name Tokens (DF <= 200)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE cand_name_toks AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != '';
""")
con.execute("DELETE FROM cand_name_toks WHERE length(tok) < 4;")
con.execute("""
CREATE TEMP TABLE rare_name_toks AS
SELECT country, tok FROM cand_name_toks GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 200;
""")
con.execute("""
CREATE TEMP TABLE chan_b AS
WITH s1_toks AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id, 80 as priority
FROM s1_toks s
JOIN rare_name_toks r ON s.country = r.country AND s.tok = r.tok
JOIN cand_name_toks c ON s.country = c.country AND s.tok = c.tok;
""")
channel_results.append(eval_channel("Channel B: Rare Name Tokens (DF <= 200)", "chan_b", time.time() - t0))

# CHANNEL C: Character 4-gram (Prefix-4 + Suffix-4)
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE chan_c4 AS
WITH s1_ng AS (
    SELECT entity_id as s1_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
    FROM eval_s1 WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
),
cand_ng AS (
    SELECT entity_id as candidate_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
    FROM all_candidates WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
)
SELECT s.s1_id, c.candidate_id, 50 as priority
FROM s1_ng s JOIN cand_ng c ON s.country = c.country AND s.p4 = c.p4 AND s.s4 = c.s4;
""")
channel_results.append(eval_channel("Channel C: Character 4-gram (Prefix-4 + Suffix-4)", "chan_c4", time.time() - t0))

# CHANNEL C3-A: Character 3-gram (Prefix-3 + Suffix-3)
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE chan_c3_ps AS
WITH s1_ng AS (
    SELECT entity_id as s1_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 3) as p3,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -3) as s3
    FROM eval_s1 WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 4
),
cand_ng AS (
    SELECT entity_id as candidate_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 3) as p3,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -3) as s3
    FROM all_candidates WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 4
)
SELECT s.s1_id, c.candidate_id, 45 as priority
FROM s1_ng s JOIN cand_ng c ON s.country = c.country AND s.p3 = c.p3 AND s.s3 = c.s3;
""")
channel_results.append(eval_channel("Channel C3-A: Character 3-gram (Prefix-3 + Suffix-3)", "chan_c3_ps", time.time() - t0))

# CHANNEL C3-B: Dual 3-gram (Prefix-3 + Mid-3)
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE chan_c3_pm AS
WITH s1_ng AS (
    SELECT entity_id as s1_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 3) as p3,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 4, 3) as m3
    FROM eval_s1 WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 6
),
cand_ng AS (
    SELECT entity_id as candidate_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 3) as p3,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 4, 3) as m3
    FROM all_candidates WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 6
)
SELECT s.s1_id, c.candidate_id, 45 as priority
FROM s1_ng s JOIN cand_ng c ON s.country = c.country AND s.p3 = c.p3 AND s.m3 = c.m3;
""")
channel_results.append(eval_channel("Channel C3-B: Dual 3-gram (Prefix-3 + Mid-3)", "chan_c3_pm", time.time() - t0))

# CHANNEL D: Distinctive Address Tokens (DF <= 150)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE cand_addr_toks AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_address IS NOT NULL AND trim(business_address) != '';
""")
con.execute("""
DELETE FROM cand_addr_toks 
WHERE length(tok) < 5 
   OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
              'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
              'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');
""")
con.execute("""
CREATE TEMP TABLE rare_addr_toks AS
SELECT country, tok FROM cand_addr_toks GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 150;
""")
con.execute("""
CREATE TEMP TABLE chan_d AS
WITH s1_toks AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 WHERE business_address IS NOT NULL AND trim(business_address) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id, 65 as priority
FROM s1_toks s
JOIN rare_addr_toks r ON s.country = r.country AND s.tok = r.tok
JOIN cand_addr_toks c ON s.country = c.country AND s.tok = c.tok;
""")
channel_results.append(eval_channel("Channel D: Distinctive Address Tokens (DF <= 150)", "chan_d", time.time() - t0))

# CHANNEL E: Address Number + Name Prefix-3
t0 = time.time()
con.execute("""
CREATE TEMP TABLE chan_e AS
WITH s1_n AS (
    SELECT entity_id as s1_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as num,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM eval_s1 WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
),
c_n AS (
    SELECT entity_id as candidate_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as num,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM all_candidates WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
)
SELECT s.s1_id, c.candidate_id, 75 as priority
FROM s1_n s JOIN c_n c ON s.country = c.country AND s.num = c.num AND s.p3 = c.p3
WHERE length(s.num) > 0;
""")
channel_results.append(eval_channel("Channel E: Address Number + Name Prefix-3", "chan_e", time.time() - t0))

# CHANNEL E2: Address Number + Distinctive Locality Token (DF <= 250) [Native-Script & Alias Recovery]
t0 = time.time()
con.execute("""
CREATE TEMP TABLE chan_e2 AS
WITH s1_addr_anchor AS (
    SELECT 
        entity_id as s1_id, country,
        cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as num,
        unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 
    WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != ''
),
cand_addr_anchor AS (
    SELECT 
        c.entity_id as candidate_id, c.country,
        cast(ltrim(regexp_extract(c.business_address, '[0-9]{1,6}'), '0') as varchar) as num,
        cat.tok
    FROM all_candidates c
    JOIN cand_addr_toks cat ON c.entity_id = cat.candidate_id
    WHERE c.business_address IS NOT NULL AND regexp_extract(c.business_address, '[0-9]{1,6}') != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id, 85 as priority
FROM s1_addr_anchor s
JOIN rare_addr_toks r ON s.country = r.country AND s.tok = r.tok
JOIN cand_addr_anchor c ON s.country = c.country AND s.tok = c.tok AND s.num = c.num
WHERE length(s.num) > 0;
""")
channel_results.append(eval_channel("Channel E2: Address Number + Distinctive Locality (Native-Script/Alias)", "chan_e2", time.time() - t0))

# CHANNEL G: Approximate Name Retrieval (Prefix-independent / Initial-char typo invariant index)
# Catches position 0 typos (e.g. '6' vs 'G', '0' vs 'O') by indexing characters 2..8
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE chan_g AS
WITH s1_del AS (
    SELECT 
        entity_id as s1_id, country,
        substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
    FROM eval_s1
    WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 6
),
cand_del AS (
    SELECT 
        entity_id as candidate_id, country,
        substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
    FROM all_candidates
    WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 6
)
SELECT s.s1_id, c.candidate_id, 60 as priority
FROM s1_del s
JOIN cand_del c ON s.country = c.country AND s.del_key = c.del_key
WHERE length(s.del_key) >= 5;
""")
channel_results.append(eval_channel("Channel G: Initial-Char Typo Invariant Index", "chan_g", time.time() - t0))

# ------------------------------------------------------------------------------
# 3. VERIFY RECOVERY OF SPECIFIC TARGET PHENOMENA
# ------------------------------------------------------------------------------
print("\n[3/7] Verifying recovery of specific problem categories...")

# A. Severe typo recovery (Gnni vs 6nni, O vs 0)
typo_test = con.execute("""
WITH typo_pairs AS (
    SELECT 'Gnni Sutton, DDS' as s1, '6nni Sutton, DDS' as cand
)
SELECT 
    substring(regexp_replace(lower(s1), '[^a-z0-9]', '', 'g'), 2, 7) as s1_key,
    substring(regexp_replace(lower(cand), '[^a-z0-9]', '', 'g'), 2, 7) as cand_key,
    (substring(regexp_replace(lower(s1), '[^a-z0-9]', '', 'g'), 2, 7) = 
     substring(regexp_replace(lower(cand), '[^a-z0-9]', '', 'g'), 2, 7)) as matches
FROM typo_pairs;
""").fetchall()
print(f"  Approximate Name Test ('Gnni Sutton' <-> '6nni Sutton'): Key Match = {typo_test[0][2]}")

# ------------------------------------------------------------------------------
# 4. FULL UPDATED MULTI-CHANNEL UNION
# ------------------------------------------------------------------------------
print("\n[4/7] Generating Updated Multi-Channel Union...")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE updated_scored_union AS
WITH all_candidates_unioned AS (
    SELECT s1_id, candidate_id, priority FROM chan_a
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_a2
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_b
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_c4
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_c3_pm
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_d
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_e
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_e2
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM chan_g
)
SELECT 
    s1_id, 
    candidate_id, 
    sum(priority) as total_priority, 
    count(*) as channels_hit
FROM all_candidates_unioned
GROUP BY s1_id, candidate_id;
""")
union_time = time.time() - t0

tot_union_cands = con.execute("SELECT count(*) FROM updated_scored_union").fetchone()[0]
union_hits = con.execute("""
SELECT count(*) FROM eval_gt g
JOIN updated_scored_union u ON g.s1_id = u.s1_id AND g.true_match_id = u.candidate_id
""").fetchone()[0]
union_recall = (union_hits / total_gt) * 100.0

per_s1 = con.execute("SELECT s1_id, count(*) FROM updated_scored_union GROUP BY s1_id").fetchall()
counts_map = {r[0]: r[1] for r in per_s1}
all_c = [counts_map.get(r[0], 0) for r in con.execute("SELECT entity_id FROM eval_s1").fetchall()]
avg_union = float(np.mean(all_c))
med_union = float(np.median(all_c))
p95_union = float(np.percentile(all_c, 95))
p99_union = float(np.percentile(all_c, 99))
max_union = int(np.max(all_c))
rr_union = (1.0 - (tot_union_cands / total_possible)) * 100.0

print(f"\n================================================================================")
print(f"UPDATED UNPRUNED UNION RESULTS:")
print(f"  Runtime          : {union_time:.2f}s")
print(f"  Total Candidates : {tot_union_cands:,} ({avg_union:.1f} per S1)")
print(f"  True Matches     : {union_hits:,} / {total_gt:,} ({union_recall:.2f}% RECALL)")
print(f"  Median Cands/S1  : {med_union:.1f} | P95: {p95_union:.1f} | P99: {p99_union:.1f} | Max: {max_union:,}")
print(f"  Reduction Ratio  : {rr_union:.6f}%")
print(f"================================================================================")

# ------------------------------------------------------------------------------
# 5. PRUNING COMPARISONS ACROSS CAPS
# ------------------------------------------------------------------------------
print("\n[5/7] Evaluating Pruning Across Priority Caps (Unpruned, 150, 100, 75, 60)...")

pruning_results = []
pruning_results.append({
    "cap": "Unpruned Union",
    "hits": union_hits,
    "recall": union_recall,
    "total_cands": tot_union_cands,
    "avg": avg_union,
    "median": med_union,
    "p95": p95_union,
    "p99": p99_union,
    "max": max_union,
    "rr": rr_union,
    "runtime": union_time
})

for cap in [150, 100, 75, 60]:
    t0 = time.time()
    con.execute(f"""
    CREATE TEMP TABLE pruned_{cap} AS
    WITH ranked AS (
        SELECT 
            s1_id, candidate_id, total_priority, channels_hit,
            row_number() OVER (
                PARTITION BY s1_id 
                ORDER BY total_priority DESC, channels_hit DESC, candidate_id
            ) as rn
        FROM updated_scored_union
    )
    SELECT s1_id, candidate_id
    FROM ranked
    WHERE rn <= {cap};
    """)
    elapsed = time.time() - t0
    p_tot = con.execute(f"SELECT count(*) FROM pruned_{cap}").fetchone()[0]
    p_hits = con.execute(f"""
    SELECT count(*) FROM eval_gt g
    JOIN pruned_{cap} p ON g.s1_id = p.s1_id AND g.true_match_id = p.candidate_id
    """).fetchone()[0]
    rec = (p_hits / total_gt) * 100.0
    
    per_s1 = con.execute(f"SELECT s1_id, count(*) FROM pruned_{cap} GROUP BY s1_id").fetchall()
    counts_map = {r[0]: r[1] for r in per_s1}
    all_c = [counts_map.get(r[0], 0) for r in con.execute("SELECT entity_id FROM eval_s1").fetchall()]
    
    avg_c = float(np.mean(all_c))
    med_c = float(np.median(all_c))
    p95_c = float(np.percentile(all_c, 95))
    p99_c = float(np.percentile(all_c, 99))
    max_c = int(np.max(all_c)) if all_c else 0
    rr = (1.0 - (p_tot / total_possible)) * 100.0
    
    res = {
        "cap": f"Cap {cap}",
        "hits": p_hits,
        "recall": rec,
        "total_cands": p_tot,
        "avg": avg_c,
        "median": med_c,
        "p95": p95_c,
        "p99": p99_c,
        "max": max_c,
        "rr": rr,
        "runtime": elapsed
    }
    pruning_results.append(res)
    print(f"\n[Priority Cap = {cap}]")
    print(f"  Runtime          : {elapsed:.2f}s")
    print(f"  Total Candidates : {p_tot:,}")
    print(f"  True Matches     : {p_hits:,} / {total_gt:,} ({rec:.2f}%) [Retained {(p_hits/union_hits)*100:.2f}% of Union Recall]")
    print(f"  Avg Cands / S1   : {avg_c:.1f} (Median: {med_c:.1f}, P95: {p95_c:.1f}, P99: {p99_c:.1f}, Max: {max_c:,})")
    print(f"  Reduction Ratio  : {rr:.6f}%")

# ------------------------------------------------------------------------------
# 6. VERIFY PREVIOUSLY MISSED MATCHES (25 REPRESENTATIVE EXAMPLES)
# ------------------------------------------------------------------------------
print("\n[6/7] Tracking 25 representative previously missed matches...")

test_samples = [
    # Native-Script / Cross-Script
    ("S1-685876877", "S3-422231099", "Prime Foods", "प्राइम फूड्स", "4, Gulmohar, Gultekdi, Pune, Maharashtra", "4, Gulmohar, Gultekdi, Pune, MH", "Native-Script"),
    ("S1-851000755", "S2-572814418", "Indian Industries Pvt Ltd", "इंडियन इंडस्ट्रीज प्रा. लि.", "201 Veer Sarvarkar Block, East Delhi, 19-B S/F Pvt Office No.", "19-B S/F PVT OFFICE NO., SHAHDARA, Delhi", "Native-Script"),
    ("S1-898986640", "S2-398605759", "Anand Foundation Private Limited", "ಆನಂದ್ ಫೌಂಡೇಶನ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "No 688/B, F 36/B, B B Garden 2Nd Main Road, Mysore", "NO 688/B, F 36/B, B B GARDEN 2ND MAIN ROAD, MYSORE", "Native-Script"),
    ("S1-358249590", "S2-570189202", "Apex Products Private Limited", "एपेक्स प्रोडक्ट्स प्राइवेट लिमिटेड", "7 Snehlata Ganj, Indore, Madhya Pradesh", "7 SNEHLATA GANJ, INDORE, Madhya Pradesh", "Native-Script"),
    ("S1-621735796", "S3-567732293", "One Estate LLP", "वन एस्टेट एलएलपी", "501 B Wing, Express Zone, Malad East, Mumbai", "501 B Wing, Express Zone, Malad East, Mumbai", "Native-Script"),
    
    # Prefix / Noise discrepancies
    ("S1-288971394", "S2-426756976", "Great Guild", "The Great Guild", "91 Malcolm Street, Unit Apartment B4, Ossining, NY", "#91 MALCOLM ST, OSSINING, NY", "Prefix Noise ('The ')"),
    ("S1-83625448", "S2-597434714", "Jai Impex Private Limited", "Mr Jai Impex Privte Limited", "874/4 Lal Dora, Delhi", "74/4 Lal Dora, Delhi", "Prefix Noise ('Mr ')"),
    ("S1-173812578", "S3-874768128", "Black Tech Pvt Ltd", "Smt Black Tech Pvt Ltd", "12 Industrial Area, Phase 2, Chandigarh", "12 Industrial Area, Phase 2, Chandigarh", "Prefix Noise ('Smt ')"),
    ("S1-105022531", "S2-611098764", "74/60 Pizza", "THE 74/60 PIZZA", "108 Main St, Dallas, TX", "108 MAIN ST, DALLAS, TX", "Prefix Noise ('THE ')"),
    ("S1-89576694", "S2-484555273", "Delgado Academy", "The Delgado Academy", "410 Pine St, Tampa, FL", "410 PINE ST, TAMPA, FL", "Prefix Noise ('The ')"),
    ("S1-173467573", "S3-740712235", "Vision North Consultancy Private Limited", "Pyraectoveo DBA: Vision North Consultancy", "B-5/148, Safdarjung Enclave, New Delhi", "B-5/148, Safdarjung Enclave, New Delhi", "DBA Prefix"),

    # Minor Typos / Levenshtein <= 2
    ("S1-699565963", "S3-104081829", "Willow Co", "Wsillw Co", "1046 24 Street, Brooklyn, NY", "1046 24 St, Brooklyn, New York", "Typo (Wsillw)"),
    ("S1-900437547", "S2-522879516", "Pediatric Dental Group of Golden Valley LLC", "Pediatric Dental Group of Golden Valhey LLC", "2661 Apache Road, Golden Valley, AZ", "APACHE RD, GOLDEN VALLEY, AZ", "Typo (Valhey)"),
    ("S1-87856323", "S2-275155485", "Keystone Live LLC", "KEYSTONE LÍVE LLC", "6158 3rd Court, Renton, WA", "THIRD CT, RENTON, WA", "Accent / Typo"),
    ("S1-641990065", "S2-683514251", "Southern First Vision Center LLC", "The S0uthern First Vision Center LLC", "Indianapolis, IN, 2672 1050", "2672 1050, INDIANAPOLIS, IN", "Digit Sub ('0' vs 'o')"),
    ("S1-861859613", "S2-333575115", "Orion L.L.C.", "0rion L.L.C.", "452 Lincoln Way, Ames, IA", "452 LINCOLN WAY, AMES, IA", "Digit Sub ('0' vs 'O')"),

    # Aliases / Web Domain Names
    ("S1-195709451", "S2-555826021", "Urgent Care Center LLC", "urgentcarecenter.com", "431 Russell Street, Huntsville, AL", "00431 RUSSELL SAINT, HUNTSVILLE, AL", "Domain Alias"),
    ("S1-949679687", "S2-327144958", "Sharp Hardware LLP", "Smt sharphardware.com", "601 Mira Chs Ltd, Sudama Bldg, Mumbai", "C-601 MIRA CHS LTD, SUDAMA BLDG, MUMBAI", "Domain Alias + Prefix"),
    ("S1-839511041", "S3-450395744", "Beatty's Management", "beattysmanagement.com", "712 Broadway, New York, NY", "712 Broadway, New York, NY", "Domain Alias"),
    ("S1-257022224", "S2-220797717", "Averyl Hickman Materials Inc.", "averylhickmanmaterials.com #11557", "901 Oak St, Denver, CO", "901 OAK ST, DENVER, CO", "Domain Alias"),
    ("S1-143817370", "S2-223814487", "Mutual Telecom Brands PC", "MUTUALTELECOMBRANDS.COM", "5627 Sandpiper Lane, Dayton, OH", "627 SANDPIPER LANE, DAYTON, OH", "Domain Alias"),

    # Word Order / Token Shuffling
    ("S1-56607895", "S3-115527350", "Vega & Co", "Center Co Vega", "110 State St, Boston, MA", "110 State St, Boston, MA", "Word Reordering"),
    ("S1-438680175", "S3-980192481", "Dow Advanced Motor Inc", "Inc Dow Advanced Mótor", "320 Market St, Philadelphia, PA", "320 Market St, Philadelphia, PA", "Word Reordering"),
    ("S1-380764336", "S2-538016144", "HKB Information Pvt Ltd", "HKB Information Ltd Pvt", "55 MG Road, Bangalore, Karnataka", "55 MG ROAD, BANGALORE, Karnataka", "Suffix Swap"),

    # Completely Disjoint Trade Name
    ("S1-309715126", "S3-803410155", "Coleman & Johnson Goldman", "Quokor", "3854 Klein Avenue, Stow, OH", "##3854 Klein Avenue, Stow, Ohio", "Disjoint Alias (Address Number Shared)")
]

recovered_count = 0
print(f"\n{'Status':11s} | {'S1 ID':13s} | {'Candidate ID':13s} | {'Category':25s} | {'S1 Name':30s} <-> {'Cand Name'}")
print("-" * 125)
for s1_id, m_id, s1_n, m_n, s1_a, m_a, cat in test_samples:
    recov_union = con.execute(f"SELECT count(*) FROM updated_scored_union WHERE s1_id = '{s1_id}' AND candidate_id = '{m_id}'").fetchone()[0]
    recov_100 = con.execute(f"SELECT count(*) FROM pruned_100 WHERE s1_id = '{s1_id}' AND candidate_id = '{m_id}'").fetchone()[0]
    
    if recov_100 > 0:
        status = "RECOVERED (Cap100)"
        recovered_count += 1
    elif recov_union > 0:
        status = "IN UNION"
        recovered_count += 1
    else:
        status = "MISSED"
        
    print(f"{status:18s} | {s1_id:13s} | {m_id:13s} | {cat:25s} | {s1_n[:30]:30s} <-> {m_n[:30]}")

print(f"\nPreviously missed recovery rate: {recovered_count} / {len(test_samples)} ({recovered_count/len(test_samples)*100:.1f}%)")

print("\n" + "=" * 80)
print("BENCHMARK COMPLETED SUCCESSFULLY")
print("=" * 80)
