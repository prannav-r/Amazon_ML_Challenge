"""
Optimized High-Recall Candidate Generation Benchmark
Tests the refined multi-channel blocker designed to recover native-script & aliases
without candidate explosion.
"""

import duckdb
import sys
import io
import time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
con = duckdb.connect()

base_train = "student_resource/dataset/train"

print("=" * 80)
print("BENCHMARKING OPTIMIZED HIGH-RECALL BLOCKING ARCHITECTURE")
print("=" * 80)

# 1. Setup evaluation S1 sample (same 10,000 S1 sample)
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
print(f"Eval S1: {total_s1:,} | Ground Truth Pairs: {total_gt:,}")

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

PREFIX_NOISE_REGEX = r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|co\s+|>>\s+|#\s*)"

# Channel 1: Exact Core Name (with dot and prefix normalization)
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE c_name_exact AS
WITH s1_clean AS (
    SELECT entity_id as s1_id, country,
           regexp_replace(regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{PREFIX_NOISE_REGEX}', '', 'g'), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT entity_id as candidate_id, country,
           regexp_replace(regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{PREFIX_NOISE_REGEX}', '', 'g'), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT s.s1_id, c.candidate_id, 100 as priority
FROM s1_clean s JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key
WHERE length(s.norm_key) >= 3;
""")
print(f"Channel 1 (Core Name Exact) generated in {time.time()-t0:.2f}s")

# Channel 2: Rare Name Tokens (DF <= 200)
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
CREATE TEMP TABLE c_name_rare AS
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
print(f"Channel 2 (Rare Name Tokens) generated in {time.time()-t0:.2f}s")

# Channel 3: Address Number + 3-char Name Prefix
t0 = time.time()
con.execute("""
CREATE TEMP TABLE c_addr_num_name_p3 AS
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
print(f"Channel 3 (Address Number + Name p3) generated in {time.time()-t0:.2f}s")

# Channel 4: Distinctive Locality/Street Tokens (DF <= 150)
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
CREATE TEMP TABLE c_addr_rare AS
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
print(f"Channel 4 (Distinctive Address Tokens) generated in {time.time()-t0:.2f}s")

# Channel 5: Character 4-gram Prefix+Suffix Inverted Index
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE c_char_4g AS
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
print(f"Channel 5 (Character 4-gram Prefix+Suffix) generated in {time.time()-t0:.2f}s")

# Channel 6: Native-Script & Alias Recovery (Address Number + Rare Address Token)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE c_native_script_recovery AS
WITH s1_pairs AS (
    SELECT entity_id as s1_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as num
    FROM eval_s1 WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != ''
),
cand_pairs AS (
    SELECT entity_id as candidate_id, country,
           cast(ltrim(regexp_extract(business_address, '[0-9]{1,6}'), '0') as varchar) as num
    FROM all_candidates WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{1,6}') != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id, 70 as priority
FROM s1_pairs s
JOIN c_addr_rare ar ON s.s1_id = ar.s1_id
JOIN cand_pairs c ON s.country = c.country AND s.num = c.num AND ar.candidate_id = c.candidate_id
WHERE length(s.num) > 0;
""")
print(f"Channel 6 (Native Script Number + Rare Address Token) generated in {time.time()-t0:.2f}s")

# FULL UNION WITH MULTI-SIGNAL PRIORITY SCORING
print("\nBuilding Full Multi-Channel Scored Union...")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE full_scored_union AS
WITH all_p AS (
    SELECT s1_id, candidate_id, priority FROM c_name_exact
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM c_name_rare
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM c_addr_num_name_p3
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM c_addr_rare
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM c_char_4g
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM c_native_script_recovery
)
SELECT 
    s1_id, candidate_id, 
    sum(priority) as total_priority, 
    count(*) as channels_hit
FROM all_p
GROUP BY s1_id, candidate_id;
""")
print(f"Scored union built in {time.time()-t0:.2f}s")

total_cands = con.execute("SELECT count(*) FROM full_scored_union").fetchone()[0]
union_hits = con.execute("""
SELECT count(*) FROM eval_gt g
JOIN full_scored_union u ON g.s1_id = u.s1_id AND g.true_match_id = u.candidate_id
""").fetchone()[0]

print(f"\n================================================================================")
print(f"FULL UNPRUNED UNION RESULTS:")
print(f"  Total Candidates : {total_cands:,} ({total_cands/total_s1:.1f} per S1)")
print(f"  True Matches     : {union_hits:,} / {total_gt:,} ({(union_hits/total_gt)*100:.2f}% RECALL)")
print(f"================================================================================")

# PRUNING COMPARISONS (Caps 150, 100, 75, 60)
print("\n--- PRUNING COMPARISONS ACROSS PRIORITY CAPS ---")
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
        FROM full_scored_union
    )
    SELECT s1_id, candidate_id
    FROM ranked
    WHERE rn <= {cap};
    """)
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
    rr = (1.0 - (p_tot / (total_s1 * cand_count))) * 100.0
    
    print(f"\n[Priority Cap = {cap}]")
    print(f"  Runtime          : {time.time()-t0:.2f}s")
    print(f"  Total Candidates : {p_tot:,}")
    print(f"  True Matches     : {p_hits:,} / {total_gt:,} ({rec:.2f}%) [Retained {(p_hits/union_hits)*100:.2f}% of Union Recall]")
    print(f"  Avg Cands / S1   : {avg_c:.1f} (Median: {med_c:.1f}, P95: {p95_c:.1f}, P99: {p99_c:.1f}, Max: {max_c:,})")
    print(f"  Reduction Ratio  : {rr:.6f}%")
