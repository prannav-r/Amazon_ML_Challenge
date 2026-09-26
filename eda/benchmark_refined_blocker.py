"""
Optimize Channel G and Multi-Channel Blocker
亚马逊 ML 挑战赛 - Candidate Generation & Blocking
"""

import duckdb
import sys
import io
import time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
con = duckdb.connect()

base_train = "student_resource/dataset/train"

# Fixed seed sample for exact reproducibility
con.execute(f"""
CREATE TEMP TABLE eval_s1 AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'US'
LIMIT 6000;
""")

con.execute(f"""
INSERT INTO eval_s1
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'India'
LIMIT 4000;
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

print(f"Eval S1: {total_s1:,} | Ground Truth: {total_gt:,} | Candidates: {cand_count:,}")

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

def eval_tab(name, tbl, elapsed):
    tot_cands = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
    hits = con.execute(f"""
    SELECT count(*) FROM eval_gt g
    JOIN {tbl} c ON g.s1_id = c.s1_id AND g.true_match_id = c.candidate_id
    """).fetchone()[0]
    rec = (hits / total_gt) * 100.0
    
    per_s1 = con.execute(f"SELECT s1_id, count(*) FROM {tbl} GROUP BY s1_id").fetchall()
    counts_map = {r[0]: r[1] for r in per_s1}
    all_c = [counts_map.get(r[0], 0) for r in con.execute("SELECT entity_id FROM eval_s1").fetchall()]
    
    avg_c = float(np.mean(all_c))
    med_c = float(np.median(all_c))
    p95_c = float(np.percentile(all_c, 95))
    p99_c = float(np.percentile(all_c, 99))
    max_c = int(np.max(all_c)) if all_c else 0
    rr = (1.0 - (tot_cands / total_possible)) * 100.0
    
    print(f"\n[{name}]")
    print(f"  Runtime          : {elapsed:.2f}s")
    print(f"  Total Candidates : {tot_cands:,}")
    print(f"  True Matches     : {hits:,} / {total_gt:,} ({rec:.2f}%)")
    print(f"  Avg Cands / S1   : {avg_c:.1f} (Median: {med_c:.1f}, P95: {p95_c:.1f}, P99: {p99_c:.1f}, Max: {max_c:,})")
    print(f"  Reduction Ratio  : {rr:.6f}%")
    return {"hits": hits, "recall": rec, "cands": tot_cands, "avg": avg_c}

# Channel A
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE ca AS
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
eval_tab("Channel A: Exact Core Name", "ca", time.time() - t0)

# Channel A2
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE ca2 AS
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
eval_tab("Channel A2: Prefix-Stripped Core Name", "ca2", time.time() - t0)

# Channel B: Rare Name Tokens (DF <= 200)
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
CREATE TEMP TABLE cb AS
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
eval_tab("Channel B: Rare Name Tokens", "cb", time.time() - t0)

# Channel C: Character 4-gram
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE cc AS
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
eval_tab("Channel C: Character 4-gram", "cc", time.time() - t0)

# Channel D: Rare Address Tokens (DF <= 150)
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
CREATE TEMP TABLE cd AS
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
eval_tab("Channel D: Distinctive Address Tokens", "cd", time.time() - t0)

# Channel E: Address Number + Name Prefix-3
t0 = time.time()
con.execute("""
CREATE TEMP TABLE ce AS
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
eval_tab("Channel E: Address Number + Name Prefix-3", "ce", time.time() - t0)

# Channel E2: Address Number + Distinctive Locality Token (Native-Script & Alias Recovery)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE ce2 AS
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
eval_tab("Channel E2: Address Number + Distinctive Locality", "ce2", time.time() - t0)

# Channel G: Frequency-Capped Approximate Name Key (DF <= 150)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE cand_del_keys AS
SELECT country, entity_id as candidate_id,
       substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
FROM all_candidates
WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7;
""")
con.execute("""
CREATE TEMP TABLE rare_del_keys AS
SELECT country, del_key FROM cand_del_keys GROUP BY country, del_key HAVING count(*) BETWEEN 2 AND 150;
""")
con.execute("""
CREATE TEMP TABLE cg_capped AS
WITH s1_del AS (
    SELECT entity_id as s1_id, country,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
    FROM eval_s1
    WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7
)
SELECT s.s1_id, c.candidate_id, 60 as priority
FROM s1_del s
JOIN rare_del_keys r ON s.country = r.country AND s.del_key = r.del_key
JOIN cand_del_keys c ON s.country = c.country AND s.del_key = c.del_key;
""")
eval_tab("Channel G (Capped DF <= 150): Fuzzy 1-Edit Initial Key", "cg_capped", time.time() - t0)

# FINAL SCORED COMBINED UNION (A + A2 + B + C + D + E + E2 + G_capped)
print("\n" + "=" * 80)
print("BUILDING FINAL RECOMMENDED COMBINED UNION")
print("=" * 80)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE final_scored_union AS
WITH all_candidates_unioned AS (
    SELECT s1_id, candidate_id, priority FROM ca
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM ca2
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM cb
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM cc
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM cd
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM ce
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM ce2
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM cg_capped
)
SELECT 
    s1_id, candidate_id, 
    sum(priority) as total_priority, 
    count(*) as channels_hit
FROM all_candidates_unioned
GROUP BY s1_id, candidate_id;
""")
eval_tab("FINAL UNPRUNED MULTI-CHANNEL UNION", "final_scored_union", time.time() - t0)

# PRUNING COMPARISONS ACROSS CAPS
print("\n" + "=" * 80)
print("PRUNING COMPARISONS (UNPRUNED vs CAPS 150, 100, 75, 60)")
print("=" * 80)
for cap in [150, 100, 75, 60]:
    t0 = time.time()
    con.execute(f"""
    CREATE TEMP TABLE final_pruned_{cap} AS
    WITH ranked AS (
        SELECT 
            s1_id, candidate_id, total_priority, channels_hit,
            row_number() OVER (
                PARTITION BY s1_id 
                ORDER BY total_priority DESC, channels_hit DESC, candidate_id
            ) as rn
        FROM final_scored_union
    )
    SELECT s1_id, candidate_id
    FROM ranked
    WHERE rn <= {cap};
    """)
    eval_tab(f"Priority Cap = {cap}", f"final_pruned_{cap}", time.time() - t0)

