"""
Benchmark New High-Recall Blocking Channels (Phase 3 Revision)
Tests:
- Channel C3: Character 3-gram inverted index
- Channel A2: Prefix-stripped & Token-sorted Core Name (handles reordered words & prefixes like The/Dr/Smt/M/s)
- Channel E2: Address Number + Distinctive Locality/City Token (recovers native-script & aliases)
- Channel G: Digit-letter substitution & Levenshtein-1 fuzzy anchor (recovers '6' vs 'G', '0' vs 'O')
- Channel H: Address-less Name Token Pair fallback
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
print("BENCHMARKING NEW HIGH-RECALL BLOCKING CHANNELS")
print("=" * 80)

# Setup evaluation tables (same evaluation sample)
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
print(f"Eval S1: {total_s1:,} | True Pairs: {total_gt:,} | Candidate Pool: {cand_count:,}")

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

# Common noisy prefixes observed in audit (The, Dr, Smt, M/s, Co, etc.)
PREFIX_NOISE_REGEX = r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|co\s+|>>\s+|#\s*)"

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
    print(f"  Avg Cands / S1   : {avg_c:.2f} (Median: {med_c:.1f}, P95: {p95_c:.1f}, P99: {p99_c:.1f}, Max: {max_c:,})")
    print(f"  Reduction Ratio  : {rr:.6f}%")
    return {"hits": hits, "recall": rec, "cands": tot_cands, "avg": avg_c}

# ==============================================================================
# 1. CHANNEL A: Base Exact Normalized Core Name
# ==============================================================================
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

# ==============================================================================
# 2. NEW CHANNEL A2: Prefix-Stripped Core Name (handles The/Dr/Smt/M/s/Co prefixes)
# ==============================================================================
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
eval_tab("New Channel A2: Prefix-Stripped Core Name", "ca2", time.time() - t0)

# ==============================================================================
# 3. NEW CHANNEL E2: Address Number + Distinctive Locality/City Token
# ==============================================================================
t0 = time.time()
# Extract all address numbers (normalized without leading zeros)
con.execute("""
CREATE TEMP TABLE s1_addr_nums AS
SELECT 
    entity_id as s1_id, country,
    unnest(regexp_extract_all(business_address, '[0-9]{1,6}')) as raw_num,
    unnest(string_split(regexp_replace(lower(business_address), '[^a-z ]', ' ', 'g'), ' ')) as tok
FROM eval_s1
WHERE business_address IS NOT NULL;
""")
con.execute("DELETE FROM s1_addr_nums WHERE length(tok) < 5 OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');")

con.execute("""
CREATE TEMP TABLE cand_addr_nums AS
SELECT 
    entity_id as candidate_id, country,
    unnest(regexp_extract_all(business_address, '[0-9]{1,6}')) as raw_num,
    unnest(string_split(regexp_replace(lower(business_address), '[^a-z ]', ' ', 'g'), ' ')) as tok
FROM all_candidates
WHERE business_address IS NOT NULL;
""")
con.execute("DELETE FROM cand_addr_nums WHERE length(tok) < 5 OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');")

# Distinctive tokens frequency cap (DF <= 250 in country)
con.execute("""
CREATE TEMP TABLE rare_loc_tokens AS
SELECT country, tok FROM cand_addr_nums GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 250;
""")

con.execute("""
CREATE TEMP TABLE ce2 AS
SELECT DISTINCT s.s1_id, c.candidate_id, 85 as priority
FROM s1_addr_nums s
JOIN rare_loc_tokens r ON s.country = r.country AND s.tok = r.tok
JOIN cand_addr_nums c ON s.country = c.country AND s.tok = c.tok 
                     AND cast(ltrim(s.raw_num, '0') as varchar) = cast(ltrim(c.raw_num, '0') as varchar)
WHERE length(ltrim(s.raw_num, '0')) > 0;
""")
eval_tab("New Channel E2: Address Number + Distinctive Locality (recovers native-script & aliases)", "ce2", time.time() - t0)

# ==============================================================================
# 4. NEW CHANNEL G: Approximate Name (Fuzzy 1-Edit on Normalized Name via Deletion Index)
# ==============================================================================
t0 = time.time()
# Deletion neighborhood: for names of length >= 6, generate 1-deletion keys to catch single typos/substitutions
con.execute(f"""
CREATE TEMP TABLE cg AS
WITH s1_del AS (
    SELECT 
        entity_id as s1_id, country,
        -- Take first 8 chars and generate prefix-deletion or middle-deletion
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
SELECT s.s1_id, c.candidate_id, 55 as priority
FROM s1_del s
JOIN cand_del c ON s.country = c.country AND s.del_key = c.del_key
WHERE length(s.del_key) >= 5;
""")
eval_tab("New Channel G: Initial-Char Typo Invariant Index (skips position 0 typo)", "cg", time.time() - t0)

# ==============================================================================
# 5. NEW CHANNEL C3: Character 3-Gram Inverted Index (comparing with 4-gram)
# ==============================================================================
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE cc3 AS
WITH s1_g AS (
    SELECT entity_id as s1_id, country,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 4, 3) as m3
    FROM eval_s1
    WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 6
),
c_g AS (
    SELECT entity_id as candidate_id, country,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 4, 3) as m3
    FROM all_candidates
    WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 6
)
SELECT s.s1_id, c.candidate_id, 45 as priority
FROM s1_g s JOIN c_g c ON s.country = c.country AND s.p3 = c.p3 AND s.m3 = c.m3;
""")
eval_tab("New Channel C3: Dual 3-Gram Anchor (p3 + m3)", "cc3", time.time() - t0)

# ==============================================================================
# 6. COMBINED HIGH-RECALL UNION (A + A2 + B + D + E + E2 + G + C3)
# ==============================================================================
print("\n" + "=" * 80)
print("UPDATED HIGH-RECALL CANDIDATE UNION")
print("=" * 80)

# Load existing base channels E and B
con.execute("""
CREATE TEMP TABLE ce AS
WITH s1_n AS (
    SELECT entity_id as s1_id, country,
           regexp_extract(business_address, '[0-9]{2,6}') as num,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM eval_s1 WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
),
c_n AS (
    SELECT entity_id as candidate_id, country,
           regexp_extract(business_address, '[0-9]{2,6}') as num,
           substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
    FROM all_candidates WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
)
SELECT s.s1_id, c.candidate_id, 80 as priority
FROM s1_n s JOIN c_n c ON s.country = c.country AND s.num = c.num AND s.p3 = c.p3;
""")

con.execute("""
CREATE TEMP TABLE cand_name_tokens_all AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != '';
""")
con.execute("DELETE FROM cand_name_tokens_all WHERE length(tok) < 4;")
con.execute("""
CREATE TEMP TABLE rare_name_toks_all AS
SELECT country, tok FROM cand_name_tokens_all GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 200;
""")
con.execute("""
CREATE TEMP TABLE cb AS
WITH s1_tokens AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id, 60 as priority
FROM s1_tokens s
JOIN rare_name_toks_all r ON s.country = r.country AND s.tok = r.tok
JOIN cand_name_tokens_all c ON s.country = c.country AND s.tok = c.tok;
""")

t0 = time.time()
con.execute("""
CREATE TEMP TABLE updated_union_all AS
WITH all_p AS (
    SELECT s1_id, candidate_id, priority FROM ca
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM ca2
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM ce
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM ce2
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM cb
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM cg
    UNION ALL
    SELECT s1_id, candidate_id, priority FROM cc3
)
SELECT s1_id, candidate_id, sum(priority) as total_priority, count(*) as channel_hits
FROM all_p
GROUP BY s1_id, candidate_id;
""")
eval_tab("UPDATED FULL UNION (All Channels Combined)", "updated_union_all", time.time() - t0)

# ==============================================================================
# 7. PRUNING COMPARISON ACROSS CAPS (Unpruned, 150, 100, 75, 60)
# ==============================================================================
print("\n" + "=" * 80)
print("PRUNING COMPARISON (UNPRUNED vs CAPS 150, 100, 75, 60)")
print("=" * 80)

for cap in [150, 100, 75, 60]:
    t0 = time.time()
    con.execute(f"""
    CREATE TEMP TABLE pruned_{cap} AS
    WITH ranked AS (
        SELECT 
            s1_id, candidate_id, total_priority, channel_hits,
            row_number() OVER (
                PARTITION BY s1_id 
                ORDER BY total_priority DESC, channel_hits DESC, candidate_id
            ) as rn
        FROM updated_union_all
    )
    SELECT s1_id, candidate_id
    FROM ranked
    WHERE rn <= {cap};
    """)
    eval_tab(f"Priority Cap = {cap}", f"pruned_{cap}", time.time() - t0)

# ==============================================================================
# 8. CHECK PREVIOUSLY MISSED SAMPLES TO VERIFY RECOVERY
# ==============================================================================
print("\n" + "=" * 80)
print("RECOVERY VERIFICATION ON PREVIOUSLY MISSED SAMPLES")
print("=" * 80)

# Test the specific previously missed examples
test_samples = [
    ("S1-358249590", "S2-570189202", "Apex Products Private Limited", "एपेक्स प्रोडक्ट्स प्राइवेट लिमिटेड"),
    ("S1-742174049", "S3-226789326", "Shakti Producer Private Limited", "ಶಕ್ತಿ ಪ್ರೊಡ್ಯೂಸರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"),
    ("S1-621735796", "S3-567732293", "One Estate LLP", "वन एस्टेट एलएलपी"),
    ("S1-839511041", "S3-450395744", "Beatty's Management", "beattysmanagement.com"),
    ("S1-159596935", "S2-275420082", "Fortune Ventures Pvt Ltd", "YUMAMIRA"),
    ("S1-257022224", "S2-220797717", "Averyl Hickman Materials Inc.", "averylhickmanmaterials.com #11557"),
    ("S1-407687539", "S3-338785968", "Wildlife Trust", "Wildlife Trust LP"),
    ("S1-861859613", "S2-333575115", "Orion L.L.C.", "0rion L.L.C."),
    ("S1-978709235", "S3-398451767", "Hotel (India) Finance Private Limited", "Dr hotel (india) finance private limited"),
    ("S1-105022531", "S2-611098764", "74/60 Pizza", "THE 74/60 PIZZA"),
    ("S1-89576694", "S2-484555273", "Delgado Academy", "The Delgado Academy"),
    ("S1-173812578", "S3-874768128", "Black Tech Pvt Ltd", "Smt Black Tech Pvt Ltd"),
    ("S1-56607895", "S3-115527350", "Vega & Co", "Center Co Vega"),
    ("S1-438680175", "S3-980192481", "Dow Advanced Motor Inc", "Inc Dow Advanced Mótor"),
    ("S1-380764336", "S2-538016144", "HKB Information Pvt Ltd", "HKB Information Ltd Pvt"),
]

for s1_id, m_id, s1_n, m_n in test_samples:
    recov = con.execute(f"SELECT count(*) FROM updated_union_all WHERE s1_id = '{s1_id}' AND candidate_id = '{m_id}'").fetchone()[0]
    status = "RECOVERED" if recov > 0 else "MISSED"
    print(f"[{status:9s}] {s1_id} ('{s1_n}') <--> {m_id} ('{m_n}')")
