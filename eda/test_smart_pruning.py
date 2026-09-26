"""
Test Smart Priority Pruning for Candidate Union
Evaluates candidate pruning by assigning priority scores based on matching channels:
Priority 1: Channel A (Exact core name match) - Highest confidence
Priority 2: Channel E (Address number + 3-char name prefix)
Priority 3: Channel B (Rare name token match)
Priority 4: Channel F (Address number + locality anchor)
Priority 5: Channel D (Distinctive address token match)
Priority 6: Channel C (4-gram prefix/suffix)
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
print("TESTING SMART PRIORITY PRUNING")
print("=" * 80)

# Re-run benchmark with channel priority tags
# We can run on the 5,208 evaluation entities from earlier
con.execute(f"""
CREATE TEMP TABLE eval_s1 AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'US'
USING SAMPLE 3000 (reservoir);
""")

con.execute(f"""
INSERT INTO eval_s1
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'India'
USING SAMPLE 2000 (reservoir);
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

total_s1 = con.execute("SELECT count(*) FROM eval_s1").fetchone()[0]
total_true_pairs = con.execute("SELECT count(*) FROM eval_gt").fetchone()[0]
print(f"Loaded {total_s1:,} eval S1 entities, {total_true_pairs:,} true match pairs.")

con.execute(f"""
CREATE TEMP TABLE all_candidates AS
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
UNION ALL
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true);
""")
print("Loaded candidates table.")

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

# Channel A
con.execute(f"""
CREATE TEMP TABLE ca AS
WITH s1_clean AS (
    SELECT entity_id as s1_id, country,
           regexp_replace(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
c_clean AS (
    SELECT entity_id as candidate_id, country,
           regexp_replace(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
    FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT s.s1_id, c.candidate_id, 100 as priority_score
FROM s1_clean s JOIN c_clean c ON s.country = c.country AND s.norm_key = c.norm_key
WHERE length(s.norm_key) >= 3;
""")

# Channel E (Number + 3-char name prefix)
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
SELECT s.s1_id, c.candidate_id, 80 as priority_score
FROM s1_n s JOIN c_n c ON s.country = c.country AND s.num = c.num AND s.p3 = c.p3;
""")

# Channel B (Rare Name Tokens)
con.execute("""
CREATE TEMP TABLE cand_name_tokens AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != '';
""")
con.execute("DELETE FROM cand_name_tokens WHERE length(tok) < 4;")
con.execute("""
CREATE TEMP TABLE rare_name_tokens AS
SELECT country, tok FROM cand_name_tokens GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 150;
""")
con.execute("""
CREATE TEMP TABLE cb AS
WITH s1_tokens AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id, 60 as priority_score
FROM s1_tokens s
JOIN rare_name_tokens r ON s.country = r.country AND s.tok = r.tok
JOIN cand_name_tokens c ON s.country = c.country AND s.tok = c.tok;
""")

# Channel D (Distinctive Address Tokens)
con.execute("""
CREATE TEMP TABLE cand_addr_tokens AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_address IS NOT NULL AND trim(business_address) != '';
""")
con.execute("""
DELETE FROM cand_addr_tokens 
WHERE length(tok) < 5 
   OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
              'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
              'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');
""")
con.execute("""
CREATE TEMP TABLE rare_addr_tokens AS
SELECT country, tok FROM cand_addr_tokens GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 100;
""")
con.execute("""
CREATE TEMP TABLE cd AS
WITH s1_tokens AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 WHERE business_address IS NOT NULL AND trim(business_address) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id, 40 as priority_score
FROM s1_tokens s
JOIN rare_addr_tokens r ON s.country = r.country AND s.tok = r.tok
JOIN cand_addr_tokens c ON s.country = c.country AND s.tok = c.tok;
""")

# Channel F (Number + Locality Anchor)
con.execute("""
CREATE TEMP TABLE cf AS
WITH s1_translit AS (
    SELECT entity_id as s1_id, country,
           regexp_extract(business_address, '[0-9]{2,6}') as num,
           split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2) as loc
    FROM eval_s1 WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2)) >= 4
),
c_translit AS (
    SELECT entity_id as candidate_id, country,
           regexp_extract(business_address, '[0-9]{2,6}') as num,
           split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2) as loc
    FROM all_candidates WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2)) >= 4
)
SELECT s.s1_id, c.candidate_id, 50 as priority_score
FROM s1_translit s JOIN c_translit c ON s.country = c.country AND s.num = c.num AND s.loc = c.loc;
""")

# Union with aggregated priority score (sum of priority scores across all firing channels)
print("\nCombining channels with multi-channel priority scoring...")
con.execute("""
CREATE TEMP TABLE combined_scored AS
WITH all_pairs AS (
    SELECT s1_id, candidate_id, priority_score FROM ca
    UNION ALL
    SELECT s1_id, candidate_id, priority_score FROM ce
    UNION ALL
    SELECT s1_id, candidate_id, priority_score FROM cb
    UNION ALL
    SELECT s1_id, candidate_id, priority_score FROM cf
    UNION ALL
    SELECT s1_id, candidate_id, priority_score FROM cd
)
SELECT s1_id, candidate_id, sum(priority_score) as total_priority, count(*) as channel_hits
FROM all_pairs
GROUP BY s1_id, candidate_id;
""")

total_union = con.execute("SELECT count(*) FROM combined_scored").fetchone()[0]
union_hits = con.execute("""
SELECT count(*) FROM eval_gt g
JOIN combined_scored c ON g.s1_id = c.s1_id AND g.true_match_id = c.candidate_id
""").fetchone()[0]
print(f"Full Union: {total_union:,} candidates ({total_union/total_s1:.1f} per S1) | Recall: {union_hits:,} / {total_true_pairs:,} ({union_hits/total_true_pairs*100:.2f}%)")

# Test Priority Pruning at caps 100, 50, 30, 20
print("\n--- Testing Priority-Ranked Pruning (Highest Priority First) ---")
for cap in [100, 60, 40, 25]:
    con.execute(f"""
    CREATE TEMP TABLE pruned_{cap} AS
    WITH ranked AS (
        SELECT 
            s1_id, candidate_id, total_priority, channel_hits,
            row_number() OVER (PARTITION BY s1_id ORDER BY total_priority DESC, channel_hits DESC, candidate_id) as rn
        FROM combined_scored
    )
    SELECT s1_id, candidate_id, total_priority, channel_hits
    FROM ranked
    WHERE rn <= {cap};
    """)
    p_count = con.execute(f"SELECT count(*) FROM pruned_{cap}").fetchone()[0]
    p_hits = con.execute(f"""
    SELECT count(*) FROM eval_gt g
    JOIN pruned_{cap} c ON g.s1_id = c.s1_id AND g.true_match_id = c.candidate_id
    """).fetchone()[0]
    rec = (p_hits / total_true_pairs) * 100.0
    print(f"Cap {cap:3d}: {p_count:,} candidates ({p_count/total_s1:.1f} avg/S1) | Recall: {p_hits:,} / {total_true_pairs:,} ({rec:.2f}%) [Retained {(p_hits/union_hits)*100:.2f}% of Union Recall]")
