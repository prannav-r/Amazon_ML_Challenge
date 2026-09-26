"""
Deep Dive Audit of Missed True Matches (Phase 3 Revision)
Extracts and categorizes all ground-truth matches missed by the unpruned A+B+C+D+E+F union.
"""

import duckdb
import sys
import io
import time
import re
from collections import Counter, defaultdict

# Ensure stdout handles UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
con = duckdb.connect()

base_train = "student_resource/dataset/train"

print("=" * 80)
print("AUDIT: 14.48% MISSED TRUE MATCHES IN CANDIDATE GENERATION")
print("=" * 80)

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

# 1. Set up evaluation S1 entities (same fixed sample for 100% reproducibility)
print("\n1. Loading evaluation S1 entities and full candidates table...")
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

total_gt = con.execute("SELECT count(*) FROM eval_gt").fetchone()[0]
print(f"Total evaluation ground truth pairs: {total_gt:,}")

# 2. Re-create the unpruned A+B+C+D+E+F union
print("2. Generating Channels A through F...")

# Channel A
con.execute(fr"""
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
SELECT s.s1_id, c.candidate_id
FROM s1_clean s JOIN cand_clean c ON s.country = c.country AND s.norm_key = c.norm_key
WHERE length(s.norm_key) >= 3;
""")

# Channel E
con.execute("""
CREATE TEMP TABLE chan_e AS
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
SELECT s.s1_id, c.candidate_id
FROM s1_n s JOIN c_n c ON s.country = c.country AND s.num = c.num AND s.p3 = c.p3;
""")

# Channel B
con.execute("""
CREATE TEMP TABLE cand_name_tokens AS
SELECT country, entity_id as candidate_id,
       unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates WHERE business_name IS NOT NULL AND trim(business_name) != '';
""")
con.execute("DELETE FROM cand_name_tokens WHERE length(tok) < 4;")
con.execute("""
CREATE TEMP TABLE rare_name_tokens AS
SELECT country, tok FROM cand_name_tokens GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 200;
""")
con.execute("""
CREATE TEMP TABLE chan_b AS
WITH s1_tokens AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id
FROM s1_tokens s
JOIN rare_name_tokens r ON s.country = r.country AND s.tok = r.tok
JOIN cand_name_tokens c ON s.country = c.country AND s.tok = c.tok;
""")

# Channel D
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
SELECT country, tok FROM cand_addr_tokens GROUP BY country, tok HAVING count(*) BETWEEN 2 AND 150;
""")
con.execute("""
CREATE TEMP TABLE chan_d AS
WITH s1_tokens AS (
    SELECT country, entity_id as s1_id,
           unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1 WHERE business_address IS NOT NULL AND trim(business_address) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id
FROM s1_tokens s
JOIN rare_addr_tokens r ON s.country = r.country AND s.tok = r.tok
JOIN cand_addr_tokens c ON s.country = c.country AND s.tok = c.tok;
""")

# Channel F
con.execute("""
CREATE TEMP TABLE chan_f AS
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
SELECT s.s1_id, c.candidate_id
FROM s1_translit s JOIN c_translit c ON s.country = c.country AND s.num = c.num AND s.loc = c.loc;
""")

# Channel C
con.execute(f"""
CREATE TEMP TABLE chan_c AS
WITH s1_ngrams AS (
    SELECT entity_id as s1_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
    FROM eval_s1
    WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
),
cand_ngrams AS (
    SELECT entity_id as candidate_id, country,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as p4,
           substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as s4
    FROM all_candidates
    WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
)
SELECT s.s1_id, c.candidate_id
FROM s1_ngrams s JOIN cand_ngrams c ON s.country = c.country AND s.p4 = c.p4 AND s.s4 = c.s4;
""")

# Unified candidate set
con.execute("""
CREATE TEMP TABLE unpruned_union AS
SELECT s1_id, candidate_id FROM chan_a
UNION
SELECT s1_id, candidate_id FROM chan_b
UNION
SELECT s1_id, candidate_id FROM chan_c
UNION
SELECT s1_id, candidate_id FROM chan_d
UNION
SELECT s1_id, candidate_id FROM chan_e
UNION
SELECT s1_id, candidate_id FROM chan_f;
""")

captured_hits = con.execute("""
SELECT count(*) FROM eval_gt g
JOIN unpruned_union u ON g.s1_id = u.s1_id AND g.true_match_id = u.candidate_id
""").fetchone()[0]

print(f"Captured by unpruned union: {captured_hits:,} / {total_gt:,} ({captured_hits/total_gt*100:.2f}%)")

# Extract ALL missed pairs with their full attributes
con.execute("""
CREATE TEMP TABLE missed_pairs AS
SELECT 
    g.s1_id,
    g.true_match_id,
    s.country,
    s.business_name as s1_name,
    c.business_name as match_name,
    s.business_address as s1_addr,
    c.business_address as match_addr,
    c.src
FROM eval_gt g
JOIN eval_s1 s ON g.s1_id = s.entity_id
JOIN all_candidates c ON g.true_match_id = c.entity_id
LEFT JOIN unpruned_union u ON g.s1_id = u.s1_id AND g.true_match_id = u.candidate_id
WHERE u.candidate_id IS NULL;
""")

missed_rows = con.execute("SELECT * FROM missed_pairs").fetchall()
total_missed = len(missed_rows)
print(f"Total missed true pairs: {total_missed:,} ({(total_missed/total_gt)*100:.2f}%)")

# 3. Categorize each missed pair
def clean_toks(s):
    if not s: return []
    return [t for t in re.sub(r'[^a-zA-Z0-9]', ' ', s.lower()).split() if len(t) > 1]

def extract_nums(s):
    if not s: return set()
    nums = re.findall(r'\b\d+\b', s)
    return {str(int(n)) for n in nums if n.isdigit()}

def is_indic(s):
    if not s: return False
    return any(0x0900 <= ord(c) <= 0x0D7F for c in s)

def levenshtein(s1, s2):
    if s1 == s2: return 0
    if len(s1) == 0: return len(s2)
    if len(s2) == 0: return len(s1)
    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)
    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = v1[:]
    return v1[len(s2)]

category_counts = Counter()
categorized_examples = defaultdict(list)

for row in missed_rows:
    s1_id, m_id, country, s1_n, m_n, s1_a, m_a, src = row
    s1_n = s1_n or ""
    m_n = m_n or ""
    s1_a = s1_a or ""
    m_a = m_a or ""

    s1_nt = set(clean_toks(s1_n))
    m_nt = set(clean_toks(m_n))
    s1_at = set(clean_toks(s1_a))
    m_at = set(clean_toks(m_a))

    s1_nums = extract_nums(s1_a)
    m_nums = extract_nums(m_a)

    shared_n_toks = s1_nt & m_nt
    shared_a_toks = s1_at & m_at
    shared_nums = s1_nums & m_nums

    alpha_s1 = re.sub(r'[^a-z0-9]', '', s1_n.lower())
    alpha_m = re.sub(r'[^a-z0-9]', '', m_n.lower())
    edit_dist = levenshtein(alpha_s1, alpha_m) if alpha_s1 and alpha_m else 999

    # Failure Category Determination
    cat = None
    if not m_a or m_a.strip().lower() in ('', 'none', 'null', 'nan'):
        cat = "Missing Match Address in S2/S3"
    elif is_indic(m_n) and not is_indic(s1_n):
        if shared_nums or shared_a_toks:
            cat = "Native-Script Name with Address Number/Locality Overlap"
        else:
            cat = "Native-Script Name with Extreme Address Discrepancy"
    elif edit_dist <= 2 and len(alpha_s1) >= 4:
        cat = "Minor Name Typo (Levenshtein <= 2)"
    elif len(alpha_s1) >= 4 and len(alpha_m) >= 4 and alpha_s1[0] != alpha_m[0] and edit_dist <= 3:
        cat = "Initial Character Substitution (e.g. 'G' vs '6', 'O' vs '0')"
    elif shared_n_toks and not shared_nums and not shared_a_toks:
        cat = "Shared Name Words but Disjoint Address"
    elif shared_nums and not shared_n_toks:
        cat = "Shared Address Number but Disjoint/Alias Name"
    elif not shared_n_toks and not shared_a_toks and not shared_nums:
        cat = "Completely Disjoint Alias/Trade-Name & Address"
    elif shared_a_toks and not shared_nums and not shared_n_toks:
        cat = "Shared Address Token but Disjoint Name and Disjoint Number"
    elif edit_dist <= 4:
        cat = "Moderate Name Fuzzy Variation (Levenshtein 3-4)"
    else:
        cat = "Other Complex Discrepancy"

    category_counts[cat] += 1
    if len(categorized_examples[cat]) < 5:
        categorized_examples[cat].append((s1_id, m_id, country, s1_n, m_n, s1_a, m_a))

print("\n--- Failure Categories of the Missed True Matches ---")
print(f"{'Category':60s} | {'Count':8s} | {'Percentage':10s}")
print("-" * 84)
for cat, count in category_counts.most_common():
    pct = (count / total_missed) * 100.0
    print(f"{cat:60s} | {count:8,d} | {pct:8.2f}%")

print("\n" + "=" * 80)
print("REPRESENTATIVE EXAMPLES ACROSS FAILURE CATEGORIES")
print("=" * 80)

ex_count = 0
for cat, ex_list in category_counts.most_common():
    print(f"\n>>> Category: {cat} ({category_counts[cat]:,} pairs, {category_counts[cat]/total_missed*100:.1f}%)")
    for ex in categorized_examples[cat][:3]:
        ex_count += 1
        print(f"  [{ex[2]}] S1 ({ex[0]}): '{ex[3]}'")
        print(f"           Addr: '{ex[5]}'")
        print(f"       MT ({ex[1]}): '{ex[4]}'")
        print(f"           Addr: '{ex[6]}'")
        print()
        if ex_count >= 25:
            break
    if ex_count >= 25:
        break
