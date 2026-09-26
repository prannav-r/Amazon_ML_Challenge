"""
Comprehensive Candidate Generation & Blocking Benchmark
Benchmarks Channels A through F (and Soundex) independently and in combination
across a held-out set of 10,000 S1 training entities against 10.3M S2+S3 records.
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
print("PHASE 3: CANDIDATE GENERATION / BLOCKING BENCHMARK")
print("=" * 80)

# Unambiguous legal entity designators regex
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

# 1. Sample 10,000 held-out S1 entities (stratified by country)
print("\n1. Selecting 10,000 held-out evaluation S1 entities...")
t0 = time.time()
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

# Ground truth for evaluation S1 entities
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
print(f"Loaded {total_s1:,} evaluation S1 entities with {total_true_pairs:,} true positive pairs in {time.time()-t0:.2f}s.")

# 2. Load all 10.3M candidate records (S2 + S3)
print("\n2. Loading unified candidate table from S2 + S3 (10.3M records)...")
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE all_candidates AS
SELECT entity_id, business_name, business_address, country, 'S2' as src
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
UNION ALL
SELECT entity_id, business_name, business_address, country, 'S3' as src
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true);
""")
cand_count = con.execute("SELECT count(*) FROM all_candidates").fetchone()[0]
print(f"Loaded {cand_count:,} candidate records in {time.time()-t0:.2f}s.")

# Total possible comparisons for reduction ratio calculation
total_possible_comparisons = total_s1 * cand_count

def evaluate_channel(name, table_name, execution_time):
    """Compute all required metrics for a candidate table."""
    # Count total pairs
    total_candidates = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    
    # Recall against ground truth
    hits = con.execute(f"""
    SELECT count(*) 
    FROM eval_gt g
    JOIN {table_name} c ON g.s1_id = c.s1_id AND g.true_match_id = c.candidate_id
    """).fetchone()[0]
    recall = (hits / total_true_pairs) * 100.0 if total_true_pairs else 0.0
    
    # Candidates per S1 distribution
    per_s1 = con.execute(f"""
    SELECT c.s1_id, count(*) as cnt
    FROM {table_name} c
    GROUP BY c.s1_id
    """).fetchall()
    
    # Include S1 entities with 0 candidates
    counts_map = {row[0]: row[1] for row in per_s1}
    all_counts = [counts_map.get(row[0], 0) for row in con.execute("SELECT entity_id FROM eval_s1").fetchall()]
    
    avg_cand = float(np.mean(all_counts))
    median_cand = float(np.median(all_counts))
    p95_cand = float(np.percentile(all_counts, 95))
    p99_cand = float(np.percentile(all_counts, 99))
    max_cand = int(np.max(all_counts)) if all_counts else 0
    
    reduction_ratio = (1.0 - (total_candidates / total_possible_comparisons)) * 100.0
    
    print(f"\n[{name}]")
    print(f"  Runtime            : {execution_time:.2f}s")
    print(f"  Total Candidates   : {total_candidates:,}")
    print(f"  True Matches Found : {hits:,} / {total_true_pairs:,}")
    print(f"  Recall             : {recall:.2f}%")
    print(f"  Average Cands / S1 : {avg_cand:.2f}")
    print(f"  Median Cands / S1  : {median_cand:.1f}")
    print(f"  95th Percentile    : {p95_cand:.1f}")
    print(f"  99th Percentile    : {p99_cand:.1f}")
    print(f"  Max Bucket Size    : {max_cand:,}")
    print(f"  Reduction Ratio    : {reduction_ratio:.6f}%")
    
    return {
        "name": name,
        "table": table_name,
        "runtime": execution_time,
        "total_cands": total_candidates,
        "hits": hits,
        "recall": recall,
        "avg": avg_cand,
        "median": median_cand,
        "p95": p95_cand,
        "p99": p99_cand,
        "max": max_cand,
        "reduction_ratio": reduction_ratio
    }

# ==============================================================================
# CHANNEL A: EXACT NORMALIZED CORE NAME (Alphanumeric with Unambiguous Legal Stripped)
# ==============================================================================
print("\n--- Benchmarking Channel A: Exact Normalized Core Name ---")
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE chan_a AS
WITH s1_clean AS (
    SELECT 
        entity_id as s1_id,
        country,
        regexp_replace(
            regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM eval_s1
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
cand_clean AS (
    SELECT 
        entity_id as candidate_id,
        country,
        regexp_replace(
            regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
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
res_a = evaluate_channel("Channel A: Exact Normalized Core Name", "chan_a", time.time() - t0)

# ==============================================================================
# CHANNEL B: RARE NAME TOKENS (Inverted Index with Document Frequency Cap)
# ==============================================================================
print("\n--- Benchmarking Channel B: Rare / Distinctive Name Tokens (DF <= 200) ---")
t0 = time.time()
# Tokenize and compute token frequencies in all_candidates by country
con.execute("""
CREATE TEMP TABLE cand_name_tokens AS
SELECT 
    country,
    entity_id as candidate_id,
    unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates
WHERE business_name IS NOT NULL AND trim(business_name) != '';
""")

# Delete short or empty tokens
con.execute("DELETE FROM cand_name_tokens WHERE length(tok) < 4;")

# Measure document frequency per token per country and keep tokens with frequency <= 200
con.execute("""
CREATE TEMP TABLE rare_name_tokens AS
SELECT country, tok, count(*) as df
FROM cand_name_tokens
GROUP BY country, tok
HAVING count(*) BETWEEN 2 AND 200;
""")

con.execute("""
CREATE TEMP TABLE chan_b AS
WITH s1_tokens AS (
    SELECT 
        country,
        entity_id as s1_id,
        unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id
FROM s1_tokens s
JOIN rare_name_tokens r ON s.country = r.country AND s.tok = r.tok
JOIN cand_name_tokens c ON s.country = c.country AND s.tok = c.tok
WHERE length(s.tok) >= 4;
""")
res_b = evaluate_channel("Channel B: Rare Name Tokens (2 <= DF <= 200)", "chan_b", time.time() - t0)

# ==============================================================================
# CHANNEL C: CHARACTER 3-GRAM SIMILARITY / INDEXING
# ==============================================================================
print("\n--- Benchmarking Channel C: Character 3-Gram Inverted Index ---")
t0 = time.time()
# Extract rare 3-grams from alphanumeric core name (length >= 5)
con.execute(f"""
CREATE TEMP TABLE chan_c AS
WITH s1_ngrams AS (
    SELECT 
        entity_id as s1_id,
        country,
        substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as prefix_4,
        substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as suffix_4
    FROM eval_s1
    WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
),
cand_ngrams AS (
    SELECT 
        entity_id as candidate_id,
        country,
        substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), 1, 4) as prefix_4,
        substring(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'), -4) as suffix_4
    FROM all_candidates
    WHERE length(regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g')) >= 5
)
SELECT s.s1_id, c.candidate_id
FROM s1_ngrams s
JOIN cand_ngrams c ON s.country = c.country AND (s.prefix_4 = c.prefix_4 AND s.suffix_4 = c.suffix_4);
""")
res_c = evaluate_channel("Channel C: Character 4-gram Prefix+Suffix Inverted Index", "chan_c", time.time() - t0)

# ==============================================================================
# CHANNEL D: DISTINCTIVE ADDRESS TOKEN BLOCKING
# ==============================================================================
print("\n--- Benchmarking Channel D: Distinctive Address Tokens ---")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE cand_addr_tokens AS
SELECT 
    country,
    entity_id as candidate_id,
    unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
FROM all_candidates
WHERE business_address IS NOT NULL AND trim(business_address) != '';
""")

# Filter out address stopwords
con.execute("""
DELETE FROM cand_addr_tokens 
WHERE length(tok) < 5 
   OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
              'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
              'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');
""")

con.execute("""
CREATE TEMP TABLE rare_addr_tokens AS
SELECT country, tok, count(*) as df
FROM cand_addr_tokens
GROUP BY country, tok
HAVING count(*) BETWEEN 2 AND 150;
""")

con.execute("""
CREATE TEMP TABLE chan_d AS
WITH s1_addr_tokens AS (
    SELECT 
        country,
        entity_id as s1_id,
        unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
    FROM eval_s1
    WHERE business_address IS NOT NULL AND trim(business_address) != ''
)
SELECT DISTINCT s.s1_id, c.candidate_id
FROM s1_addr_tokens s
JOIN rare_addr_tokens r ON s.country = r.country AND s.tok = r.tok
JOIN cand_addr_tokens c ON s.country = c.country AND s.tok = c.tok
WHERE length(s.tok) >= 5;
""")
res_d = evaluate_channel("Channel D: Distinctive Address Tokens (2 <= DF <= 150)", "chan_d", time.time() - t0)

# ==============================================================================
# CHANNEL E: ADDRESS NUMERIC ANCHORS (PIN / Street Number + Name Prefix)
# ==============================================================================
print("\n--- Benchmarking Channel E: Address Numeric Anchors (PIN / Street Number + 3-char Name Prefix) ---")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE chan_e AS
WITH s1_numeric AS (
    SELECT 
        entity_id as s1_id,
        country,
        regexp_extract(business_address, '[0-9]{2,6}') as addr_num,
        substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as name_p3
    FROM eval_s1
    WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
),
cand_numeric AS (
    SELECT 
        entity_id as candidate_id,
        country,
        regexp_extract(business_address, '[0-9]{2,6}') as addr_num,
        substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as name_p3
    FROM all_candidates
    WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
)
SELECT s.s1_id, c.candidate_id
FROM s1_numeric s
JOIN cand_numeric c ON s.country = c.country AND s.addr_num = c.addr_num AND s.name_p3 = c.name_p3;
""")
res_e = evaluate_channel("Channel E: Address Number + 3-char Name Prefix", "chan_e", time.time() - t0)

# ==============================================================================
# CHANNEL F: NATIVE-SCRIPT / TRANSLITERATION FALLBACK (Latin Address & Numbers)
# ==============================================================================
print("\n--- Benchmarking Channel F: Native-Script Fallback (Address Number + Locality Token) ---")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE chan_f AS
WITH s1_translit AS (
    SELECT 
        entity_id as s1_id,
        country,
        regexp_extract(business_address, '[0-9]{2,6}') as addr_num,
        split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2) as addr_tok2
    FROM eval_s1
    WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2)) >= 4
),
cand_translit AS (
    SELECT 
        entity_id as candidate_id,
        country,
        regexp_extract(business_address, '[0-9]{2,6}') as addr_num,
        split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2) as addr_tok2
    FROM all_candidates
    WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{2,6}') != ''
      AND length(split_part(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g'), ' ', 2)) >= 4
)
SELECT s.s1_id, c.candidate_id
FROM s1_translit s
JOIN cand_translit c ON s.country = c.country AND s.addr_num = c.addr_num AND s.addr_tok2 = c.addr_tok2;
""")
res_f = evaluate_channel("Channel F: Address Number + Locality Anchor", "chan_f", time.time() - t0)

# ==============================================================================
# PHONETIC / SOUNDEX EXPERIMENT (Separately measured as requested)
# ==============================================================================
print("\n--- Experiment: Phonetic / Soundex Blocking ---")
t0 = time.time()
con.execute("""
CREATE TEMP TABLE chan_soundex AS
WITH s1_sndx AS (
    SELECT 
        entity_id as s1_id,
        country,
        substring(split_part(regexp_replace(lower(trim(business_name)), '[^a-z ]', ' ', 'g'), ' ', 1), 1, 4) as sndx_key
    FROM eval_s1
    WHERE length(split_part(regexp_replace(lower(trim(business_name)), '[^a-z ]', ' ', 'g'), ' ', 1)) >= 4
),
cand_sndx AS (
    SELECT 
        entity_id as candidate_id,
        country,
        substring(split_part(regexp_replace(lower(trim(business_name)), '[^a-z ]', ' ', 'g'), ' ', 1), 1, 4) as sndx_key
    FROM all_candidates
    WHERE length(split_part(regexp_replace(lower(trim(business_name)), '[^a-z ]', ' ', 'g'), ' ', 1)) >= 4
)
SELECT s.s1_id, c.candidate_id
FROM s1_sndx s
JOIN cand_sndx c ON s.country = c.country AND s.sndx_key = c.sndx_key;
""")
res_sndx = evaluate_channel("Experiment: Soundex/Phonetic Prefix", "chan_soundex", time.time() - t0)

# ==============================================================================
# COMBINATIONS VIA UNION
# ==============================================================================
print("\n" + "=" * 80)
print("EVALUATING COMBINATIONS USING UNION")
print("=" * 80)

# Union A + B
t0 = time.time()
con.execute("""
CREATE TEMP TABLE union_ab AS
SELECT s1_id, candidate_id FROM chan_a
UNION
SELECT s1_id, candidate_id FROM chan_b;
""")
res_ab = evaluate_channel("Union: Channel A + B", "union_ab", time.time() - t0)

# Union A + B + D + E + F
t0 = time.time()
con.execute("""
CREATE TEMP TABLE union_all AS
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
res_union_all = evaluate_channel("Union: Channels A + B + C + D + E + F", "union_all", time.time() - t0)

# ==============================================================================
# CANDIDATE PRUNING (Max Candidates Cap per S1 Entity)
# ==============================================================================
print("\n" + "=" * 80)
print("PRUNING BENCHMARK: Capping Max Candidates per S1")
print("=" * 80)

for cap in [50, 30, 20]:
    t0 = time.time()
    con.execute(f"""
    CREATE TEMP TABLE pruned_{cap} AS
    WITH ranked AS (
        SELECT 
            s1_id, 
            candidate_id,
            row_number() OVER (PARTITION BY s1_id ORDER BY candidate_id) as rn
        FROM union_all
    )
    SELECT s1_id, candidate_id
    FROM ranked
    WHERE rn <= {cap};
    """)
    evaluate_channel(f"Pruned Union (Cap = {cap} per S1)", f"pruned_{cap}", time.time() - t0)

# ==============================================================================
# INSPECT EXAMPLES OF CAPTURED & MISSED MATCHES
# ==============================================================================
print("\n" + "=" * 80)
print("INSPECTING EXAMPLES OF CAPTURED AND MISSED MATCHES")
print("=" * 80)

# Check which channel captured what
captured_examples = con.execute(f"""
SELECT 
    g.s1_id,
    g.true_match_id,
    s.country,
    s.business_name as s1_name,
    c.business_name as match_name,
    s.business_address as s1_addr,
    c.business_address as match_addr
FROM eval_gt g
JOIN eval_s1 s ON g.s1_id = s.entity_id
JOIN all_candidates c ON g.true_match_id = c.entity_id
JOIN union_all u ON g.s1_id = u.s1_id AND g.true_match_id = u.candidate_id
LIMIT 5;
""").fetchall()

print("\n--- Examples of Captured True Matches ---")
for ex in captured_examples:
    print(f"[{ex[2]}] S1: '{ex[3]}' | '{ex[5]}'")
    print(f"       MT: '{ex[4]}' | '{ex[6]}'\n")

missed_examples = con.execute(f"""
SELECT 
    g.s1_id,
    g.true_match_id,
    s.country,
    s.business_name as s1_name,
    c.business_name as match_name,
    s.business_address as s1_addr,
    c.business_address as match_addr
FROM eval_gt g
JOIN eval_s1 s ON g.s1_id = s.entity_id
JOIN all_candidates c ON g.true_match_id = c.entity_id
LEFT JOIN union_all u ON g.s1_id = u.s1_id AND g.true_match_id = u.candidate_id
WHERE u.candidate_id IS NULL
LIMIT 5;
""").fetchall()

print("\n--- Examples of Missed True Matches ---")
for ex in missed_examples:
    print(f"[{ex[2]}] S1: '{ex[3]}' | '{ex[5]}'")
    print(f"       MT: '{ex[4]}' | '{ex[6]}'\n")

print("\nBenchmark completed successfully.")
