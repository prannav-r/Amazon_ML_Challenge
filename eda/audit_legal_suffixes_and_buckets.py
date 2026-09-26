"""
Audit Legal Suffix Dictionary & Global Bucket Sizes
1. Separate unambiguous legal designators from generic business words.
2. Measure global bucket sizes (how many records share each normalized name key).
3. Report largest normalized-name buckets and their sizes under different normalization schemes.
4. Verify whether aggressive normalization creates dangerous unrelated candidate buckets.
"""

import duckdb
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
con = duckdb.connect()

base_train = "student_resource/dataset/train"

print("=" * 80)
print("AUDIT: LEGAL SUFFIXES VS BUSINESS DESCRIPTORS & GLOBAL BUCKET SIZES")
print("=" * 80)

# Unambiguous legal entity designators ONLY (corporation, limited, LLC, etc.)
UNAMBIGUOUS_LEGAL_REGEX = (
    r"(?:[,\s\-\(\[\#]+|\b)("
    r"private\s+limited|public\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|"
    r"limited\s+liability\s+company|limited\s+liability\s+partnership|"
    r"corporation|incorporated|corp|inc|llc|llp|plc|ltd|limited|"
    r"societe\s+a\s+responsabilite\s+limitee|societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|"
    r"societe\s+par\s+actions\s+simplifiee|entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|"
    r"societe\s+anonyme|societe\s+civile|sarlu|sasu|sarl|sas|eurl|sci|snc|sa|ei|et\s+fils|fils|"
    r"प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|लिमिटेड|एलएलपी"
    r")(?:[,\s\.\)\]]*)$"
)

# Overly broad business words that might cause bucket explosion if stripped:
# e.g., 'enterprises', 'industries', 'technologies', 'solutions', 'services', 'group', 'holdings'
BROAD_BUSINESS_WORDS_REGEX = (
    r"(?:[,\s\-\(\[\#]+|\b)("
    r"enterprises|industries|technologies|solutions|services|holdings|group|associates|partners|consultancy|consulting"
    r")(?:[,\s\.\)\]]*)$"
)

print("\n1. Measuring global bucket sizes on 2,206,821 Source 1 records...")

# Test Scheme 1: Raw alphanumeric (strip punctuation only)
t0 = time.time()
scheme1 = con.execute(f"""
WITH t AS (
    SELECT 
        regexp_replace(lower(trim(business_name)), '[^a-z0-9]', '', 'g') as norm_key,
        count(*) as bucket_size
    FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
    GROUP BY 1
)
SELECT 
    count(*) as unique_keys,
    max(bucket_size) as max_bucket,
    avg(bucket_size) as avg_bucket,
    count(CASE WHEN bucket_size > 10 THEN 1 END) as buckets_gt_10,
    count(CASE WHEN bucket_size > 100 THEN 1 END) as buckets_gt_100
FROM t;
""").fetchall()[0]
print(f"Scheme 1 (Punctuation/space strip only): {scheme1[0]:,} unique keys | Max bucket: {scheme1[1]:,} | Buckets > 10: {scheme1[3]:,} | Buckets > 100: {scheme1[4]} (in {time.time()-t0:.2f}s)")

# Test Scheme 2: Unambiguous Legal Suffixes ONLY stripped
t0 = time.time()
scheme2 = con.execute(f"""
WITH cleaned AS (
    SELECT 
        regexp_replace(
            regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
t AS (
    SELECT norm_key, count(*) as bucket_size
    FROM cleaned
    GROUP BY 1
)
SELECT 
    count(*) as unique_keys,
    max(bucket_size) as max_bucket,
    avg(bucket_size) as avg_bucket,
    count(CASE WHEN bucket_size > 10 THEN 1 END) as buckets_gt_10,
    count(CASE WHEN bucket_size > 100 THEN 1 END) as buckets_gt_100
FROM t;
""").fetchall()[0]
print(f"Scheme 2 (Unambiguous Legal Suffixes stripped): {scheme2[0]:,} unique keys | Max bucket: {scheme2[1]:,} | Buckets > 10: {scheme2[3]:,} | Buckets > 100: {scheme2[4]} (in {time.time()-t0:.2f}s)")

# Test Scheme 3: Unambiguous Legal + Aggressive Business Words (Enterprises, Services, etc.) stripped
t0 = time.time()
scheme3 = con.execute(f"""
WITH cleaned AS (
    SELECT 
        regexp_replace(
            regexp_replace(
                regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
                '{BROAD_BUSINESS_WORDS_REGEX}', '', 'g'
            ),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
),
t AS (
    SELECT norm_key, count(*) as bucket_size
    FROM cleaned
    GROUP BY 1
)
SELECT 
    count(*) as unique_keys,
    max(bucket_size) as max_bucket,
    avg(bucket_size) as avg_bucket,
    count(CASE WHEN bucket_size > 10 THEN 1 END) as buckets_gt_10,
    count(CASE WHEN bucket_size > 100 THEN 1 END) as buckets_gt_100
FROM t;
""").fetchall()[0]
print(f"Scheme 3 (Aggressive stripping: Legal + Business words): {scheme3[0]:,} unique keys | Max bucket: {scheme3[1]:,} | Buckets > 10: {scheme3[3]:,} | Buckets > 100: {scheme3[4]} (in {time.time()-t0:.2f}s)")

# Let's inspect the largest buckets under Scheme 2 vs Scheme 3
print("\n--- Top 15 Largest Buckets under Scheme 2 (Unambiguous Legal Suffixes Only) ---")
top_s2 = con.execute(f"""
WITH cleaned AS (
    SELECT 
        business_name,
        regexp_replace(
            regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT norm_key, count(*) as cnt
FROM cleaned
WHERE length(norm_key) > 0
GROUP BY 1
ORDER BY 2 DESC
LIMIT 15;
""").fetchall()
for k, c in top_s2:
    print(f"  Key: '{k:25s}' -> {c:,} S1 entities")

print("\n--- Top 15 Largest Buckets under Scheme 3 (Aggressive Stripping) ---")
top_s3 = con.execute(f"""
WITH cleaned AS (
    SELECT 
        business_name,
        regexp_replace(
            regexp_replace(
                regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
                '{BROAD_BUSINESS_WORDS_REGEX}', '', 'g'
            ),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT norm_key, count(*) as cnt
FROM cleaned
WHERE length(norm_key) > 0
GROUP BY 1
ORDER BY 2 DESC
LIMIT 15;
""").fetchall()
for k, c in top_s3:
    print(f"  Key: '{k:25s}' -> {c:,} S1 entities")

# Now let's see what happens across S2 and S3 (10.3M records) for these top keys!
print("\n--- Checking Max Candidate Size on Combined S2 + S3 (10.3M records) ---")
top_s2_s3 = con.execute(f"""
WITH all_candidates AS (
    SELECT entity_id, business_name, country FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, country FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
),
cleaned AS (
    SELECT 
        country,
        regexp_replace(
            regexp_replace(lower(trim(business_name)), '{UNAMBIGUOUS_LEGAL_REGEX}', '', 'g'),
            '[^a-z0-9]', '', 'g'
        ) as norm_key
    FROM all_candidates
    WHERE business_name IS NOT NULL AND trim(business_name) != ''
)
SELECT norm_key, country, count(*) as cand_count
FROM cleaned
WHERE length(norm_key) > 0
GROUP BY 1, 2
ORDER BY 3 DESC
LIMIT 15;
""").fetchall()

for k, c, cnt in top_s2_s3:
    print(f"  [{c}] Key: '{k:25s}' -> {cnt:,} candidate records in S2+S3")
