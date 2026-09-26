import duckdb
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Comparing Source 1, Source 2, and Source 3 Specific Characteristics ---")

for src_name, path in [
    ("Source 1", f"{base_train}/train_source1.tsv"),
    ("Source 2", f"{base_train}/train_source2.tsv"),
    ("Source 3", f"{base_train}/train_source3.tsv"),
]:
    stats = con.execute(f"""
    SELECT 
        count(*) as total,
        avg(length(business_name)) as avg_name_len,
        avg(length(business_address)) as avg_addr_len,
        count(CASE WHEN business_name = upper(business_name) THEN 1 END) as all_upper_names,
        count(CASE WHEN business_name = lower(business_name) THEN 1 END) as all_lower_names,
        count(CASE WHEN regexp_matches(business_name, '[^\\x00-\\x7F]') THEN 1 END) as non_ascii_names,
        count(CASE WHEN regexp_matches(business_name, '\\.com|www\\.') THEN 1 END) as web_names,
        count(CASE WHEN regexp_matches(business_name, '[0-9]') THEN 1 END) as digit_names,
        count(CASE WHEN business_address IS NULL OR trim(business_address) = '' THEN 1 END) as missing_addr
    FROM read_csv('{path}', delim='\\t', header=true, quote='', all_varchar=true)
    """).fetchall()[0]
    total = stats[0]
    print(f"\n{src_name} ({total:,} rows):")
    print(f"  Avg name len: {stats[1]:.1f}, Avg addr len: {stats[2]:.1f}")
    print(f"  All-UPPERCASE names: {stats[3]:,} ({stats[3]/total*100:.2f}%)")
    print(f"  All-lowercase names: {stats[4]:,} ({stats[4]/total*100:.2f}%)")
    print(f"  Non-ASCII (Indic/Diacritic) names: {stats[5]:,} ({stats[5]/total*100:.2f}%)")
    print(f"  Web / URL names (.com, www): {stats[6]:,} ({stats[6]/total*100:.2f}%)")
    print(f"  Digits in names: {stats[7]:,} ({stats[7]/total*100:.2f}%)")
    print(f"  Missing address: {stats[8]:,} ({stats[8]/total*100:.2f}%)")

# Compare True Matches from S2 vs S3
print("\n--- Comparing True Matches S1-S2 vs S1-S3 ---")
match_comparison = con.execute(f"""
WITH pairs AS (
    SELECT 
        source1_entity_id as s1_id,
        unnest(string_split(matched_entity_ids, ',')) as matched_id
    FROM read_csv('{base_train}/train_ground_truth.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
),
s1 AS (
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
),
combined AS (
    SELECT entity_id, business_name, business_address, country, 'S2' as src
    FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, business_address, country, 'S3' as src
    FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
)
SELECT 
    c.src,
    count(*) as total_matches,
    count(CASE WHEN lower(trim(s1.business_name)) = lower(trim(c.business_name)) THEN 1 END) as exact_case_name,
    count(CASE WHEN regexp_matches(c.business_name, '[^\\x00-\\x7F]') THEN 1 END) as non_ascii_match_name,
    count(CASE WHEN c.business_address IS NULL OR trim(c.business_address) = '' THEN 1 END) as missing_addr_matches
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN combined c ON p.matched_id = c.entity_id
GROUP BY c.src;
""").fetchall()

for row in match_comparison:
    src, tot, exact_n, non_asc, miss_a = row
    print(f"\n{src} Matches ({tot:,} total):")
    print(f"  Exact case-insensitive name match: {exact_n:,} ({exact_n/tot*100:.2f}%)")
    print(f"  Non-ASCII match names: {non_asc:,} ({non_asc/tot*100:.2f}%)")
    print(f"  Missing address in match: {miss_a:,} ({miss_a/tot*100:.2f}%)")
