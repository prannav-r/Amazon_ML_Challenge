import duckdb
import re

con = duckdb.connect()
base_train = "student_resource/dataset/train"
base_test = "student_resource/dataset/test"

print("--- Investigating S2/S3 Characteristics, Duplicates, and France Test Set ---")

# 1. Check nulls / empty fields across all files
for fname, path in [
    ("train_s1", f"{base_train}/train_source1.tsv"),
    ("train_s2", f"{base_train}/train_source2.tsv"),
    ("train_s3", f"{base_train}/train_source3.tsv"),
    ("test_s1", f"{base_test}/test_source1.tsv"),
    ("test_s2", f"{base_test}/test_source2.tsv"),
    ("test_s3", f"{base_test}/test_source3.tsv"),
]:
    stats = con.execute(f"""
    SELECT 
        count(*) as total,
        count(CASE WHEN business_name IS NULL OR trim(business_name) = '' THEN 1 END) as empty_name,
        count(CASE WHEN business_address IS NULL OR trim(business_address) = '' THEN 1 END) as empty_addr,
        count(CASE WHEN country IS NULL OR trim(country) = '' THEN 1 END) as empty_country,
        avg(length(business_name)) as avg_name_len,
        avg(length(business_address)) as avg_addr_len
    FROM read_csv('{path}', delim='\\t', header=true, quote='', all_varchar=true)
    """).fetchall()[0]
    print(f"{fname:8s}: {stats[0]:,} rows | Empty Names: {stats[1]} | Empty Addrs: {stats[2]} | Empty Country: {stats[3]} | Avg Name Len: {stats[4]:.1f} | Avg Addr Len: {stats[5]:.1f}")

# 2. Check duplicate / near-duplicate records within S2 and S3
print("\n--- Duplicate / Redundant Records in S2 and S3 ---")
for src_name, path in [("train_s2", f"{base_train}/train_source2.tsv"), ("train_s3", f"{base_train}/train_source3.tsv")]:
    dups = con.execute(f"""
    WITH t AS (
        SELECT 
            lower(trim(business_name)) as b_name,
            lower(trim(business_address)) as b_addr,
            country,
            count(*) as cnt
        FROM read_csv('{path}', delim='\\t', header=true, quote='', all_varchar=true)
        GROUP BY 1, 2, 3
    )
    SELECT 
        sum(cnt) as total_rows,
        count(*) as unique_name_addr_pairs,
        sum(CASE WHEN cnt > 1 THEN cnt ELSE 0 END) as rows_in_duplicate_groups,
        max(cnt) as max_duplicates_for_single_pair
    FROM t
    """).fetchall()[0]
    print(f"{src_name:8s}: Total rows: {dups[0]:,} | Unique (name, addr): {dups[1]:,} | Rows sharing exact (name, addr): {dups[2]:,} ({dups[2]/dups[0]*100:.2f}%) | Max dup count: {dups[3]}")

# 3. Check France records in test set
print("\n--- Inspecting France Test Records ---")
fr_samples = con.execute(f"""
SELECT entity_id, business_name, business_address, country
FROM read_csv('{base_test}/test_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country = 'France'
LIMIT 10
""").fetchall()
for s in fr_samples:
    print(f"  {s[0]} | Name: '{s[1]}' | Addr: '{s[2]}'")
