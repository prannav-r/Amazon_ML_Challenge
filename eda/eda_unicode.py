import duckdb
import sys
import io
import re

# Ensure stdout handles UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

con = duckdb.connect()
base_train = "student_resource/dataset/train"
base_test = "student_resource/dataset/test"

print("--- Inspecting Non-ASCII and Script Characters ---")

# Let's check non-ascii characters in train_s1, train_s2, train_s3
for name, path in [
    ("train_s1", f"{base_train}/train_source1.tsv"),
    ("train_s2", f"{base_train}/train_source2.tsv"),
    ("train_s3", f"{base_train}/train_source3.tsv"),
    ("test_s1", f"{base_test}/test_source1.tsv"),
]:
    stats = con.execute(f"""
    SELECT 
        count(*) as total,
        count(CASE WHEN regexp_matches(business_name, '[^\\x00-\\x7F]') THEN 1 END) as non_ascii_names,
        count(CASE WHEN regexp_matches(business_address, '[^\\x00-\\x7F]') THEN 1 END) as non_ascii_addr
    FROM read_csv('{path}', delim='\\t', header=true, quote='', all_varchar=true)
    """).fetchall()[0]
    print(f"{name}: Total {stats[0]:,} | Non-ASCII Names: {stats[1]:,} ({stats[1]/stats[0]*100:.2f}%) | Non-ASCII Addr: {stats[2]:,} ({stats[2]/stats[0]*100:.2f}%)")

# Let's inspect some non-ascii examples
samples = con.execute(f"""
SELECT country, business_name, business_address
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE regexp_matches(business_name, '[^\\x00-\\x7F]') OR regexp_matches(business_address, '[^\\x00-\\x7F]')
LIMIT 15
""").fetchall()

print("\nSamples of Non-ASCII entries in Train S1:")
for c, n, a in samples:
    print(f"  [{c}] Name: '{n}' | Addr: '{a}'")
