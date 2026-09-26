import duckdb
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Samples of Non-ASCII Names in S2 and S3 ---")
samples_s2 = con.execute(f"""
SELECT country, business_name, business_address
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE regexp_matches(business_name, '[^\\x00-\\x7F]')
LIMIT 15
""").fetchall()

for c, n, a in samples_s2:
    print(f"  [S2][{c}] Name: '{n}' | Addr: '{a}'")
