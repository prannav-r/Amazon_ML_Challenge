import duckdb
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Inspecting True Pairs with Indic / Non-ASCII Script Names ---")
pairs = con.execute(f"""
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
    p.s1_id,
    p.matched_id,
    c.src,
    s1.country,
    s1.business_name as s1_name,
    c.business_name as match_name,
    s1.business_address as s1_addr,
    c.business_address as match_addr
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN combined c ON p.matched_id = c.entity_id
WHERE regexp_matches(c.business_name, '[^\\x00-\\x7F]')
LIMIT 10;
""").fetchall()

for row in pairs:
    print(f"\nS1: {row[0]} | Country: {row[3]}")
    print(f"  S1 Name  : {row[4]}")
    print(f"  Match Name ({row[2]}): {row[5]}")
    print(f"  S1 Addr  : {row[6]}")
    print(f"  Match Addr: {row[7]}")
