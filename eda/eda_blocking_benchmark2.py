import duckdb
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Benchmarking Alternative Blocking Rules ---")

# Let's inspect the missed true matches from earlier to see WHY they were missed
# and what features connect them!
query = f"""
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
    s1.country,
    s1.business_name as s1_n,
    c.business_name as m_n,
    s1.business_address as s1_a,
    c.business_address as m_a
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN combined c ON p.matched_id = c.entity_id
USING SAMPLE 20 (reservoir);
"""
samples = con.execute(query).fetchall()
print("\nExamining 20 Random Ground Truth Pairs to identify invariant blocking anchors:")
for s in samples:
    print(f"\nCountry: {s[2]}")
    print(f"  S1: '{s[3]}' | '{s[5]}'")
    print(f"  MT: '{s[4]}' | '{s[6]}'")
