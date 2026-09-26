import duckdb
import time

con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Checking Country Consistency in Ground Truth Matches ---")
t0 = time.time()

# Let's create tables for S1, S2, S3, and unnest ground truth matches
con.execute(f"""
CREATE TEMP TABLE s1 AS 
SELECT entity_id, business_name, business_address, country 
FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true);
""")

con.execute(f"""
CREATE TEMP TABLE s2 AS 
SELECT entity_id, business_name, business_address, country 
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true);
""")

con.execute(f"""
CREATE TEMP TABLE s3 AS 
SELECT entity_id, business_name, business_address, country 
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true);
""")

# Unnest ground truth into pairs
con.execute(f"""
CREATE TEMP TABLE pairs AS
SELECT 
    source1_entity_id as s1_id,
    unnest(string_split(matched_entity_ids, ',')) as matched_id
FROM read_csv('{base_train}/train_ground_truth.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != '';
""")

print(f"Loaded tables and unnested pairs in {time.time()-t0:.2f}s")

total_pairs = con.execute("SELECT count(*) FROM pairs").fetchone()[0]
print(f"Total positive match pairs in training: {total_pairs:,}")

# Let's count how many matches are S2 vs S3
s2_s3_dist = con.execute("""
SELECT 
    CASE WHEN matched_id LIKE 'S2-%' THEN 'S2'
         WHEN matched_id LIKE 'S3-%' THEN 'S3'
         ELSE 'OTHER'
    END as src,
    count(*) as cnt
FROM pairs
GROUP BY src
""").fetchall()
print("\nMatch pairs by source:", s2_s3_dist)

# Now join pairs with s1 and (s2 union s3) to check country match
con.execute("""
CREATE TEMP TABLE s2_s3_combined AS
SELECT entity_id, business_name, business_address, country FROM s2
UNION ALL
SELECT entity_id, business_name, business_address, country FROM s3;
""")

con.execute("""
CREATE TEMP TABLE pair_details AS
SELECT 
    p.s1_id,
    p.matched_id,
    s1.country as s1_country,
    c.country as match_country,
    s1.business_name as s1_name,
    c.business_name as match_name,
    s1.business_address as s1_address,
    c.business_address as match_address
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN s2_s3_combined c ON p.matched_id = c.entity_id;
""")

country_match_stats = con.execute("""
SELECT 
    count(*) as total_evaluated_pairs,
    count(CASE WHEN s1_country = match_country THEN 1 END) as same_country,
    count(CASE WHEN s1_country != match_country THEN 1 END) as diff_country,
    count(CASE WHEN s1_country IS NULL OR match_country IS NULL THEN 1 END) as null_country
FROM pair_details
""").fetchall()

print("\n--- Country Match Analysis on Ground Truth Pairs ---")
print("Stats:", country_match_stats)
tot_eval, same_c, diff_c, null_c = country_match_stats[0]
print(f"Total evaluated pairs: {tot_eval:,}")
print(f"Same country: {same_c:,} ({same_c/tot_eval*100:.4f}%)")
print(f"Different country: {diff_c:,} ({diff_c/tot_eval*100:.4f}%)")
print(f"Null country: {null_c:,}")

if diff_c > 0:
    diff_examples = con.execute("""
    SELECT s1_id, matched_id, s1_country, match_country, s1_name, match_name, s1_address, match_address
    FROM pair_details
    WHERE s1_country != match_country
    LIMIT 10
    """).fetchall()
    print("\nExamples of different country matches:")
    for ex in diff_examples:
        print(ex)
else:
    print("\nRESULT: EXACTLY 0% of true matches cross country boundaries! Every single positive match has s1_country == match_country!")
