import duckdb
import time

con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Analyzing Ground Truth Structure ---")
t0 = time.time()

# Let's inspect ground truth
query = f"""
CREATE TEMP TABLE gt AS 
SELECT 
    source1_entity_id,
    matched_entity_ids,
    CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0
         ELSE length(matched_entity_ids) - length(replace(matched_entity_ids, ',', '')) + 1
    END as match_count
FROM read_csv('{base_train}/train_ground_truth.tsv', delim='\\t', header=true, quote='', all_varchar=true);
"""
con.execute(query)

print(f"Loaded GT table in {time.time()-t0:.2f}s")

# Total rows and match_count distribution
stats = con.execute("""
SELECT 
    count(*) as total_s1,
    count(CASE WHEN match_count = 0 THEN 1 END) as singletons,
    count(CASE WHEN match_count = 1 THEN 1 END) as one_match,
    count(CASE WHEN match_count > 1 THEN 1 END) as multi_match,
    min(match_count) as min_matches,
    max(match_count) as max_matches,
    avg(match_count) as avg_matches
FROM gt
""").fetchall()

print("Ground Truth Summary:", stats)
total_s1, singletons, one_match, multi_match, min_m, max_m, avg_m = stats[0]
print(f"Total S1: {total_s1:,}")
print(f"Singletons (0 matches): {singletons:,} ({singletons/total_s1*100:.2f}%)")
print(f"One match (1 match): {one_match:,} ({one_match/total_s1*100:.2f}%)")
print(f"Multiple matches (>1): {multi_match:,} ({multi_match/total_s1*100:.2f}%)")
print(f"Max matches: {max_m}, Avg matches per S1: {avg_m:.4f}")

# Distribution of match counts
match_dist = con.execute("""
SELECT match_count, count(*) as freq 
FROM gt 
GROUP BY match_count 
ORDER BY match_count
""").fetchall()
print("\nMatch count frequency distribution:")
for mc, freq in match_dist[:15]:
    print(f"  {mc} matches: {freq:,} ({freq/total_s1*100:.2f}%)")
if len(match_dist) > 15:
    print(f"  ... up to {match_dist[-1][0]} matches")
