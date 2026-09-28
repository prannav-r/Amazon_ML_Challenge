"""
Phase 9: Detailed Blocker Misses and Candidate Rank Analysis
Amazon ML Challenge 2026: Business Entity Resolution

Analyzes:
1. True matches missed by Cap 150:
   - Category A: Retrieved by at least one channel, but pruned by Cap 150 (Rank > 150).
   - Category B: Never retrieved by ANY of the 8 channels.
2. Candidate rank of true matches:
   - Recall at Cap 100, 150, 200, 250, 300, 500, 1000.
   - Total candidates, average/S1, median, P95, max.
   - Multi-channel vs single-channel firing patterns.
"""

import sys
import os
import io
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import CandidateBlocker, UNAMBIGUOUS_LEGAL_REGEX, PREFIX_NOISE_REGEX

gt_path = "output/eval_ground_truth.pkl"
base_train = "student_resource/dataset/train"

print("=" * 80)
print("PHASE 9: BLOCKER MISS ANALYSIS & CANDIDATE RANK DISTRIBUTION")
print("=" * 80)

# 1. Load Ground Truth and 3-way Split
print("\n[1/4] Loading ground truth metadata and establishing split...")
with open(gt_path, "rb") as f:
    meta = pickle.load(f)

gt_dict = meta["gt_dict"]
s1_list_all = meta["s1_list"]
s1_lookup = meta["s1_lookup"]

strata = defaultdict(list)
for s1 in s1_list_all:
    c = s1_lookup[s1]["country"]
    num_matches = len(gt_dict.get(s1, set()))
    if num_matches == 0:
        m_type = "zero"
    elif num_matches == 1:
        m_type = "single"
    else:
        m_type = "multi"
    strata[(c, m_type)].append(s1)

train_s1, dev_s1, holdout_s1 = [], [], []
rng = np.random.RandomState(42)
for (c, m_type), ids in sorted(strata.items()):
    shuffled = rng.permutation(ids)
    n = len(shuffled)
    n_train = int(n * 0.60)
    n_dev = int(n * 0.20)
    train_s1.extend(shuffled[:n_train])
    dev_s1.extend(shuffled[n_train:n_train + n_dev])
    holdout_s1.extend(shuffled[n_train + n_dev:])

print(f"  Dev S1 Count: {len(dev_s1):,}")
dev_gt_pairs = set()
for s in dev_s1:
    for c in gt_dict.get(s, set()):
        dev_gt_pairs.add((s, c))

total_dev_true = len(dev_gt_pairs)
print(f"  Dev True Matches: {total_dev_true:,}")

# 2. Setup DuckDB with Dev S1 and Full Candidate pool (Source 2 + Source 3 for US and India)
print("\n[2/4] Setting up DuckDB tables from raw train TSVs...")
con = duckdb.connect()

# Create Dev S1 table
dev_s1_rows = [
    (s, s1_lookup[s]["business_name"], s1_lookup[s]["business_address"], s1_lookup[s]["country"])
    for s in dev_s1
]
df_dev_s1 = pd.DataFrame(dev_s1_rows, columns=["entity_id", "business_name", "business_address", "country"])
con.register("dev_s1_table", df_dev_s1)

# Load candidate pool from train_source2 and train_source3 for US and India
t0_load = time.time()
con.execute(f"""
CREATE TEMP TABLE all_candidates AS
SELECT entity_id, business_name, business_address, country, 'S2' as src
FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN ('US', 'India')
UNION ALL
SELECT entity_id, business_name, business_address, country, 'S3' as src
FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
WHERE country IN ('US', 'India');
""")
cand_count = con.execute("SELECT count(*) FROM all_candidates").fetchone()[0]
print(f"  Loaded {cand_count:,} candidates (S2 + S3) for US and India in {time.time()-t0_load:.2f}s.")

# 3. Run CandidateBlocker with high cap (1,000) to capture unpruned candidates
print("\n[3/4] Running CandidateBlocker with Cap 1,000 to observe unpruned pool...")
blocker = CandidateBlocker(con)
t0 = time.time()
block_res = blocker.generate_candidates(
    s1_table_or_path="dev_s1_table",
    cand_table_or_path="all_candidates",
    output_table="dev_unpruned_candidates",
    max_candidates_per_s1=1000
)
t_block = time.time() - t0
print(f"  Blocking executed in {t_block:.2f}s.")

df_cands = con.execute("SELECT * FROM dev_unpruned_candidates").fetchdf()
print(f"  Total Candidates Generated: {len(df_cands):,}")

# 4. Analyze True Matches in the Unpruned Pool
print("\n[4/4] Analyzing True Match Retrieval & Rank Distribution...")

# Create mapping from (s1_id, candidate_id) -> record
cand_dict = {}
for r in df_cands.itertuples(index=False):
    cand_dict[(r.source1_entity_id, r.candidate_entity_id)] = r

cat_a_matches = []  # Retrieved, but rank > 150
cat_b_matches = []  # Not retrieved by any channel
cap_150_retained = []  # Retrieved with rank <= 150

for s1_id, cand_id in dev_gt_pairs:
    pair = (s1_id, cand_id)
    if pair in cand_dict:
        r = cand_dict[pair]
        if r.rank_order <= 150:
            cap_150_retained.append(r)
        else:
            cat_a_matches.append(r)
    else:
        cat_b_matches.append((s1_id, cand_id))

print("\n" + "=" * 80)
print("TRUE MATCH RETRIEVAL DECOMPOSITION AT CAP 150")
print("=" * 80)
print(f"Total True Matches on Dev               : {total_dev_true:,} (100.0%)")
print(f"Retained at Cap 150 (Rank <= 150)       : {len(cap_150_retained):,} ({len(cap_150_retained)/total_dev_true*100:.2f}%)")
print(f"Total Missed at Cap 150                 : {len(cat_a_matches) + len(cat_b_matches):,} ({(len(cat_a_matches) + len(cat_b_matches))/total_dev_true*100:.2f}%)")
print(f"  - Category A (Retrieved, Rank > 150)  : {len(cat_a_matches):,} ({len(cat_a_matches)/total_dev_true*100:.2f}% of all true, {len(cat_a_matches)/(len(cat_a_matches)+len(cat_b_matches))*100:.1f}% of misses)")
print(f"  - Category B (Never Retrieved by any) : {len(cat_b_matches):,} ({len(cat_b_matches)/total_dev_true*100:.2f}% of all true, {len(cat_b_matches)/(len(cat_a_matches)+len(cat_b_matches))*100:.1f}% of misses)")

# 5. Study Candidate Rank of True Matches across Caps
caps = [100, 150, 200, 250, 300, 500, 1000]
print("\n" + "=" * 80)
print("TRUE-MATCH RECALL & CANDIDATE VOLUME ACROSS CAPS")
print("=" * 80)
print(f"{'Cap':<6} | {'Total Cands':<12} | {'Avg/S1':<8} | {'Median':<8} | {'P95':<6} | {'Max':<6} | {'True Recall':<12} | {'True Matches':<12} | {'Growth vs 150':<14}")
print("-" * 95)

base_cands_150 = len(df_cands[df_cands["rank_order"] <= 150])

for c in caps:
    df_cap = df_cands[df_cands["rank_order"] <= c]
    counts_per_s1 = df_cap.groupby("source1_entity_id")["candidate_entity_id"].count().to_dict()
    # Add zero-candidate S1s
    for s in dev_s1:
        if s not in counts_per_s1:
            counts_per_s1[s] = 0
            
    total_cands = len(df_cap)
    vals = list(counts_per_s1.values())
    avg_cands = float(np.mean(vals))
    median_cands = float(np.median(vals))
    p95_cands = float(np.percentile(vals, 95))
    max_cands = int(np.max(vals))
    
    # Count true matches
    true_retained = sum(1 for (s, cand) in dev_gt_pairs if (s, cand) in cand_dict and cand_dict[(s, cand)].rank_order <= c)
    recall_pct = true_retained / total_dev_true * 100.0
    growth = (total_cands - base_cands_150) / base_cands_150 * 100.0 if base_cands_150 > 0 else 0.0
    
    print(f"{c:<6} | {total_cands:<12,} | {avg_cands:<8.1f} | {median_cands:<8.0f} | {p95_cands:<6.0f} | {max_cands:<6.0f} | {recall_pct:<6.2f}% ({true_retained}) | {true_retained:<12,} | {growth:+6.1f}%")

# 6. Rank Distribution of Category A Matches
if cat_a_matches:
    ranks_a = [r.rank_order for r in cat_a_matches]
    print("\n" + "=" * 80)
    print("RANK BREAKDOWN OF CATEGORY A MISSES (RETRIEVED BUT PRUNED BY CAP 150)")
    print("=" * 80)
    print(f"Rank 151 - 200 : {sum(1 for r in ranks_a if 151 <= r <= 200)} matches")
    print(f"Rank 201 - 250 : {sum(1 for r in ranks_a if 201 <= r <= 250)} matches")
    print(f"Rank 251 - 300 : {sum(1 for r in ranks_a if 251 <= r <= 300)} matches")
    print(f"Rank 301 - 500 : {sum(1 for r in ranks_a if 301 <= r <= 500)} matches")
    print(f"Rank 501 - 1000: {sum(1 for r in ranks_a if 501 <= r <= 1000)} matches")
    print(f"Rank > 1000    : {sum(1 for r in ranks_a if r > 1000)} matches")

# 7. Channel Firing Analysis for Category A Matches
print("\n" + "=" * 80)
print("CHANNEL FIRING ANALYSIS FOR CATEGORY A MATCHES")
print("=" * 80)
chan_counts = defaultdict(int)
for r in cat_a_matches:
    if r.fired_chan_a: chan_counts["Chan A (Exact Name)"] += 1
    if r.fired_chan_a2: chan_counts["Chan A2 (Prefix Core)"] += 1
    if r.fired_chan_b: chan_counts["Chan B (Rare Name Toks)"] += 1
    if r.fired_chan_c: chan_counts["Chan C (4-gram Name)"] += 1
    if r.fired_chan_d: chan_counts["Chan D (Address Toks)"] += 1
    if r.fired_chan_e: chan_counts["Chan E (Addr Num + Name P3)"] += 1
    if r.fired_chan_e2: chan_counts["Chan E2 (Addr Num + Locality)"] += 1
    if r.fired_chan_g: chan_counts["Chan G (Initial Typo)"] += 1

for chan, cnt in sorted(chan_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {chan:<30}: {cnt} matches ({cnt/len(cat_a_matches)*100:.1f}%)")

# 8. Sample Inspection of Category B Matches (Never Retrieved)
print("\n" + "=" * 80)
print("SAMPLE INSPECTION OF CATEGORY B MISSES (FETCHING RAW FROM DUCKDB)")
print("=" * 80)
con.execute("CREATE TEMP TABLE missed_pairs (s1_id VARCHAR, cand_id VARCHAR);")
con.executemany("INSERT INTO missed_pairs VALUES (?, ?)", [(s, c) for s, c in cat_b_matches[:15]])

res_missed = con.execute("""
SELECT 
    m.s1_id, m.cand_id,
    s.business_name as s1_name, c.business_name as cand_name,
    s.business_address as s1_addr, c.business_address as cand_addr,
    s.country
FROM missed_pairs m
JOIN dev_s1_table s ON m.s1_id = s.entity_id
JOIN all_candidates c ON m.cand_id = c.entity_id;
""").fetchall()

for idx, r in enumerate(res_missed):
    print(f"Miss #{idx+1}: S1={r[0]} vs Cand={r[1]} [{r[6]}]")
    print(f"  S1 Name  : {r[2]}")
    print(f"  Cand Name: {r[3]}")
    print(f"  S1 Addr  : {r[4]}")
    print(f"  Cand Addr: {r[5]}")
    print("-" * 60)
