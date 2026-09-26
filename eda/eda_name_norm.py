import duckdb
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Testing Name Normalization Coverage ---")

pairs = con.execute(f"""
WITH pairs AS (
    SELECT 
        source1_entity_id as s1_id,
        unnest(string_split(matched_entity_ids, ',')) as matched_id
    FROM read_csv('{base_train}/train_ground_truth.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
),
s1 AS (
    SELECT entity_id, business_name, country 
    FROM read_csv('{base_train}/train_source1.tsv', delim='\\t', header=true, quote='', all_varchar=true)
),
combined AS (
    SELECT entity_id, business_name, country
    FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, country
    FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
)
SELECT 
    s1.country,
    s1.business_name as s1_name,
    c.business_name as match_name
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN combined c ON p.matched_id = c.entity_id
USING SAMPLE 50000 (reservoir);
""").fetchall()

LEGAL = r'\b(pvt|ltd|limited|private|inc|incorporated|corp|corporation|llc|llp|co|company)\b'
URLS = r'(\.com|\.in|\.org|\.net|\.co|\.us|www\.)'

def norm_alphanum(s):
    if not s: return ""
    s = s.lower()
    s = re.sub(URLS, '', s)
    s = re.sub(LEGAL, '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

exact_raw = 0
exact_case = 0
exact_norm = 0
for c, s1, m in pairs:
    s1 = s1 or ""
    m = m or ""
    if s1 == m: exact_raw += 1
    if s1.lower() == m.lower(): exact_case += 1
    if norm_alphanum(s1) == norm_alphanum(m) and len(norm_alphanum(s1)) > 2:
        exact_norm += 1

total = len(pairs)
print(f"Sample size: {total:,}")
print(f"Exact raw: {exact_raw:,} ({exact_raw/total*100:.2f}%)")
print(f"Exact case-insensitive: {exact_case:,} ({exact_case/total*100:.2f}%)")
print(f"Normalized alphanumeric match (legal suffix + URL stripped): {exact_norm:,} ({exact_norm/total*100:.2f}%)")
