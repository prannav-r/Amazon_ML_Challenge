import duckdb
import sys
import io
import re
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Analyzing Common Abbreviations in Names and Addresses ---")

# Pull 50k pairs to examine word-level token replacements
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
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{base_train}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{base_train}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
)
SELECT 
    s1.country,
    s1.business_name as s1_name,
    c.business_name as match_name,
    s1.business_address as s1_addr,
    c.business_address as match_addr
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN combined c ON p.matched_id = c.entity_id
USING SAMPLE 50000 (reservoir);
""").fetchall()

print(f"Loaded {len(pairs):,} pairs.")

def tokenize(text):
    if not text: return []
    # Replace common punctuation with space, keep alphanumeric
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 0]

name_abbrev_counter = Counter()
addr_abbrev_counter = Counter()

KNOWN_NAME_ABBREVS = {
    ('pvt', 'private'), ('ltd', 'limited'), ('corp', 'corporation'),
    ('inc', 'incorporated'), ('co', 'company'), ('mfg', 'manufacturing'),
    ('tech', 'technology'), ('intl', 'international'), ('assn', 'association'),
    ('dept', 'department'), ('mgmt', 'management'), ('grp', 'group'),
    ('ent', 'enterprises'), ('ind', 'industries'), ('serv', 'services'),
    ('soln', 'solutions'), ('syst', 'systems'), ('hldg', 'holdings'),
    ('engr', 'engineering'), ('dist', 'distribution'), ('comm', 'commercial')
}

KNOWN_ADDR_ABBREVS = {
    ('rd', 'road'), ('st', 'street'), ('ave', 'avenue'), ('blvd', 'boulevard'),
    ('dr', 'drive'), ('ln', 'lane'), ('ct', 'court'), ('pl', 'place'),
    ('hwy', 'highway'), ('pkwy', 'parkway'), ('apt', 'apartment'),
    ('ste', 'suite'), ('fl', 'floor'), ('bldg', 'building'), ('nr', 'near'),
    ('opp', 'opposite'), ('chq', 'chowk'), ('sec', 'sector'), ('dist', 'district'),
    ('no', 'number'), ('pl', 'plot'), ('bl', 'block'), ('w', 'west'), ('e', 'east'),
    ('n', 'north'), ('s', 'south')
}

# Normalize bidirectional check
canonical_name_map = {}
for a, b in KNOWN_NAME_ABBREVS:
    canonical_name_map[a] = b

canonical_addr_map = {}
for a, b in KNOWN_ADDR_ABBREVS:
    canonical_addr_map[a] = b

found_name_abbrevs = Counter()
found_addr_abbrevs = Counter()

for row in pairs:
    country, s1_n, m_n, s1_a, m_a = row
    s1_nt = set(tokenize(s1_n))
    m_nt = set(tokenize(m_n))
    
    for a, b in KNOWN_NAME_ABBREVS:
        if (a in s1_nt and b in m_nt) or (b in s1_nt and a in m_nt):
            found_name_abbrevs[f"{a} <-> {b}"] += 1

    s1_at = set(tokenize(s1_a))
    m_at = set(tokenize(m_a))
    for a, b in KNOWN_ADDR_ABBREVS:
        if (a in s1_at and b in m_at) or (b in s1_at and a in m_at):
            found_addr_abbrevs[f"{a} <-> {b}"] += 1

total = len(pairs)
print("\n--- Name Abbreviations Found (Sample 50,000 True Pairs) ---")
for k, v in found_name_abbrevs.most_common(15):
    print(f"  {k:25s}: {v:,} ({v/total*100:.2f}%)")

print("\n--- Address Abbreviations Found (Sample 50,000 True Pairs) ---")
for k, v in found_addr_abbrevs.most_common(15):
    print(f"  {k:25s}: {v:,} ({v/total*100:.2f}%)")
