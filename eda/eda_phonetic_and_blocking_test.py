import duckdb
import re
import string
from collections import defaultdict
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

con = duckdb.connect()
base_train = "student_resource/dataset/train"

print("--- Testing Transliteration, Abbreviations, and Blocking Rules Empirically ---")

# Pull 200,000 true pairs across US and India with text
pairs_data = con.execute(f"""
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
    s1.business_name as s1_name,
    c.business_name as match_name,
    s1.business_address as s1_addr,
    c.business_address as match_addr
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN combined c ON p.matched_id = c.entity_id
USING SAMPLE 50000 (reservoir);
""").fetchall()

print(f"Sampled {len(pairs_data):,} ground truth pairs.")

# Let's inspect transliterations and phonetic variations in India vs US
# Common transliterations: e.g. Laxmi vs Lakshmi, Choudhary vs Chowdhary, Prasad vs Prashad, etc.
translit_examples = []
abbrev_examples = []

def clean_tokens(text):
    if not text: return []
    # lower and replace punctuation
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    return [t for t in text.split() if len(t) > 1]

# Soundex implementation for empirical test
def soundex(name):
    if not name: return ""
    name = name.upper()
    name = re.sub(r'[^A-Z]', '', name)
    if not name: return ""
    first = name[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = [first]
    prev = mapping.get(first, '')
    for char in name[1:]:
        curr = mapping.get(char, '')
        if curr != '' and curr != prev:
            encoded.append(curr)
        prev = curr
    res = "".join(encoded)
    res = (res + "0000")[:4]
    return res

# Measure recall of various candidate generation / blocking keys on these 50k true pairs!
key_hits = defaultdict(int)
total = len(pairs_data)

for row in pairs_data:
    s1_id, m_id, country, s1_n, m_n, s1_a, m_a = row
    s1_tokens = set(clean_tokens(s1_n))
    m_tokens = set(clean_tokens(m_n))
    
    s1_a_tokens = set(clean_tokens(s1_a))
    m_a_tokens = set(clean_tokens(m_a))
    
    # 1. Exact country match
    # 2. At least one name token overlap
    common_name_tokens = s1_tokens & m_tokens
    if common_name_tokens:
        key_hits['name_token_overlap'] += 1
        
    # 3. First token of name exact match
    s1_tok_list = clean_tokens(s1_n)
    m_tok_list = clean_tokens(m_n)
    if s1_tok_list and m_tok_list and s1_tok_list[0] == m_tok_list[0]:
        key_hits['first_name_token_exact'] += 1
        
    # 4. Soundex of first name token
    if s1_tok_list and m_tok_list and soundex(s1_tok_list[0]) == soundex(m_tok_list[0]):
        key_hits['first_token_soundex'] += 1
        
    # 5. Soundex of all tokens overlap
    s1_soundex = {soundex(t) for t in s1_tok_list}
    m_soundex = {soundex(t) for t in m_tok_list}
    if s1_soundex & m_soundex:
        key_hits['soundex_token_overlap'] += 1

    # 6. Character 3-gram overlap in name (at least 2 shared 3-grams)
    def char_3grams(text):
        t = " " + re.sub(r'[^a-z0-9]', '', (text or "").lower()) + " "
        return {t[i:i+3] for i in range(len(t)-2)} if len(t) >= 3 else set()
    
    s1_3g = char_3grams(s1_n)
    m_3g = char_3grams(m_n)
    if len(s1_3g & m_3g) >= 2:
        key_hits['name_3gram_overlap_ge_2'] += 1
    if len(s1_3g & m_3g) >= 1:
        key_hits['name_3gram_overlap_ge_1'] += 1

    # 7. Address token overlap
    common_addr_tokens = s1_a_tokens & m_a_tokens
    if common_addr_tokens:
        key_hits['addr_token_overlap'] += 1

    # 8. Combined: (name_token_overlap OR (addr_token_overlap AND name_3gram_ge_1))
    if common_name_tokens or (common_addr_tokens and len(s1_3g & m_3g) >= 1):
        key_hits['union_name_or_addr_with_3gram'] += 1

    # Look for interesting transliteration/abbreviation patterns
    if country == 'India' and len(translit_examples) < 10:
        if s1_tok_list and m_tok_list and s1_tok_list[0] != m_tok_list[0]:
            if len(s1_3g & m_3g) > 2:
                translit_examples.append((s1_n, m_n))

    if country == 'US' and len(abbrev_examples) < 10:
        if s1_tok_list and m_tok_list and s1_tok_list[0] != m_tok_list[0]:
            abbrev_examples.append((s1_n, m_n))

print("\n--- Empirical Recall of Individual Blocking Keys on 50,000 True Pairs ---")
for k, hits in sorted(key_hits.items(), key=lambda x: -x[1]):
    print(f"  {k:30s}: {hits:,} / {total:,} ({hits/total*100:.2f}% recall)")

print("\n--- Examples of Name Discrepancies in India ---")
for s1, mn in translit_examples:
    print(f"  S1: '{s1}' <--> MATCH: '{mn}'")

print("\n--- Examples of Name Discrepancies in US ---")
for s1, mn in abbrev_examples:
    print(f"  S1: '{s1}' <--> MATCH: '{mn}'")
