import duckdb
import re
import string
from collections import Counter
import random
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

con = duckdb.connect()
base_train = "student_resource/dataset/train"
base_test = "student_resource/dataset/test"

print("--- Deep Dive EDA Script ---")

# Let's sample 100,000 ground truth pairs joined with S1 and (S2/S3)
# To be representative, we select pairs across US and India
sample_pairs = con.execute(f"""
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
    s1.business_address as s1_address,
    c.business_address as match_address
FROM pairs p
JOIN s1 ON p.s1_id = s1.entity_id
JOIN combined c ON p.matched_id = c.entity_id
USING SAMPLE 100000 (reservoir);
""").fetchall()

print(f"Sampled {len(sample_pairs):,} pairs for in-depth linguistic and structural analysis.")

# Legal suffixes to test
LEGAL_SUFFIXES = [
    r'\bpvt\s+ltd\b', r'\bltd\b', r'\blimited\b', r'\bprivate\s+limited\b',
    r'\binc\b', r'\bincorporated\b', r'\bcorp\b', r'\bcorporation\b',
    r'\bllc\b', r'\bllp\b', r'\bco\b', r'\bcompany\b', r'\benterprises\b',
    r'\bservices\b', r'\bgroup\b', r'\bindustries\b', r'\bassociates\b',
    r'\bholdings\b', r'\bpartners\b'
]
legal_regex = re.compile(r'|'.join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

def strip_punct_and_case(s):
    if not s: return ""
    s = s.lower()
    for c in string.punctuation:
        s = s.replace(c, " ")
    return " ".join(s.split())

def strip_legal(s):
    s = strip_punct_and_case(s)
    s = legal_regex.sub(" ", s)
    return " ".join(s.split())

# 2. How often names differ only by punctuation/case/legal suffixes
exact_raw_count = 0
exact_case_count = 0
exact_punct_count = 0
exact_legal_count = 0
diff_examples = []
legal_diff_examples = []

# 3. Name abbreviations
acronym_matches = 0
token_jaccard_scores = []

# 4 & 5. Address analysis: numbers, postal/PIN code regex
# US zip: 5 digits (\b\d{5}\b), India PIN: 6 digits (\b\d{6}\b), France postal: 5 digits (\b\d{5}\b)
pin_re = re.compile(r'\b\d{5,6}\b')
num_re = re.compile(r'\b\d+\b')

s1_has_pin = 0
match_has_pin = 0
both_have_pin = 0
pin_exact_match = 0
pin_conflict = 0

addr_exact_clean = 0
addr_token_jaccards = []
addr_token_reordered = 0
addr_missing_components = 0

# Source specific: S2 vs S3
s2_name_exact = 0
s2_count = 0
s3_name_exact = 0
s3_count = 0

for row in sample_pairs:
    s1_id, m_id, src, country, s1_n, m_n, s1_a, m_a = row
    s1_n = s1_n or ""
    m_n = m_n or ""
    s1_a = s1_a or ""
    m_a = m_a or ""

    if src == 'S2': s2_count += 1
    else: s3_count += 1

    # Name checks
    if s1_n == m_n:
        exact_raw_count += 1
        if src == 'S2': s2_name_exact += 1
        else: s3_name_exact += 1
    
    if s1_n.lower() == m_n.lower():
        exact_case_count += 1
    
    clean_s1_n = strip_punct_and_case(s1_n)
    clean_m_n = strip_punct_and_case(m_n)
    if clean_s1_n == clean_m_n:
        exact_punct_count += 1
    
    no_legal_s1 = strip_legal(s1_n)
    no_legal_m = strip_legal(m_n)
    if no_legal_s1 == no_legal_m:
        exact_legal_count += 1
        if clean_s1_n != clean_m_n and len(legal_diff_examples) < 10:
            legal_diff_examples.append((country, s1_n, m_n))
    else:
        if len(diff_examples) < 10:
            diff_examples.append((country, s1_n, m_n))

    # Check for abbreviation/acronym: e.g. "ABC Corp" vs "American Broadcasting Company" or initials
    s1_tokens = clean_s1_n.split()
    m_tokens = clean_m_n.split()
    if s1_tokens and m_tokens:
        s1_set = set(s1_tokens)
        m_set = set(m_tokens)
        jaccard = len(s1_set & m_set) / len(s1_set | m_set)
        token_jaccard_scores.append(jaccard)
        
        # Check if one is acronym of other
        s1_acronym = "".join([t[0] for t in s1_tokens if t])
        m_acronym = "".join([t[0] for t in m_tokens if t])
        if len(s1_tokens) == 1 and s1_tokens[0] == m_acronym and len(m_acronym) >= 2:
            acronym_matches += 1
        elif len(m_tokens) == 1 and m_tokens[0] == s1_acronym and len(s1_acronym) >= 2:
            acronym_matches += 1

    # Address checks
    clean_s1_a = strip_punct_and_case(s1_a)
    clean_m_a = strip_punct_and_case(m_a)
    if clean_s1_a == clean_m_a:
        addr_exact_clean += 1
    
    s1_a_tokens = clean_s1_a.split()
    m_a_tokens = clean_m_a.split()
    if s1_a_tokens and m_a_tokens:
        s1_a_set = set(s1_a_tokens)
        m_a_set = set(m_a_tokens)
        a_jaccard = len(s1_a_set & m_a_set) / len(s1_a_set | m_a_set)
        addr_token_jaccards.append(a_jaccard)
        
        if s1_a_set == m_a_set and clean_s1_a != clean_m_a:
            addr_token_reordered += 1
        elif s1_a_set.issubset(m_a_set) or m_a_set.issubset(s1_a_set):
            if s1_a_set != m_a_set:
                addr_missing_components += 1

    # PIN/Zip code checks
    s1_pins = set(pin_re.findall(s1_a))
    m_pins = set(pin_re.findall(m_a))
    if s1_pins: s1_has_pin += 1
    if m_pins: match_has_pin += 1
    if s1_pins and m_pins:
        both_have_pin += 1
        if s1_pins & m_pins:
            pin_exact_match += 1
        else:
            pin_conflict += 1

N = len(sample_pairs)
print("\n--- Name Matching Statistics (out of 100,000 sample pairs) ---")
print(f"Exact string match (raw): {exact_raw_count:,} ({exact_raw_count/N*100:.2f}%)")
print(f"Case-insensitive match: {exact_case_count:,} ({exact_case_count/N*100:.2f}%)")
print(f"Punctuation & whitespace stripped match: {exact_punct_count:,} ({exact_punct_count/N*100:.2f}%)")
print(f"Legal suffix + punctuation stripped match: {exact_legal_count:,} ({exact_legal_count/N*100:.2f}%)")
print(f"Average Name Token Jaccard: {sum(token_jaccard_scores)/len(token_jaccard_scores):.4f}")
print(f"Detected pure acronym matches: {acronym_matches:,} ({acronym_matches/N*100:.2f}%)")

print("\nExamples differing by legal suffix / punctuation:")
for c, s, m in legal_diff_examples[:5]:
    print(f"  [{c}] S1: '{s}' <--> MATCH: '{m}'")

print("\nExamples of more complex name differences:")
for c, s, m in diff_examples[:5]:
    print(f"  [{c}] S1: '{s}' <--> MATCH: '{m}'")

print("\n--- Address Matching Statistics (out of 100,000 sample pairs) ---")
print(f"Address exact clean match: {addr_exact_clean:,} ({addr_exact_clean/N*100:.2f}%)")
print(f"Address pure token reordering (identical token sets, different order): {addr_token_reordered:,} ({addr_token_reordered/N*100:.2f}%)")
print(f"Address strict subset / missing components (one contains all tokens of the other): {addr_missing_components:,} ({addr_missing_components/N*100:.2f}%)")
print(f"Average Address Token Jaccard: {sum(addr_token_jaccards)/len(addr_token_jaccards):.4f}")

print("\n--- Postal / PIN Code Statistics ---")
print(f"S1 has 5-6 digit postal/PIN code: {s1_has_pin:,} ({s1_has_pin/N*100:.2f}%)")
print(f"Match record has postal/PIN code: {match_has_pin:,} ({match_has_pin/N*100:.2f}%)")
print(f"Both have postal/PIN code: {both_have_pin:,} ({both_have_pin/N*100:.2f}%)")
if both_have_pin > 0:
    print(f"When both have PIN: Exact PIN overlap: {pin_exact_match:,} ({pin_exact_match/both_have_pin*100:.2f}%)")
    print(f"When both have PIN: PIN conflict: {pin_conflict:,} ({pin_conflict/both_have_pin*100:.2f}%)")

print("\n--- S2 vs S3 Source Differences ---")
print(f"S2 pairs: {s2_count:,}, Exact raw name match: {s2_name_exact:,} ({s2_name_exact/s2_count*100:.2f}%)")
print(f"S3 pairs: {s3_count:,}, Exact raw name match: {s3_name_exact:,} ({s3_name_exact/s3_count*100:.2f}%)")
