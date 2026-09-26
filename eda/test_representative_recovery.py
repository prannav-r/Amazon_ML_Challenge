"""
Test 25 Representative Missed Pairs against Candidate Generation Channels
Amazon ML Challenge: Business Entity Resolution
"""

import duckdb
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
con = duckdb.connect()

base_train = "student_resource/dataset/train"

# Representative pairs from the audit
pairs = [
    # 1. Native-Script with Address Number/Locality Overlap
    ("S1-685876877", "S3-422231099", "Prime Foods", "प्राइम फूड्स", "4, Gulmohar, Gultekdi, Pune, Maharashtra", "4, Gulmohar, Gultekdi, Pune, MH", "Native-Script"),
    ("S1-851000755", "S2-572814418", "Indian Industries Pvt Ltd", "इंडियन इंडस्ट्रीज प्रा. लि.", "201 Veer Sarvarkar Block, East Delhi, 19-B S/F Pvt Office No.", "19-B S/F PVT OFFICE NO., SHAHDARA, Delhi", "Native-Script"),
    ("S1-898986640", "S2-398605759", "Anand Foundation Private Limited", "ಆನಂದ್ ಫೌಂಡೇಶನ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "No 688/B, F 36/B, B B Garden 2Nd Main Road, Mysore", "NO 688/B, F 36/B, B B GARDEN 2ND MAIN ROAD, MYSORE", "Native-Script"),
    ("S1-358249590", "S2-570189202", "Apex Products Private Limited", "एपेक्स प्रोडक्ट्स प्राइवेट लिमिटेड", "7 Snehlata Ganj, Indore, Madhya Pradesh", "7 SNEHLATA GANJ, INDORE, Madhya Pradesh", "Native-Script"),
    ("S1-621735796", "S3-567732293", "One Estate LLP", "वन एस्टेट एलएलपी", "501 B Wing, Express Zone, Malad East, Mumbai", "501 B Wing, Express Zone, Malad East, Mumbai", "Native-Script"),
    ("S1-742174049", "S3-226789326", "Shakti Producer Private Limited", "ಶಕ್ತಿ ಪ್ರೊಡ್ಯೂಸರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "Door No 2-108, Heggadadevankote, Mysore, Karnataka", "Door No 2-108, Heggadadevankote, Mysore, Karnataka", "Native-Script"),
    
    # 2. Prefix / Noise discrepancies
    ("S1-288971394", "S2-426756976", "Great Guild", "The Great Guild", "91 Malcolm Street, Unit Apartment B4, Ossining, NY", "#91 MALCOLM ST, OSSINING, NY", "Prefix Noise ('The ')"),
    ("S1-83625448", "S2-597434714", "Jai Impex Private Limited", "Mr Jai Impex Privte Limited", "874/4 Lal Dora, Delhi", "74/4 Lal Dora, Delhi", "Prefix Noise ('Mr ')"),
    ("S1-173812578", "S3-874768128", "Black Tech Pvt Ltd", "Smt Black Tech Pvt Ltd", "12 Industrial Area, Phase 2, Chandigarh", "12 Industrial Area, Phase 2, Chandigarh", "Prefix Noise ('Smt ')"),
    ("S1-105022531", "S2-611098764", "74/60 Pizza", "THE 74/60 PIZZA", "108 Main St, Dallas, TX", "108 MAIN ST, DALLAS, TX", "Prefix Noise ('THE ')"),
    ("S1-89576694", "S2-484555273", "Delgado Academy", "The Delgado Academy", "410 Pine St, Tampa, FL", "410 PINE ST, TAMPA, FL", "Prefix Noise ('The ')"),
    ("S1-173467573", "S3-740712235", "Vision North Consultancy Private Limited", "Pyraectoveo DBA: Vision North Consultancy", "B-5/148, Safdarjung Enclave, New Delhi", "B-5/148, Safdarjung Enclave, New Delhi", "DBA Prefix"),

    # 3. Minor Typos / Levenshtein <= 2
    ("S1-699565963", "S3-104081829", "Willow Co", "Wsillw Co", "1046 24 Street, Brooklyn, NY", "1046 24 St, Brooklyn, New York", "Typo (Wsillw)"),
    ("S1-900437547", "S2-522879516", "Pediatric Dental Group of Golden Valley LLC", "Pediatric Dental Group of Golden Valhey LLC", "2661 Apache Road, Golden Valley, AZ", "APACHE RD, GOLDEN VALLEY, AZ", "Typo (Valhey)"),
    ("S1-87856323", "S2-275155485", "Keystone Live LLC", "KEYSTONE LÍVE LLC", "6158 3rd Court, Renton, WA", "THIRD CT, RENTON, WA", "Accent / Typo"),
    ("S1-641990065", "S2-683514251", "Southern First Vision Center LLC", "The S0uthern First Vision Center LLC", "Indianapolis, IN, 2672 1050", "2672 1050, INDIANAPOLIS, IN", "Digit Sub ('0' vs 'o')"),
    ("S1-861859613", "S2-333575115", "Orion L.L.C.", "0rion L.L.C.", "452 Lincoln Way, Ames, IA", "452 LINCOLN WAY, AMES, IA", "Digit Sub ('0' vs 'O')"),

    # 4. Aliases / Web Domain Names
    ("S1-195709451", "S2-555826021", "Urgent Care Center LLC", "urgentcarecenter.com", "431 Russell Street, Huntsville, AL", "00431 RUSSELL SAINT, HUNTSVILLE, AL", "Domain Alias"),
    ("S1-949679687", "S2-327144958", "Sharp Hardware LLP", "Smt sharphardware.com", "601 Mira Chs Ltd, Sudama Bldg, Mumbai", "C-601 MIRA CHS LTD, SUDAMA BLDG, MUMBAI", "Domain Alias + Prefix"),
    ("S1-839511041", "S3-450395744", "Beatty's Management", "beattysmanagement.com", "712 Broadway, New York, NY", "712 Broadway, New York, NY", "Domain Alias"),
    ("S1-257022224", "S2-220797717", "Averyl Hickman Materials Inc.", "averylhickmanmaterials.com #11557", "901 Oak St, Denver, CO", "901 OAK ST, DENVER, CO", "Domain Alias"),
    ("S1-143817370", "S2-223814487", "Mutual Telecom Brands PC", "MUTUALTELECOMBRANDS.COM", "5627 Sandpiper Lane, Dayton, OH", "627 SANDPIPER LANE, DAYTON, OH", "Domain Alias"),

    # 5. Word Order / Token Shuffling
    ("S1-56607895", "S3-115527350", "Vega & Co", "Center Co Vega", "110 State St, Boston, MA", "110 State St, Boston, MA", "Word Reordering"),
    ("S1-438680175", "S3-980192481", "Dow Advanced Motor Inc", "Inc Dow Advanced Mótor", "320 Market St, Philadelphia, PA", "320 Market St, Philadelphia, PA", "Word Reordering"),
    ("S1-380764336", "S2-538016144", "HKB Information Pvt Ltd", "HKB Information Ltd Pvt", "55 MG Road, Bangalore, Karnataka", "55 MG ROAD, BANGALORE, Karnataka", "Suffix Swap"),

    # 6. Completely Disjoint Trade Name
    ("S1-309715126", "S3-803410155", "Coleman & Johnson Goldman", "Quokor", "3854 Klein Avenue, Stow, OH", "##3854 Klein Avenue, Stow, Ohio", "Disjoint Alias (Address Number Shared)")
]

UNAMBIGUOUS_LEGAL_REGEX = (
    r"(?:[,\s\-\(\[\#]+|\b)("
    r"private\s+limited\s+company|limited\s+liability\s+company|limited\s+liability\s+partnership|"
    r"public\s+limited\s+company|private\s+limited|public\s+limited|"
    r"pvt\s+ltd|pvt\s+limited|private\s+ltd|pub\s+ltd|pub\s+limited|"
    r"corporation|incorporated|limited|company|corp|inc|llc|llp|plc|ltd|co|pvt|"
    r"societe\s+a\s+responsabilite\s+limitee|societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|"
    r"societe\s+par\s+actions\s+simplifiee|entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|"
    r"societe\s+anonyme|societe\s+civile|sarlu|sasu|sarl|sas|eurl|sci|snc|sa|ei|et\s+fils|fils|"
    r"प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|लिमिटेड|एलएलपी"
    r")(?:[,\s\.\)\]]*)$"
)
PREFIX_NOISE_REGEX = r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|mr\s+|co\s+|>>\s+|#\s*)"

def norm_core(name):
    if not name: return ""
    clean = name.lower().replace('.', '')
    clean = re.sub(UNAMBIGUOUS_LEGAL_REGEX, '', clean)
    clean = re.sub(r'[^a-z0-9]', '', clean)
    return clean

def norm_prefix_stripped(name):
    if not name: return ""
    clean = name.lower().replace('.', '')
    clean = re.sub(PREFIX_NOISE_REGEX, '', clean)
    clean = re.sub(r'\b(?:dba[:\s]+)', ' ', clean)
    clean = re.sub(UNAMBIGUOUS_LEGAL_REGEX, '', clean)
    clean = re.sub(r'[^a-z0-9]', '', clean)
    return clean

def extract_num(addr):
    if not addr: return ""
    m = re.findall(r'\b\d+\b', addr)
    if not m:
        m = re.findall(r'\d+', addr)
    return str(int(m[0])) if m else ""

def extract_addr_tokens(addr):
    if not addr: return set()
    stops = {'street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
             'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
             'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near'}
    toks = [t for t in re.sub(r'[^a-z0-9 ]', ' ', addr.lower()).split() if len(t) >= 4 and t not in stops]
    return set(toks)

def extract_name_tokens(name):
    if not name: return set()
    stops = {'private', 'limited', 'pvt', 'ltd', 'inc', 'corp', 'company', 'llc', 'the'}
    toks = [t for t in re.sub(r'[^a-z0-9 ]', ' ', name.lower()).split() if len(t) >= 3 and t not in stops]
    return set(toks)

print("=" * 100)
print(f"{'#':2s} | {'Pair':26s} | {'Category':24s} | {'Channel Hits & Recovery Mechanism'}")
print("=" * 100)

recovered_count = 0
for idx, (s1_id, m_id, s1_n, m_n, s1_a, m_a, cat) in enumerate(pairs, 1):
    core_s1 = norm_core(s1_n)
    core_m = norm_core(m_n)
    
    pstrip_s1 = norm_prefix_stripped(s1_n)
    pstrip_m = norm_prefix_stripped(m_n)
    
    num_s1 = extract_num(s1_a)
    num_m = extract_num(m_a)
    
    atoks_s1 = extract_addr_tokens(s1_a)
    atoks_m = extract_addr_tokens(m_a)
    shared_atoks = atoks_s1 & atoks_m
    
    ntoks_s1 = extract_name_tokens(s1_n)
    ntoks_m = extract_name_tokens(m_n)
    shared_ntoks = ntoks_s1 & ntoks_m
    
    p3_s1 = core_s1[:3] if len(core_s1) >= 3 else ""
    p3_m = core_m[:3] if len(core_m) >= 3 else ""
    
    del_s1 = core_s1[1:7] if len(core_s1) >= 6 else ""
    del_m = core_m[1:7] if len(core_m) >= 6 else ""
    
    hits = []
    # Test Channel A
    if core_s1 and core_s1 == core_m:
        hits.append("Chan A (Exact Core)")
    # Test Channel A2
    if pstrip_s1 and pstrip_s1 == pstrip_m:
        hits.append("Chan A2 (Prefix-Stripped)")
    # Test Channel B
    if shared_ntoks:
        hits.append(f"Chan B (Name Toks: {','.join(list(shared_ntoks)[:2])})")
    # Test Channel E
    if num_s1 and num_s1 == num_m and p3_s1 and p3_s1 == p3_m:
        hits.append("Chan E (Num+p3)")
    # Test Channel E2
    if num_s1 and num_s1 == num_m and shared_atoks:
        hits.append(f"Chan E2 (Num '{num_s1}' + Loc '{list(shared_atoks)[0]}')")
    # Test Channel G (1-edit initial)
    if del_s1 and del_s1 == del_m:
        hits.append(f"Chan G (Fuzzy initial: '{del_s1}')")
        
    status = "RECOVERED via " + " & ".join(hits) if hits else "NOT RECOVERED (Inherent Discrepancy)"
    if hits:
        recovered_count += 1
        
    print(f"{idx:2d} | {s1_id} <-> {m_id:12s} | {cat:24s} | {status}")
    print(f"   S1: '{s1_n}' | Addr: '{s1_a}'")
    print(f"   MT: '{m_n}' | Addr: '{m_a}'")
    print()

print("=" * 100)
print(f"Summary: {recovered_count} / {len(pairs)} ({recovered_count/len(pairs)*100:.1f}%) recovered by new targeted channels!")
print("=" * 100)
