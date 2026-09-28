"""
Business Entity Resolution Preprocessing & Normalization Module
Amazon ML Challenge 2026

Provides multi-representation normalization for business names and addresses:
- Preserves raw values alongside normalized representations.
- Unicode normalization (NFKC) and mojibake repair without losing native scripts.
- Configurable legal suffix dictionaries (US/UK, French, Indic).
- Web domain and URL normalization (.com, .org, www., etc.).
- Explicit missing-value handling (flags, empty representations, never dummy strings).
- Address abbreviation standardization (st -> street, rd -> road, etc.).
- Postal/PIN code extraction (regex, open-set 5-6 digit codes).
- Structural and numeric token extraction.
- Character n-gram extraction for fuzzy and typo-tolerant matching.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any

# ==============================================================================
# 1. MOJIBAKE & UNICODE REPAIR MAPPINGS
# ==============================================================================

# Common mojibake strings produced when UTF-8 bytes are decoded as CP1252 / Latin-1
MOJIBAKE_MAP: Dict[str, str] = {
    # French & Western European UTF-8 bytes decoded as Latin-1 / Windows-1252
    "Ã©": "é",
    "Ã¨": "è",
    "Ãª": "ê",
    "Ã«": "ë",
    "Ã ": "à",
    "Ã¢": "â",
    "Ã®": "î",
    "Ã¯": "ï",
    "Ã´": "ô",
    "Ã¹": "ù",
    "Ã»": "û",
    "Ã§": "ç",
    "Ã‰": "É",
    "Ãˆ": "È",
    "Â€“": "-",       # en-dash (U+2013)
    "â€“": "-",       # en-dash
    "Â€”": " - ",     # em-dash (U+2014)
    "â€”": " - ",     # em-dash
    "Â€˜": "'",       # left single quote (U+2018)
    "â€˜": "'",       # left single quote
    "Â€™": "'",       # right single quote (U+2019)
    "â€™": "'",       # right single quote
    "Â“": '"',        # left double quote (U+201C)
    "â€œ": '"',       # left double quote
    "Â”": '"',        # right double quote (U+201D)
    "â€\x9d": '"',     # right double quote variant
    "â€¦": "...",     # ellipsis (U+2026)
    "Â€": "",         # stray euro/control artifact
    "â€": "",         # stray control artifact
    "Â": "",          # lone non-breaking space / artifact
    "\u00a0": " ",    # non-breaking space
    "\ufeff": "",     # byte order mark (BOM)
    "\u200b": "",     # zero-width space
    "": "",          # Unicode replacement character
}

# ==============================================================================
# 2. LEGAL SUFFIX DICTIONARY (Explicit, Configurable, Ordered by Length)
# ==============================================================================

# English, US, UK, and Commonwealth legal entity designators
LEGAL_SUFFIXES_EN = [
    # Unambiguous legal entity designators ONLY (never generic business descriptors)
    "private limited company",
    "limited liability company",
    "limited liability partnership",
    "public limited company",
    "private limited",
    "public limited",
    "pvt. ltd.",
    "pvt ltd",
    "pvt. limited",
    "pvt limited",
    "private ltd",
    "pub. ltd.",
    "pub ltd",
    "corporation",
    "incorporated",
    "limited",
    "company",
    "corp.",
    "corp",
    "inc.",
    "inc",
    "ltd.",
    "ltd",
    "llc.",
    "llc",
    "llp.",
    "llp",
    "plc.",
    "plc",
    "co.",
    "co",
    "pvt",
]

# French legal entity designators (observed in test dataset)
LEGAL_SUFFIXES_FR = [
    "societe a responsabilite limitee",
    "societe par actions simplifiee unipersonnelle",
    "societe par actions simplifiee",
    "entreprise unipersonnelle a responsabilite limitee",
    "societe civile immobiliere",
    "societe civile",
    "societe anonyme",
    "et fils",
    "& fils",
    "s.a.r.l.u.",
    "sarlu",
    "s.a.s.u.",
    "sasu",
    "s.a.r.l.",
    "sarl",
    "s.a.s.",
    "sas",
    "e.u.r.l.",
    "eurl",
    "s.c.i.",
    "sci",
    "s.n.c.",
    "snc",
    "s.a.",
    "sa",
    "e.i.",
    "ei",
    "fils",
    "groupe",
    "cie",
    "ets",
]

# Indic script legal suffix transliterations (observed in S2/S3)
LEGAL_SUFFIXES_INDIC = [
    "प्राइवेट लिमिटेड",
    "प्राइवेट लि.",
    "प्रा. लि.",
    "प्रा लि",
    "लिमिटेड",
    "एलएलपी",
    "प्राइवेट",
    "பிரைவேட் லிமிடெட்",
    "லிமிடெட்",
    "પ્રાઇવેટ લિમિટેડ",
    "પ્રા. લિ.",
    "પ્રા લિ",
    "લિમિટેડ",
]

# Combined list sorted strictly by descending string length
# to guarantee multi-word phrases match before single-word substrings
ALL_LEGAL_SUFFIXES_ORDERED = sorted(
    list(set(LEGAL_SUFFIXES_EN + LEGAL_SUFFIXES_FR + LEGAL_SUFFIXES_INDIC)),
    key=lambda s: len(s),
    reverse=True,
)

# Regex matching legal suffix anchored at the end of string or enclosed in brackets
_SUFFIX_PATTERNS = []
for s in ALL_LEGAL_SUFFIXES_ORDERED:
    # Escape special regex characters in the suffix string
    esc = re.escape(s)
    # Match at the end of string preceded by word boundary / space / comma / dash
    _SUFFIX_PATTERNS.append(rf"(?:[,\s\-\(\[\#]+|\b){esc}(?:[,\s\.\)\]]*)$")
    # Also match suffix inside parentheses or brackets anywhere at the end
    _SUFFIX_PATTERNS.append(r"[\(\[\{]" + esc + r"[\)\]\}]$")

LEGAL_SUFFIX_REGEX = re.compile("|".join(_SUFFIX_PATTERNS), flags=re.IGNORECASE)

# ==============================================================================
# 3. WEB & DOMAIN EXTENSIONS
# ==============================================================================

WEB_PREFIX_REGEX = re.compile(r"^(?:https?://|ftp://)?(?:www\.)?", flags=re.IGNORECASE)
WEB_SUFFIX_REGEX = re.compile(
    r"\.(?:com|org|net|in|co\.in|fr|us|co|biz|info|io|ai|gov|edu|online|store|tech|org\.in|gov\.in)(?:/[^\s]*)?$",
    flags=re.IGNORECASE,
)

# ==============================================================================
# 4. ADDRESS ABBREVIATIONS DICTIONARY
# ==============================================================================

ADDR_ABBREVIATIONS: Dict[str, str] = {
    # Street types
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "bd": "boulevard",
    "pkwy": "parkway",
    "hwy": "highway",
    "cir": "circle",
    "sq": "square",
    "ter": "terrace",
    "trl": "trail",
    "way": "way",
    "aly": "alley",
    # French thoroughfares
    "av": "avenue",
    "imp": "impasse",
    "che": "chemin",
    "chem": "chemin",
    "all": "allee",
    "r": "rue",
    # Unit / Sub-premise types
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "rm": "room",
    "unit": "unit",
    "no": "number",
    # Positional / Landmark terms
    "nr": "near",
    "opp": "opposite",
    "sec": "sector",
    "dist": "district",
    "pl": "plot",
    "plt": "plot",
    "bl": "block",
    "blk": "block",
    "chq": "chowk",
    "chauraha": "chowk",
    # Compass directions (when standalone)
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# Compiled regex for address abbreviation substitution (strictly whole-word boundary)
_ADDR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ADDR_ABBREVIATIONS.keys()) + r")\b",
    flags=re.IGNORECASE,
)

# Regex for postal/PIN code extraction: 5 to 6 consecutive digits
POSTAL_CODE_REGEX = re.compile(r"\b\d{5,6}\b")

# Regex for numeric tokens: 1 to 6 digits, optionally followed by a single unit letter (e.g. 1027, 308C, 12B)
NUMERIC_TOKEN_REGEX = re.compile(r"\b\d+[a-zA-Z]?\b")


# ==============================================================================
# 5. CORE NORMALIZATION FUNCTIONS
# ==============================================================================

def clean_unicode_and_mojibake(text: Optional[str]) -> str:
    """Repair mojibake artifacts, normalize Unicode to NFKC, and trim whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Direct mojibake substring replacement
    for bad, good in MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)

    # 2. Unicode NFKC normalization (composes accents, standardizes fullwidth forms)
    text = unicodedata.normalize("NFKC", text)

    # 3. Collapse intra-acronym dots between single letters (e.g. L.L.P. -> LLP, S.A.R.L. -> SARL)
    text = re.sub(r"(?<=\b[a-zA-Z])\.(?=[a-zA-Z]\b|\s|$)", "", text)

    # 4. Collapse multiple whitespaces
    text = " ".join(text.split())
    return text.strip()


def strip_latin_accents(text: str) -> str:
    """
    Remove accents/diacritics specifically from Latin characters (e.g. 'é' -> 'e', 'à' -> 'a')
    using NFKD decomposition without damaging Indic scripts or non-Latin alphabets.
    """
    if not text:
        return ""
    result = []
    for ch in text:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
            decomposed = unicodedata.normalize("NFKD", ch)
            cleaned = "".join(c for c in decomposed if not unicodedata.combining(c))
            result.append(cleaned)
        else:
            result.append(ch)
    return "".join(result)


def detect_script(text: Optional[str]) -> str:
    """
    Detect the primary writing script of a text string without external libraries.
    Returns: 'latin', 'devanagari', 'tamil', 'gujarati', 'bengali', 'telugu', or 'other'.
    """
    if not text:
        return "empty"

    devanagari_count = 0
    tamil_count = 0
    gujarati_count = 0
    bengali_count = 0
    latin_count = 0

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            devanagari_count += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            tamil_count += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            gujarati_count += 1
        elif 0x0980 <= cp <= 0x09FF:
            bengali_count += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            latin_count += 1

    counts = [
        ("devanagari", devanagari_count),
        ("tamil", tamil_count),
        ("gujarati", gujarati_count),
        ("bengali", bengali_count),
        ("latin", latin_count),
    ]
    best_script, max_c = max(counts, key=lambda x: x[1])
    return best_script if max_c > 0 else "latin"


def strip_web_domains(text: str) -> str:
    """
    Strip web prefixes (www., http://) and domain extensions (.com, .org, etc.)
    if the name is structured as a URL / website handle.
    Preserves the root brand name.
    """
    if not text:
        return ""
    t = text.strip()
    t = WEB_PREFIX_REGEX.sub("", t)
    t = WEB_SUFFIX_REGEX.sub("", t)
    return t.strip()


def strip_legal_suffixes(text: str) -> str:
    """
    Remove recognized legal suffixes from the end of a business name.
    Iteratively strips up to 2 suffix layers (e.g. 'Co LLC' or 'Services Ltd').
    Never removes words from the beginning or middle of the name.
    """
    if not text:
        return ""
    t = text.strip()
    for _ in range(2):
        prev = t
        t = LEGAL_SUFFIX_REGEX.sub("", t).strip()
        # Clean trailing commas, dashes, dots left behind after suffix removal
        t = re.sub(r"[,\-\.\#]+$", "", t).strip()
        if t == prev or len(t) <= 2:
            break
    return t if t else text.strip()


def to_alphanumeric(text: str) -> str:
    """
    Extract strictly lowercase alphanumeric characters (and native script letters),
    removing all punctuation, symbols, and spaces.
    e.g., 'Gonzalez, Altman & Thomas LLC' -> 'gonzalezaltmanthomasllc'
    """
    if not text:
        return ""
    # Retain Unicode letters and numbers, strip punctuation and whitespace
    clean = re.sub(r"[^\w\d]+", "", text.lower(), flags=re.UNICODE)
    return clean.replace("_", "")


def tokenize(text: str) -> List[str]:
    """
    Split text into normalized, lowercase word tokens, stripping punctuation.
    Supports Unicode alphanumeric tokens.
    """
    if not text:
        return []
    # Replace non-word characters (except Indic/Unicode word chars) with spaces
    tokens = re.split(r"[^\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t and t != "_"]


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Extract character n-grams from an alphanumeric or cleaned string with boundary markers.
    e.g., 'berry' (n=3) -> {'^be', 'ber', 'err', 'rry', 'ry$'}
    """
    if not text:
        return set()
    s = f"^{text}$"
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def standardize_address_abbreviations(text: str) -> str:
    """
    Standardize common street and address abbreviations using whole-word boundary replacement.
    e.g., '308C N Main St' -> '308C north main street'
    """
    if not text:
        return ""

    def _replace_match(m: re.Match) -> str:
        word = m.group(1).lower()
        return ADDR_ABBREVIATIONS.get(word, word)

    return _ADDR_PATTERN.sub(_replace_match, text.lower())


def extract_numeric_tokens(text: str) -> Tuple[List[str], Set[str]]:
    """
    Extract all numeric/alphanumeric house, plot, and unit numbers from an address.
    Normalizes leading zeros (e.g. '0012' -> '12', '00145b' -> '145b') so that
    different zero-padding conventions match seamlessly.
    Returns: (ordered_list, deduplicated_set)
    """
    if not text:
        return [], set()
    raw_nums = [m.lower() for m in NUMERIC_TOKEN_REGEX.findall(text)]
    norm_nums = []
    for num in raw_nums:
        if num.isdigit():
            norm_nums.append(str(int(num)))
        else:
            m = re.match(r"^0*(\d+.*)$", num)
            norm_nums.append(m.group(1) if m else num)
    # Return both normalized and raw for maximum recall
    all_nums = list(dict.fromkeys(norm_nums + raw_nums))
    return all_nums, set(all_nums)


def extract_postal_codes(text: str) -> Set[str]:
    """
    Extract all 5-digit and 6-digit postal / PIN codes from an address string.
    Works for US (5 digits), France (5 digits), and India (6 digits).
    """
    if not text:
        return set()
    return set(POSTAL_CODE_REGEX.findall(text))


# ==============================================================================
# 6. STRUCTURED DATA REPRESENTATION CLASSES
# ==============================================================================

@dataclass
class NormalizedName:
    raw: str
    is_missing: bool
    is_non_ascii: bool
    detected_script: str
    unicode_clean: str
    lowercased: str
    domain_stripped: str
    legal_stripped: str
    alphanumeric_clean: str
    alphanumeric_no_legal: str
    alphanumeric_ascii_folded: str
    tokens: List[str]
    tokens_set: Set[str]
    tokens_no_legal: List[str]
    tokens_no_legal_set: Set[str]
    char_3grams: Set[str]


@dataclass
class NormalizedAddress:
    raw: str
    is_missing: bool
    unicode_clean: str
    lowercased: str
    standardized: str
    standardized_ascii_folded: str
    tokens: List[str]
    tokens_set: Set[str]
    numeric_tokens: List[str]
    numeric_tokens_set: Set[str]
    postal_codes: Set[str]
    has_postal_code: bool


@dataclass
class NormalizedRecord:
    entity_id: str
    source_prefix: str  # 'S1', 'S2', 'S3'
    country: str        # Raw string label, open-set
    name: NormalizedName
    address: NormalizedAddress


# ==============================================================================
# 7. HIGH-LEVEL PREPROCESSOR CLASS
# ==============================================================================

class EntityPreprocessor:
    """
    End-to-end Preprocessor that transforms raw input records into rich,
    multi-representation data structures for candidate blocking and matching.
    """

    def __init__(self):
        pass

    def preprocess_name(self, raw_name: Optional[str]) -> NormalizedName:
        """Create all normalized representations for a business name."""
        is_missing = raw_name is None or not isinstance(raw_name, str) or raw_name.strip() == ""
        if is_missing:
            return NormalizedName(
                raw=raw_name or "",
                is_missing=True,
                is_non_ascii=False,
                detected_script="empty",
                unicode_clean="",
                lowercased="",
                domain_stripped="",
                legal_stripped="",
                alphanumeric_clean="",
                alphanumeric_no_legal="",
                alphanumeric_ascii_folded="",
                tokens=[],
                tokens_set=set(),
                tokens_no_legal=[],
                tokens_no_legal_set=set(),
                char_3grams=set(),
            )

        # 1. Unicode & Mojibake repair
        clean_name = clean_unicode_and_mojibake(raw_name)

        # 2. Script detection & Non-ASCII check
        script = detect_script(clean_name)
        is_non_ascii = any(ord(c) > 127 for c in clean_name)

        # 3. Lowercasing
        lowered = clean_name.lower()

        # 4. Domain / Web suffix removal
        domain_stripped = strip_web_domains(lowered)

        # 5. Legal suffix removal
        legal_stripped = strip_legal_suffixes(domain_stripped)

        # 6. Alphanumeric representations
        alpha_clean = to_alphanumeric(domain_stripped)
        alpha_no_legal = to_alphanumeric(legal_stripped)
        alpha_ascii_folded = to_alphanumeric(strip_latin_accents(legal_stripped if legal_stripped else domain_stripped))

        # 7. Tokenization
        tokens = tokenize(domain_stripped)
        tokens_set = set(tokens)

        tokens_no_legal = tokenize(legal_stripped)
        tokens_no_legal_set = set(tokens_no_legal)

        # 8. Character 3-grams
        char_3grams = extract_char_ngrams(alpha_no_legal if alpha_no_legal else alpha_clean, n=3)

        return NormalizedName(
            raw=raw_name,
            is_missing=False,
            is_non_ascii=is_non_ascii,
            detected_script=script,
            unicode_clean=clean_name,
            lowercased=lowered,
            domain_stripped=domain_stripped,
            legal_stripped=legal_stripped,
            alphanumeric_clean=alpha_clean,
            alphanumeric_no_legal=alpha_no_legal,
            alphanumeric_ascii_folded=alpha_ascii_folded,
            tokens=tokens,
            tokens_set=tokens_set,
            tokens_no_legal=tokens_no_legal,
            tokens_no_legal_set=tokens_no_legal_set,
            char_3grams=char_3grams,
        )

    def preprocess_address(self, raw_address: Optional[str]) -> NormalizedAddress:
        """Create all normalized representations for a business address."""
        is_missing = (
            raw_address is None
            or not isinstance(raw_address, str)
            or raw_address.strip() == ""
        )
        if is_missing:
            return NormalizedAddress(
                raw=raw_address or "",
                is_missing=True,
                unicode_clean="",
                lowercased="",
                standardized="",
                standardized_ascii_folded="",
                tokens=[],
                tokens_set=set(),
                numeric_tokens=[],
                numeric_tokens_set=set(),
                postal_codes=set(),
                has_postal_code=False,
            )

        # 1. Unicode & Mojibake repair
        clean_addr = clean_unicode_and_mojibake(raw_address)

        # 2. Lowercase
        lowered = clean_addr.lower()

        # 3. Abbreviation standardization
        standardized = standardize_address_abbreviations(lowered)
        std_ascii_folded = strip_latin_accents(standardized)

        # 4. Tokenization
        tokens = tokenize(standardized)
        tokens_set = set(tokens)

        # 5. Numeric tokens extraction
        num_list, num_set = extract_numeric_tokens(standardized)

        # 6. Postal / PIN code extraction
        postals = extract_postal_codes(clean_addr)

        return NormalizedAddress(
            raw=raw_address,
            is_missing=False,
            unicode_clean=clean_addr,
            lowercased=lowered,
            standardized=standardized,
            standardized_ascii_folded=std_ascii_folded,
            tokens=tokens,
            tokens_set=tokens_set,
            numeric_tokens=num_list,
            numeric_tokens_set=num_set,
            postal_codes=postals,
            has_postal_code=len(postals) > 0,
        )

    def preprocess_record(
        self,
        entity_id: str,
        business_name: Optional[str],
        business_address: Optional[str],
        country: str,
    ) -> NormalizedRecord:
        """
        Process a complete business record into a NormalizedRecord.
        Prefix is determined directly from entity_id (S1, S2, S3).
        Country is retained as an open-set string.
        """
        prefix = entity_id[:2] if entity_id and len(entity_id) >= 2 else "UNKNOWN"
        norm_name = self.preprocess_name(business_name)
        norm_addr = self.preprocess_address(business_address)

        return NormalizedRecord(
            entity_id=entity_id,
            source_prefix=prefix,
            country=country.strip() if country else "",
            name=norm_name,
            address=norm_addr,
        )
