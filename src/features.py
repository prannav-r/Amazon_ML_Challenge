"""
Pairwise Feature Engineering Module (Phase 4)
Amazon ML Challenge 2026: Business Entity Resolution

Transforms blocked (Source 1, Candidate S2/S3) pairs into a rich, interpretable,
leakage-free feature matrix for entity matching.

Feature Groups (63 total features):
- Group A: Country Features (2 features, open-set)
- Group B: Name Similarity Features (22 features: exact, edit, token, legal, script)
- Group C: Address Similarity Features (17 features: exact, token, numeric, postal, missing)
- Group D: Cross-Field Interaction Features (7 features: multi-field agreements)
- Group E: Source-Specific Features (4 features: S2/S3 noise patterns)
- Group F: Blocking Provenance Features (11 features: channel flags, priority score, rank order)

Guarantees:
- Strictly open-set country handling (France, US, India, etc.).
- No external lookups or web identity lookups.
- Deterministic missingness handling (no dummy string injection).
- Zero label leakage (ground-truth labels are only attached as targets, never inputs).
- S1-level grouped train/validation splitting.
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple, Any
import rapidfuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src.feature_schema import FEATURE_NAMES, FEATURE_SPECS, get_feature_names as schema_get_feature_names

# Precompiled regex patterns for feature extraction
UNAMBIGUOUS_LEGAL_REGEX = re.compile(
    r"(?:[,\s\-\(\[\#]+|\b)("
    r"private\s+limited\s+company|limited\s+liability\s+company|limited\s+liability\s+partnership|"
    r"public\s+limited\s+company|private\s+limited|public\s+limited|"
    r"pvt\s+ltd|pvt\s+limited|private\s+ltd|pub\s+ltd|pub\s+limited|"
    r"corporation|incorporated|limited|company|corp|inc|llc|llp|plc|ltd|co|pvt|"
    r"societe\s+a\s+responsabilite\s+limitee|societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|"
    r"societe\s+par\s+actions\s+simplifiee|entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|"
    r"societe\s+anonyme|societe\s+civile|sarlu|sasu|sarl|sas|eurl|sci|snc|sa|ei|et\s+fils|fils|"
    r"प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|लिमिटेड|एलएलपी"
    r")(?:[,\s\.\)\]]*)$",
    re.IGNORECASE
)

PREFIX_NOISE_REGEX = re.compile(r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|mr\s+|co\s+|>>\s+|#\s*)", re.IGNORECASE)
DOMAIN_REGEX = re.compile(r"\.(?:com|org|net|in|co|io|biz|info|gov|edu|ai|us|fr)\b|^www\.", re.IGNORECASE)
POSTAL_CODE_REGEX = re.compile(r"\b\d{5,6}\b")
NUMERIC_REGEX = re.compile(r"\b\d+\b")
NON_ALPHANUM_REGEX = re.compile(r"[^\w\d]+", re.UNICODE)

ADDRESS_STOPWORDS = {
    'street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
    'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
    'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near',
    'st', 'rd', 'ave', 'dr', 'ln', 'blvd', 'fl', 'ste', 'apt', 'bldg', 'sec'
}

COMMON_BUSINESS_WORDS = {
    'enterprises', 'services', 'solutions', 'holdings', 'group', 'industries', 
    'products', 'international', 'technologies', 'consultancy', 'trading', 
    'ventures', 'associates', 'consultants', 'agency', 'management', 'global',
    'systems', 'energy', 'logistics', 'development', 'financial', 'properties'
}


# ==============================================================================
# 1. HELPER PREPROCESSING & TOKENIZATION FUNCTIONS
# ==============================================================================

def clean_str(s: Any) -> str:
    """Safely converts input to a trimmed Unicode NFKC string."""
    if s is None or pd.isna(s):
        return ""
    text = str(s).strip()
    return unicodedata.normalize("NFKC", text)

def clean_alphanumeric(s: str) -> str:
    """Lowercase alphanumeric string with punctuation removed."""
    if not s:
        return ""
    return NON_ALPHANUM_REGEX.sub("", s.lower()).replace("_", "")

def strip_prefix_noise(s: str) -> str:
    """Removes common leading honorifics / noise tokens."""
    if not s:
        return ""
    return PREFIX_NOISE_REGEX.sub("", s).strip()

def strip_legal_suffix(s: str) -> str:
    """Removes unambiguous trailing legal entity designators."""
    if not s:
        return ""
    return UNAMBIGUOUS_LEGAL_REGEX.sub("", s).strip()

def strip_domain_artifacts(s: str) -> str:
    """Removes domain and URL artifacts."""
    if not s:
        return ""
    return DOMAIN_REGEX.sub("", s).strip()

def get_word_tokens(s: str, min_len: int = 2, stopset: Optional[Set[str]] = None) -> List[str]:
    """Tokenizes string into lowercase word tokens."""
    if not s:
        return []
    toks = [t for t in re.split(r"[^\w]+", s.lower(), flags=re.UNICODE) if len(t) >= min_len and t != "_"]
    if stopset:
        toks = [t for t in toks if t not in stopset]
    return toks

def get_char_3grams(s: str) -> Set[str]:
    """Extracts character 3-grams with boundary markers."""
    if not s:
        return set()
    s_clean = clean_alphanumeric(s)
    if len(s_clean) < 3:
        return {s_clean} if s_clean else set()
    padded = f"^{s_clean}$"
    return {padded[i : i + 3] for i in range(len(padded) - 2)}

def extract_primary_number(addr: str) -> str:
    """Extracts primary building/house number without leading zeros."""
    if not addr:
        return ""
    nums = NUMERIC_REGEX.findall(addr)
    return str(int(nums[0])) if nums else ""

def extract_all_numeric_tokens(addr: str) -> Set[str]:
    """Extracts all numeric tokens from an address string."""
    if not addr:
        return set()
    nums = NUMERIC_REGEX.findall(addr)
    return {str(int(n)) for n in nums if n.isdigit()}

def extract_postal_codes(addr: str) -> Set[str]:
    """Extracts 5-6 digit postal/PIN codes."""
    if not addr:
        return set()
    return set(POSTAL_CODE_REGEX.findall(addr))

def is_native_script(s: str) -> bool:
    """Checks if text contains Indic, Arabic, Cyrillic or other non-Latin scripts."""
    if not s:
        return False
    return any(ord(c) >= 0x0600 for c in s)


# ==============================================================================
# 2. VECTORIZED PAIRWISE SIMILARITY CALCULATION
# ==============================================================================

def compute_token_similarities(toks1: List[str], toks2: List[str]) -> Tuple[float, float, int, int, int]:
    """
    Computes Jaccard, Overlap Coefficient, shared count, containment s1 in s2, containment s2 in s1.
    """
    s1, s2 = set(toks1), set(toks2)
    if not s1 or not s2:
        return 0.0, 0.0, 0, 0, 0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    min_len = min(len(s1), len(s2))
    
    jaccard = inter / union if union > 0 else 0.0
    overlap = inter / min_len if min_len > 0 else 0.0
    cont_s1 = 1 if s1.issubset(s2) else 0
    cont_s2 = 1 if s2.issubset(s1) else 0
    return jaccard, overlap, inter, cont_s1, cont_s2


# ==============================================================================
# 3. HIGH-PERFORMANCE PAIRWISE FEATURE EXTRACTOR CLASS
# ==============================================================================

class PairwiseFeatureExtractor:
    """
    Computes all 65 pairwise matching features for candidate (S1, S2/S3) pairs.
    Compatible with both pandas DataFrames and DuckDB connection pipelines.
    Feature schema is defined in src.feature_schema.
    """

    def __init__(self):
        self.feature_names = schema_get_feature_names()

    def get_feature_names(self) -> List[str]:
        """Returns the canonical list of 65 extracted feature column names."""
        return list(self.feature_names)

    def extract_features(
        self,
        df_pairs: pd.DataFrame,
        s1_lookup: Dict[str, Dict[str, str]],
        cand_lookup: Dict[str, Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Extracts pairwise feature matrix from a dataframe of candidate pairs.
        df_pairs must contain:
          'source1_entity_id', 'candidate_entity_id', and optionally provenance columns:
          'total_priority', 'channels_fired', 'rank_order', 'fired_chan_a', ...
        """
        n_pairs = len(df_pairs)
        if n_pairs == 0:
            return pd.DataFrame(columns=self.feature_names)

        # Preallocate feature arrays for maximum speed
        feats = {col: np.zeros(n_pairs, dtype=np.float32) for col in self.feature_names}

        # Provenance columns from candidate generation if available
        prov_cols = [
            "fired_chan_a", "fired_chan_a2", "fired_chan_b", "fired_chan_c",
            "fired_chan_d", "fired_chan_e", "fired_chan_e2", "fired_chan_g",
            "channels_fired_count", "blocking_priority_score", "blocking_rank_order"
        ]
        
        # Populate provenance directly if present in df_pairs
        if "fired_chan_a" in df_pairs.columns:
            for c in ["fired_chan_a", "fired_chan_a2", "fired_chan_b", "fired_chan_c",
                      "fired_chan_d", "fired_chan_e", "fired_chan_e2", "fired_chan_g"]:
                feats[c] = df_pairs[c].fillna(0).values.astype(np.float32)
        if "channels_fired" in df_pairs.columns:
            feats["channels_fired_count"] = df_pairs["channels_fired"].fillna(1).values.astype(np.float32)
        elif "channels_fired_count" in df_pairs.columns:
            feats["channels_fired_count"] = df_pairs["channels_fired_count"].fillna(1).values.astype(np.float32)
        if "total_priority" in df_pairs.columns:
            feats["blocking_priority_score"] = df_pairs["total_priority"].fillna(0).values.astype(np.float32)
        if "rank_order" in df_pairs.columns:
            feats["blocking_rank_order"] = df_pairs["rank_order"].fillna(1).values.astype(np.float32)

        s1_ids = df_pairs["source1_entity_id"].values
        cand_ids = df_pairs["candidate_entity_id"].values

        for idx in range(n_pairs):
            s1_id = s1_ids[idx]
            cand_id = cand_ids[idx]

            s1_rec = s1_lookup.get(s1_id, {})
            cand_rec = cand_lookup.get(cand_id, {})

            s1_country = s1_rec.get("country", "")
            cand_country = cand_rec.get("country", "")

            s1_name_raw = s1_rec.get("business_name", "")
            cand_name_raw = cand_rec.get("business_name", "")

            s1_addr_raw = s1_rec.get("business_address", "")
            cand_addr_raw = cand_rec.get("business_address", "")

            # ------------------------------------------------------------------
            # Group A: Country Features
            # ------------------------------------------------------------------
            c_match = 1.0 if (s1_country and s1_country == cand_country) else 0.0
            c_miss = 1.0 if (not s1_country or not cand_country) else 0.0
            feats["country_exact_match"][idx] = c_match
            feats["country_missing_either"][idx] = c_miss

            # ------------------------------------------------------------------
            # Group B: Name Similarity Features
            # ------------------------------------------------------------------
            s1_n_clean = clean_str(s1_name_raw)
            cand_n_clean = clean_str(cand_name_raw)

            name_missing = (not s1_n_clean or not cand_n_clean)
            feats["name_missing_either"][idx] = 1.0 if name_missing else 0.0

            if not name_missing:
                # 1. Exact Name Matches across representations
                s1_n_low = s1_n_clean.lower()
                cand_n_low = cand_n_clean.lower()
                feats["name_raw_exact"][idx] = 1.0 if s1_n_low == cand_n_low else 0.0

                s1_n_legal = strip_legal_suffix(s1_n_clean).lower()
                cand_n_legal = strip_legal_suffix(cand_n_clean).lower()
                feats["name_legal_stripped_exact"][idx] = 1.0 if (s1_n_legal and s1_n_legal == cand_n_legal) else 0.0

                s1_n_dom = strip_domain_artifacts(s1_n_clean).lower()
                cand_n_dom = strip_domain_artifacts(cand_n_clean).lower()
                feats["name_domain_stripped_exact"][idx] = 1.0 if (s1_n_dom and s1_n_dom == cand_n_dom) else 0.0

                s1_n_alpha = clean_alphanumeric(s1_n_clean)
                cand_n_alpha = clean_alphanumeric(cand_n_clean)
                feats["name_alphanumeric_exact"][idx] = 1.0 if (s1_n_alpha and s1_n_alpha == cand_n_alpha) else 0.0

                # 2. Character / Edit Similarities
                if s1_n_alpha and cand_n_alpha:
                    feats["name_levenshtein_sim"][idx] = Levenshtein.normalized_similarity(s1_n_alpha, cand_n_alpha)
                    feats["name_jaro_winkler_sim"][idx] = JaroWinkler.similarity(s1_n_alpha, cand_n_alpha)
                    
                    g1 = get_char_3grams(s1_n_alpha)
                    g2 = get_char_3grams(cand_n_alpha)
                    g_inter = len(g1 & g2)
                    g_union = len(g1 | g2)
                    feats["name_char_3gram_jaccard"][idx] = g_inter / g_union if g_union > 0 else 0.0

                    l1, l2 = len(s1_n_alpha), len(cand_n_alpha)
                    feats["name_length_diff"][idx] = float(abs(l1 - l2))
                    feats["name_length_ratio"][idx] = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

                # 3. Token Similarities
                toks1 = get_word_tokens(s1_n_clean)
                toks2 = get_word_tokens(cand_n_clean)
                tj, to, t_inter, c1, c2 = compute_token_similarities(toks1, toks2)
                feats["name_token_jaccard"][idx] = tj
                feats["name_token_overlap_coef"][idx] = to
                feats["name_token_containment_s1"][idx] = float(c1)
                feats["name_token_containment_cand"][idx] = float(c2)
                feats["name_shared_token_count"][idx] = float(t_inter)
                feats["name_token_count_diff"][idx] = float(abs(len(toks1) - len(toks2)))

                if toks1 and toks2:
                    feats["name_first_token_match"][idx] = 1.0 if toks1[0] == toks2[0] else 0.0
                    feats["name_last_token_match"][idx] = 1.0 if toks1[-1] == toks2[-1] else 0.0

                # 4. Legal Suffix & Business Word Signals
                has_leg1 = bool(UNAMBIGUOUS_LEGAL_REGEX.search(s1_n_clean))
                has_leg2 = bool(UNAMBIGUOUS_LEGAL_REGEX.search(cand_n_clean))
                m1 = UNAMBIGUOUS_LEGAL_REGEX.search(s1_n_clean)
                m2 = UNAMBIGUOUS_LEGAL_REGEX.search(cand_n_clean)
                leg_str1 = m1.group(1).lower() if m1 else ""
                leg_str2 = m2.group(1).lower() if m2 else ""

                if (not has_leg1 and not has_leg2) or (has_leg1 and has_leg2 and leg_str1 == leg_str2):
                    feats["legal_suffix_agreement"][idx] = 1.0
                elif has_leg1 and has_leg2 and leg_str1 != leg_str2:
                    feats["legal_suffix_mismatch"][idx] = 1.0

                bwords1 = set(toks1) & COMMON_BUSINESS_WORDS
                bwords2 = set(toks2) & COMMON_BUSINESS_WORDS
                feats["business_word_overlap_count"][idx] = float(len(bwords1 & bwords2))

                # 5. Script & Non-ASCII Signals
                is_nat1 = is_native_script(s1_n_clean)
                is_nat2 = is_native_script(cand_n_clean)
                feats["name_both_latin"][idx] = 1.0 if (not is_nat1 and not is_nat2) else 0.0
                feats["name_cross_script"][idx] = 1.0 if (is_nat1 != is_nat2) else 0.0
                feats["name_has_non_ascii"][idx] = 1.0 if (any(ord(c) > 127 for c in s1_n_clean) or any(ord(c) > 127 for c in cand_n_clean)) else 0.0

            # ------------------------------------------------------------------
            # Group C: Address Similarity Features
            # ------------------------------------------------------------------
            s1_a_clean = clean_str(s1_addr_raw)
            cand_a_clean = clean_str(cand_addr_raw)

            addr_miss_s1 = (not s1_a_clean)
            addr_miss_cand = (not cand_a_clean)
            feats["address_missing_s1"][idx] = 1.0 if addr_miss_s1 else 0.0
            feats["address_missing_cand"][idx] = 1.0 if addr_miss_cand else 0.0

            if not addr_miss_s1 and not addr_miss_cand:
                s1_a_low = s1_a_clean.lower()
                cand_a_low = cand_a_clean.lower()
                feats["address_raw_exact"][idx] = 1.0 if s1_a_low == cand_a_low else 0.0

                # Token similarities (with address stopwords stripped)
                atoks1 = get_word_tokens(s1_a_clean, min_len=3, stopset=ADDRESS_STOPWORDS)
                atoks2 = get_word_tokens(cand_a_clean, min_len=3, stopset=ADDRESS_STOPWORDS)
                atj, ato, at_inter, ac1, ac2 = compute_token_similarities(atoks1, atoks2)
                feats["address_token_jaccard"][idx] = atj
                feats["address_token_overlap_coef"][idx] = ato
                feats["address_shared_token_count"][idx] = float(at_inter)
                feats["address_token_containment"][idx] = float(ac1 or ac2)
                feats["address_token_count_diff"][idx] = float(abs(len(atoks1) - len(atoks2)))

                all_atoks1 = get_word_tokens(s1_a_clean, min_len=2)
                all_atoks2 = get_word_tokens(cand_a_clean, min_len=2)
                if all_atoks1 and all_atoks2:
                    feats["address_first_token_match"][idx] = 1.0 if all_atoks1[0] == all_atoks2[0] else 0.0
                    feats["address_last_token_match"][idx] = 1.0 if all_atoks1[-1] == all_atoks2[-1] else 0.0

                # Building / Numeric Token Anchors
                pnum1 = extract_primary_number(s1_a_clean)
                pnum2 = extract_primary_number(cand_a_clean)
                feats["address_building_num_match"][idx] = 1.0 if (pnum1 and pnum1 == pnum2) else 0.0

                nums1 = extract_all_numeric_tokens(s1_a_clean)
                nums2 = extract_all_numeric_tokens(cand_a_clean)
                shared_nums = nums1 & nums2
                union_nums = nums1 | nums2
                feats["address_shared_numeric_count"][idx] = float(len(shared_nums))
                feats["address_has_shared_numeric"][idx] = 1.0 if len(shared_nums) > 0 else 0.0
                feats["address_numeric_token_jaccard"][idx] = len(shared_nums) / len(union_nums) if union_nums else 0.0

                # Postal Code Verification
                post1 = extract_postal_codes(s1_a_clean)
                post2 = extract_postal_codes(cand_a_clean)
                has_post1 = len(post1) > 0
                has_post2 = len(post2) > 0
                feats["address_postal_both_present"][idx] = 1.0 if (has_post1 and has_post2) else 0.0
                if has_post1 and has_post2:
                    if len(post1 & post2) > 0:
                        feats["address_postal_exact_match"][idx] = 1.0
                    else:
                        feats["address_postal_mismatch"][idx] = 1.0

            # ------------------------------------------------------------------
            # Group D: Cross-Field Interaction Features
            # ------------------------------------------------------------------
            feats["exact_name_and_address_number_match"][idx] = (
                1.0 if (feats["name_alphanumeric_exact"][idx] == 1.0 and feats["address_building_num_match"][idx] == 1.0) else 0.0
            )
            feats["high_name_sim_and_address_overlap"][idx] = (
                1.0 if (feats["name_levenshtein_sim"][idx] >= 0.85 and feats["address_token_jaccard"][idx] >= 0.30) else 0.0
            )
            feats["shared_name_and_shared_number"][idx] = (
                1.0 if (feats["name_shared_token_count"][idx] >= 1.0 and feats["address_has_shared_numeric"][idx] == 1.0) else 0.0
            )
            feats["exact_postal_and_strong_name"][idx] = (
                1.0 if (feats["address_postal_exact_match"][idx] == 1.0 and feats["name_token_jaccard"][idx] >= 0.50) else 0.0
            )
            feats["cross_script_and_strong_address"][idx] = (
                1.0 if (feats["name_cross_script"][idx] == 1.0 and (feats["address_building_num_match"][idx] == 1.0 or feats["address_token_jaccard"][idx] >= 0.35)) else 0.0
            )
            feats["weak_name_and_strong_address"][idx] = (
                1.0 if (feats["name_token_jaccard"][idx] < 0.20 and feats["address_building_num_match"][idx] == 1.0 and feats["address_shared_token_count"][idx] >= 2.0) else 0.0
            )
            feats["name_missing_and_strong_address"][idx] = (
                1.0 if (feats["name_missing_either"][idx] == 1.0 and feats["address_token_jaccard"][idx] >= 0.50) else 0.0
            )

            # ------------------------------------------------------------------
            # Group E: Source-Specific Features
            # ------------------------------------------------------------------
            is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
            is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0
            feats["candidate_is_source2"][idx] = is_s2
            feats["candidate_is_source3"][idx] = is_s3
            feats["cand_name_has_url"][idx] = 1.0 if bool(DOMAIN_REGEX.search(cand_n_clean)) else 0.0
            feats["cand_name_has_noise_prefix"][idx] = 1.0 if bool(PREFIX_NOISE_REGEX.search(cand_n_clean)) else 0.0

        # Construct final DataFrame retaining candidate identification columns
        df_out = pd.DataFrame(feats)
        df_out.insert(0, "source1_entity_id", df_pairs["source1_entity_id"].values)
        df_out.insert(1, "candidate_entity_id", df_pairs["candidate_entity_id"].values)
        return df_out

    def attach_ground_truth_labels(
        self,
        df_features: pd.DataFrame,
        gt_pairs_set: Set[Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Attaches the binary ground-truth target label:
          1 if (source1_entity_id, candidate_entity_id) in gt_pairs_set, else 0.
        Ensures zero label leakage.
        """
        s1_arr = df_features["source1_entity_id"].values
        cand_arr = df_features["candidate_entity_id"].values
        labels = np.zeros(len(df_features), dtype=np.int32)

        for i in range(len(df_features)):
            if (s1_arr[i], cand_arr[i]) in gt_pairs_set:
                labels[i] = 1

        df_result = df_features.copy()
        df_result["match_label"] = labels
        return df_result

    def assign_s1_group_split(
        self,
        df_features: pd.DataFrame,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Performs S1-level grouped train/validation splitting.
        All candidate pairs belonging to the same Source 1 entity are strictly
        assigned to the SAME split group ('train' or 'val') to eliminate leakage.
        """
        unique_s1 = sorted(df_features["source1_entity_id"].unique())
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(unique_s1)

        val_size = int(len(shuffled) * val_ratio)
        val_set = set(shuffled[:val_size])

        split_col = np.array(["val" if s1 in val_set else "train" for s1 in df_features["source1_entity_id"].values])
        df_result = df_features.copy()
        df_result["split_group"] = split_col
        return df_result
