"""
Pairwise Feature Engineering Module (Config D1: Exactly 76 Features)
Amazon ML Challenge 2026: Business Entity Resolution

Extracts the exact 76 features of the frozen Config D1 architecture:
- 65 Canonical Features (Groups A-F)
- 9 Selected Phase 8 Features (Groups G, H, I, K-part1)
- 2 Group K Address Disambiguation Features (feat_same_building_weak_name, feat_high_addr_low_name_penalty)
"""

import re
import unicodedata
from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np
import pandas as pd
import rapidfuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src.feature_schema import FEATURE_NAMES, get_feature_names

# Precompiled regex patterns
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
    re.IGNORECASE,
)

PREFIX_NOISE_REGEX = re.compile(r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|mr\s+|co\s+|>>\s+|#\s*)", re.IGNORECASE)
DOMAIN_REGEX = re.compile(r"\.(?:com|org|net|in|co|io|biz|info|gov|edu|ai|us|fr)\b|^www\.", re.IGNORECASE)
POSTAL_CODE_REGEX = re.compile(r"\b\d{5,6}\b")
NUMERIC_REGEX = re.compile(r"\b\d+\b")
NON_ALPHANUM_REGEX = re.compile(r"[^\w\d]+", re.UNICODE)

ADDRESS_STOPWORDS = {
    "street", "road", "avenue", "drive", "lane", "boulevard", "floor", "suite",
    "apartment", "building", "sector", "district", "nagar", "north", "south",
    "east", "west", "house", "block", "first", "second", "third", "opposite", "near",
    "st", "rd", "ave", "dr", "ln", "blvd", "fl", "ste", "apt", "bldg", "sec"
}

COMMON_BUSINESS_WORDS = {
    "enterprises", "services", "solutions", "holdings", "group", "industries",
    "products", "international", "technologies", "consultancy", "trading",
    "ventures", "associates", "consultants", "agency", "management", "global",
    "systems", "energy", "logistics", "development", "financial", "properties"
}


def clean_str(s: Any) -> str:
    if s is None or pd.isna(s):
        return ""
    text = str(s).strip()
    return unicodedata.normalize("NFKC", text)


def clean_alphanumeric(s: str) -> str:
    if not s:
        return ""
    return NON_ALPHANUM_REGEX.sub("", s.lower()).replace("_", "")


def strip_prefix_noise(s: str) -> str:
    if not s:
        return ""
    return PREFIX_NOISE_REGEX.sub("", s).strip()


def strip_legal_suffix(s: str) -> str:
    if not s:
        return ""
    return UNAMBIGUOUS_LEGAL_REGEX.sub("", s).strip()


def strip_domain_artifacts(s: str) -> str:
    if not s:
        return ""
    return DOMAIN_REGEX.sub("", s).strip()


def get_word_tokens(s: str, min_len: int = 1, stopset: Optional[Set[str]] = None) -> List[str]:
    if not s:
        return []
    cleaned = NON_ALPHANUM_REGEX.sub(" ", s.lower())
    toks = [w for w in cleaned.split() if len(w) >= min_len]
    if stopset:
        toks = [w for w in toks if w not in stopset]
    return toks


def get_char_3grams(s: str) -> Set[str]:
    if not s or len(s) < 3:
        return set()
    return {s[i:i+3] for i in range(len(s) - 2)}


def get_char_ngrams(s: str, n: int) -> Set[str]:
    if not s or len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def extract_primary_number(s: str) -> str:
    if not s:
        return ""
    m = NUMERIC_REGEX.search(s)
    return m.group(0) if m else ""


def extract_all_numeric_tokens(s: str) -> Set[str]:
    if not s:
        return set()
    return set(NUMERIC_REGEX.findall(s))


def extract_postal_codes(s: str) -> Set[str]:
    if not s:
        return set()
    return set(POSTAL_CODE_REGEX.findall(s))


def is_native_script(s: str) -> bool:
    if not s:
        return False
    return any(ord(c) >= 0x0900 for c in s)


def compute_token_similarities(toks1: List[str], toks2: List[str]) -> Tuple[float, float, int, float, float]:
    if not toks1 or not toks2:
        return 0.0, 0.0, 0, 0.0, 0.0
    s1, s2 = set(toks1), set(toks2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    jaccard = inter / union if union > 0 else 0.0
    overlap = inter / min(len(s1), len(s2)) if min(len(s1), len(s2)) > 0 else 0.0
    c1 = inter / len(s1) if len(s1) > 0 else 0.0
    c2 = inter / len(s2) if len(s2) > 0 else 0.0
    return jaccard, overlap, inter, c1, c2


def get_distinctive_tokens(toks: List[str]) -> List[str]:
    return [w for w in toks if len(w) >= 3 and w not in ADDRESS_STOPWORDS and w not in COMMON_BUSINESS_WORDS]


def compute_best_token_similarity(toks1: List[str], toks2: List[str]) -> float:
    if not toks1 or not toks2:
        return 0.0
    best = 0.0
    for t1 in toks1:
        for t2 in toks2:
            sim = float(Levenshtein.normalized_similarity(t1, t2))
            if sim > best:
                best = sim
                if best == 1.0:
                    return 1.0
    return best


class PairwiseFeatureExtractor:
    """
    Production Pairwise Feature Extractor implementing the exact 76 features
    of the frozen Config D1 architecture.
    """

    def __init__(self):
        self.feature_names = list(FEATURE_NAMES)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    def extract_features(
        self,
        df_pairs: pd.DataFrame,
        s1_lookup: Dict[str, Dict[str, str]],
        cand_lookup: Dict[str, Dict[str, str]],
    ) -> pd.DataFrame:
        n_pairs = len(df_pairs)
        if n_pairs == 0:
            return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + self.feature_names)

        s1_ids = df_pairs["source1_entity_id"].values
        cand_ids = df_pairs["candidate_entity_id"].values

        # Preallocate dictionary of feature arrays
        f = {col: np.zeros(n_pairs, dtype=np.float32) for col in self.feature_names}

        # Blocking provenance
        for ch in ["a", "a2", "b", "c", "d", "e", "e2", "g"]:
            col = f"fired_chan_{ch}"
            if col in df_pairs.columns and col in f:
                f[col] = df_pairs[col].values.astype(np.float32)

        if "channels_fired" in df_pairs.columns and "channels_fired_count" in f:
            f["channels_fired_count"] = df_pairs["channels_fired"].values.astype(np.float32)
        elif "channels_fired_count" in df_pairs.columns:
            f["channels_fired_count"] = df_pairs["channels_fired_count"].values.astype(np.float32)

        if "total_priority" in df_pairs.columns and "blocking_priority_score" in f:
            f["blocking_priority_score"] = df_pairs["total_priority"].values.astype(np.float32)
        elif "blocking_priority_score" in df_pairs.columns:
            f["blocking_priority_score"] = df_pairs["blocking_priority_score"].values.astype(np.float32)

        if "rank_order" in df_pairs.columns and "blocking_rank_order" in f:
            f["blocking_rank_order"] = df_pairs["rank_order"].values.astype(np.float32)
        elif "blocking_rank_order" in df_pairs.columns:
            f["blocking_rank_order"] = df_pairs["blocking_rank_order"].values.astype(np.float32)

        for idx in range(n_pairs):
            s1_id = s1_ids[idx]
            cand_id = cand_ids[idx]

            s1 = s1_lookup.get(s1_id, {})
            cand = cand_lookup.get(cand_id, {})

            c_s1 = clean_str(s1.get("country", ""))
            c_cand = clean_str(cand.get("country", ""))

            n_s1_raw = clean_str(s1.get("business_name", ""))
            n_cand_raw = clean_str(cand.get("business_name", ""))

            a_s1_raw = clean_str(s1.get("business_address", ""))
            a_cand_raw = clean_str(cand.get("business_address", ""))

            # Group A: Country
            if c_s1 and c_cand:
                f["country_exact_match"][idx] = 1.0 if c_s1 == c_cand else 0.0
            else:
                f["country_missing_either"][idx] = 1.0

            # Group B: Name Similarities
            if not n_s1_raw or not n_cand_raw:
                f["name_missing_either"][idx] = 1.0
            else:
                f["name_raw_exact"][idx] = 1.0 if n_s1_raw.lower() == n_cand_raw.lower() else 0.0

                n_s1_pfx = strip_prefix_noise(n_s1_raw)
                n_cand_pfx = strip_prefix_noise(n_cand_raw)

                n_s1_legal = strip_legal_suffix(n_s1_pfx)
                n_cand_legal = strip_legal_suffix(n_cand_pfx)
                f["name_legal_stripped_exact"][idx] = 1.0 if n_s1_legal.lower() == n_cand_legal.lower() else 0.0

                n_s1_dom = strip_domain_artifacts(n_s1_legal)
                n_cand_dom = strip_domain_artifacts(n_cand_legal)
                f["name_domain_stripped_exact"][idx] = 1.0 if n_s1_dom.lower() == n_cand_dom.lower() else 0.0

                n_s1_alpha = clean_alphanumeric(n_s1_dom)
                n_cand_alpha = clean_alphanumeric(n_cand_dom)
                f["name_alphanumeric_exact"][idx] = 1.0 if (n_s1_alpha and n_s1_alpha == n_cand_alpha) else 0.0

                f["name_levenshtein_sim"][idx] = float(Levenshtein.normalized_similarity(n_s1_dom.lower(), n_cand_dom.lower()))
                f["name_jaro_winkler_sim"][idx] = float(JaroWinkler.similarity(n_s1_dom.lower(), n_cand_dom.lower()))

                # 3-grams
                g1 = get_char_3grams(n_s1_alpha)
                g2 = get_char_3grams(n_cand_alpha)
                g_inter = len(g1 & g2)
                g_union = len(g1 | g2)
                f["name_char_3gram_jaccard"][idx] = g_inter / g_union if g_union > 0 else 0.0

                # 2-grams (Group I Phase 8)
                bg1 = get_char_ngrams(n_s1_alpha, 2)
                bg2 = get_char_ngrams(n_cand_alpha, 2)
                bg_inter = len(bg1 & bg2)
                bg_union = len(bg1 | bg2)
                f["feat_name_char_2gram_jaccard"][idx] = bg_inter / bg_union if bg_union > 0 else 0.0

                len1, len2 = len(n_s1_dom), len(n_cand_dom)
                f["name_length_diff"][idx] = abs(len1 - len2)
                f["name_length_ratio"][idx] = min(len1, len2) / max(1, max(len1, len2))

                toks1 = get_word_tokens(n_s1_dom)
                toks2 = get_word_tokens(n_cand_dom)
                t_jac, t_ovl, t_inter, t_c1, t_c2 = compute_token_similarities(toks1, toks2)
                f["name_token_jaccard"][idx] = t_jac
                f["name_token_overlap_coef"][idx] = t_ovl
                f["name_token_containment_s1"][idx] = t_c1
                f["name_token_containment_cand"][idx] = t_c2
                f["name_shared_token_count"][idx] = t_inter
                f["name_token_count_diff"][idx] = abs(len(toks1) - len(toks2))

                if toks1 and toks2:
                    f["name_first_token_match"][idx] = 1.0 if toks1[0] == toks2[0] else 0.0
                    f["name_last_token_match"][idx] = 1.0 if toks1[-1] == toks2[-1] else 0.0

                # Legal suffix agreement / mismatch
                m1 = UNAMBIGUOUS_LEGAL_REGEX.search(n_s1_raw)
                m2 = UNAMBIGUOUS_LEGAL_REGEX.search(n_cand_raw)
                leg1 = m1.group(1).lower() if m1 else ""
                leg2 = m2.group(1).lower() if m2 else ""
                if (not leg1 and not leg2) or (leg1 and leg2 and leg1 == leg2):
                    f["legal_suffix_agreement"][idx] = 1.0
                elif leg1 and leg2 and leg1 != leg2:
                    f["legal_suffix_mismatch"][idx] = 1.0

                bw1 = set(toks1) & COMMON_BUSINESS_WORDS
                bw2 = set(toks2) & COMMON_BUSINESS_WORDS
                f["business_word_overlap_count"][idx] = float(len(bw1 & bw2))

                # Script checks
                s1_native = is_native_script(n_s1_raw)
                cand_native = is_native_script(n_cand_raw)
                f["name_both_latin"][idx] = 1.0 if (not s1_native and not cand_native) else 0.0
                f["name_cross_script"][idx] = 1.0 if (s1_native != cand_native) else 0.0
                f["name_has_non_ascii"][idx] = 1.0 if (any(ord(c) > 127 for c in n_s1_raw) or any(ord(c) > 127 for c in n_cand_raw)) else 0.0

                # Distinctive tokens & aliases (Group H Phase 8)
                dist1 = get_distinctive_tokens(toks1)
                dist2 = get_distinctive_tokens(toks2)
                d_jac, _, d_inter, _, _ = compute_token_similarities(dist1, dist2)
                f["feat_name_distinctive_jaccard"][idx] = d_jac
                f["feat_distinctive_name_zero_overlap"][idx] = 1.0 if (len(dist1) > 0 and len(dist2) > 0 and d_inter == 0) else 0.0
                f["feat_name_best_token_similarity"][idx] = compute_best_token_similarity(dist1 or toks1, dist2 or toks2)

            # Group C: Address Similarities
            cand_addr_missing = 1.0 if (not a_cand_raw) else 0.0
            s1_addr_missing = 1.0 if (not a_s1_raw) else 0.0
            f["address_missing_s1"][idx] = s1_addr_missing
            f["address_missing_cand"][idx] = cand_addr_missing

            addr_num_match = 0.0
            a_jac = 0.0

            if not s1_addr_missing and not cand_addr_missing:
                f["address_raw_exact"][idx] = 1.0 if a_s1_raw.lower() == a_cand_raw.lower() else 0.0
                a_toks1 = get_word_tokens(a_s1_raw, min_len=3, stopset=ADDRESS_STOPWORDS)
                a_toks2 = get_word_tokens(a_cand_raw, min_len=3, stopset=ADDRESS_STOPWORDS)
                a_jac, a_ovl, a_inter, ac1, ac2 = compute_token_similarities(a_toks1, a_toks2)
                f["address_token_jaccard"][idx] = a_jac
                f["address_token_overlap_coef"][idx] = a_ovl
                f["address_token_containment"][idx] = float(ac1 or ac2)
                f["address_shared_token_count"][idx] = float(a_inter)
                f["address_token_count_diff"][idx] = abs(len(a_toks1) - len(a_toks2))

                all_atoks1 = get_word_tokens(a_s1_raw, min_len=2)
                all_atoks2 = get_word_tokens(a_cand_raw, min_len=2)
                if all_atoks1 and all_atoks2:
                    f["address_first_token_match"][idx] = 1.0 if all_atoks1[0] == all_atoks2[0] else 0.0
                    f["address_last_token_match"][idx] = 1.0 if all_atoks1[-1] == all_atoks2[-1] else 0.0

                num1 = extract_primary_number(a_s1_raw)
                num2 = extract_primary_number(a_cand_raw)
                addr_num_match = 1.0 if (num1 and num2 and num1 == num2) else 0.0
                f["address_building_num_match"][idx] = addr_num_match

                nums1 = extract_all_numeric_tokens(a_s1_raw)
                nums2 = extract_all_numeric_tokens(a_cand_raw)
                shared_nums = nums1 & nums2
                union_nums = nums1 | nums2
                f["address_shared_numeric_count"][idx] = float(len(shared_nums))
                f["address_has_shared_numeric"][idx] = 1.0 if len(shared_nums) > 0 else 0.0
                f["address_numeric_token_jaccard"][idx] = len(shared_nums) / len(union_nums) if union_nums else 0.0

                post1 = extract_postal_codes(a_s1_raw)
                post2 = extract_postal_codes(a_cand_raw)
                has_post1 = len(post1) > 0
                has_post2 = len(post2) > 0
                f["address_postal_both_present"][idx] = 1.0 if (has_post1 and has_post2) else 0.0
                if has_post1 and has_post2:
                    if len(post1 & post2) > 0:
                        f["address_postal_exact_match"][idx] = 1.0
                    else:
                        f["address_postal_mismatch"][idx] = 1.0

                # Group K Address Disambiguation
                f["feat_same_postal_weak_name"][idx] = 1.0 if (f["address_postal_exact_match"][idx] == 1.0 and f["name_token_jaccard"][idx] < 0.20) else 0.0
                f["feat_same_building_weak_name"][idx] = 1.0 if (addr_num_match == 1.0 and f["name_token_jaccard"][idx] < 0.20) else 0.0
                f["feat_high_addr_low_name_penalty"][idx] = 1.0 if (a_jac >= 0.60 and f["name_token_jaccard"][idx] <= 0.25) else 0.0

            # Group G: Missing-Address Recovery
            if cand_addr_missing == 1.0:
                f["feat_name_token_jaccard_cand_addr_missing"][idx] = f["name_token_jaccard"][idx]
                f["feat_name_3gram_cand_addr_missing"][idx] = f["name_char_3gram_jaccard"][idx]
                f["feat_name_containment_cand_addr_missing"][idx] = f["name_token_overlap_coef"][idx]
                f["feat_name_lev_cand_addr_missing"][idx] = f["name_levenshtein_sim"][idx]

            # Group D: Cross-Field Interactions
            f["exact_name_and_address_number_match"][idx] = 1.0 if (f["name_alphanumeric_exact"][idx] == 1.0 and addr_num_match == 1.0) else 0.0
            f["high_name_sim_and_address_overlap"][idx] = 1.0 if (f["name_levenshtein_sim"][idx] >= 0.85 and a_jac >= 0.30) else 0.0
            f["shared_name_and_shared_number"][idx] = 1.0 if (f["name_shared_token_count"][idx] >= 1.0 and f["address_has_shared_numeric"][idx] == 1.0) else 0.0
            f["exact_postal_and_strong_name"][idx] = 1.0 if (f["address_postal_exact_match"][idx] == 1.0 and f["name_token_jaccard"][idx] >= 0.50) else 0.0
            f["cross_script_and_strong_address"][idx] = 1.0 if (f["name_cross_script"][idx] == 1.0 and a_jac >= 0.60) else 0.0
            f["weak_name_and_strong_address"][idx] = 1.0 if (f["name_token_jaccard"][idx] <= 0.20 and a_jac >= 0.60) else 0.0
            f["name_missing_and_strong_address"][idx] = 1.0 if (f["name_missing_either"][idx] == 1.0 and a_jac >= 0.60) else 0.0

            # Group E: Source-Specific Signals
            c_str = str(cand_id)
            if c_str.startswith("S2-") or c_str.startswith("s2-"):
                f["candidate_is_source2"][idx] = 1.0
            elif c_str.startswith("S3-") or c_str.startswith("s3-"):
                f["candidate_is_source3"][idx] = 1.0

            if DOMAIN_REGEX.search(n_cand_raw):
                f["cand_name_has_url"][idx] = 1.0
            if PREFIX_NOISE_REGEX.search(n_cand_raw):
                f["cand_name_has_noise_prefix"][idx] = 1.0

        # Construct DataFrame
        df_out = pd.DataFrame({
            "source1_entity_id": s1_ids,
            "candidate_entity_id": cand_ids,
        })
        for col in self.feature_names:
            df_out[col] = f[col]

        return df_out
