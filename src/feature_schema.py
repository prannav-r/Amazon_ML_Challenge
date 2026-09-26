"""
Authoritative Feature Schema Definition
Amazon ML Challenge 2026: Business Entity Resolution

This module serves as the SINGLE SOURCE OF TRUTH for all pairwise model features:
- Exactly 65 features across 6 functional groups.
- Defines feature names, groups, types, descriptions, missing-value defaults, and thresholds.
- Guarantees deterministic ordering for training, validation, and test inference pipelines.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    dtype: str
    description: str
    missing_value_behavior: str
    threshold: Optional[str] = None
    available_at_test: bool = True
    depends_on_ground_truth: bool = False

# Authoritative list of 65 feature specifications in canonical evaluation order
FEATURE_SPECS: List[FeatureSpec] = [
    # --------------------------------------------------------------------------
    # GROUP A: COUNTRY INVARIANTS (2 features)
    # --------------------------------------------------------------------------
    FeatureSpec(
        name="country_exact_match",
        group="Group A: Country Invariants",
        dtype="float32",
        description="Binary indicator: 1.0 if s1.country equals cand.country, else 0.0.",
        missing_value_behavior="Evaluates to 0.0 if either country is null/empty.",
        threshold="s1.country == cand.country"
    ),
    FeatureSpec(
        name="country_missing_either",
        group="Group A: Country Invariants",
        dtype="float32",
        description="Binary indicator: 1.0 if s1.country or cand.country is null/empty.",
        missing_value_behavior="Evaluates to 1.0 if missing, else 0.0.",
        threshold="s1.country == '' or cand.country == ''"
    ),

    # --------------------------------------------------------------------------
    # GROUP B: NAME SIMILARITY FEATURES (24 features)
    # --------------------------------------------------------------------------
    # Exact Equalities across representations
    FeatureSpec(
        name="name_raw_exact",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if lowercase trimmed raw names are identical.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="s1_name.lower() == cand_name.lower()"
    ),
    FeatureSpec(
        name="name_legal_stripped_exact",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if legal-suffix-stripped names are identical.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="strip_legal(s1) == strip_legal(cand)"
    ),
    FeatureSpec(
        name="name_domain_stripped_exact",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if domain/URL-stripped names are identical.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="strip_domain(s1) == strip_domain(cand)"
    ),
    FeatureSpec(
        name="name_alphanumeric_exact",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if lowercase alphanumeric stripped names match.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="alphanumeric(s1) == alphanumeric(cand)"
    ),

    # Character & Edit Similarities
    FeatureSpec(
        name="name_levenshtein_sim",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Normalized Levenshtein similarity: 1 - dist / max(L1, L2) on alphanumeric names.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="name_jaro_winkler_sim",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Jaro-Winkler similarity on cleaned alphanumeric names.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="name_char_3gram_jaccard",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Jaccard similarity over boundary-padded character 3-grams of alphanumeric names.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="name_length_diff",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Absolute difference in alphanumeric character count: |L1 - L2|.",
        missing_value_behavior="Evaluates to 0.0 if both missing, else length of present string.",
        threshold="Non-negative integer"
    ),
    FeatureSpec(
        name="name_length_ratio",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Length ratio: min(L1, L2) / max(L1, L2).",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="Continuous in [0.0, 1.0]"
    ),

    # Token Similarities
    FeatureSpec(
        name="name_token_jaccard",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Word token Jaccard similarity: |T1 ∩ T2| / |T1 ∪ T2|.",
        missing_value_behavior="Evaluates to 0.0 if either token set is empty.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="name_token_overlap_coef",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Word token Overlap/Simpson coefficient: |T1 ∩ T2| / min(|T1|, |T2|).",
        missing_value_behavior="Evaluates to 0.0 if either token set is empty.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="name_token_containment_s1",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if S1 word tokens are a strict subset of Candidate tokens.",
        missing_value_behavior="Evaluates to 0.0 if either token set is empty.",
        threshold="T1 ⊆ T2"
    ),
    FeatureSpec(
        name="name_token_containment_cand",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if Candidate word tokens are a strict subset of S1 tokens.",
        missing_value_behavior="Evaluates to 0.0 if either token set is empty.",
        threshold="T2 ⊆ T1"
    ),
    FeatureSpec(
        name="name_shared_token_count",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Count of identical word tokens shared between S1 and Candidate names: |T1 ∩ T2|.",
        missing_value_behavior="Evaluates to 0.0 if either token set is empty.",
        threshold="Non-negative integer"
    ),
    FeatureSpec(
        name="name_token_count_diff",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Absolute difference in word token counts: ||T1| - |T2||.",
        missing_value_behavior="Evaluates to 0.0 if both missing, else count of present tokens.",
        threshold="Non-negative integer"
    ),
    FeatureSpec(
        name="name_first_token_match",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if first word token matches exactly.",
        missing_value_behavior="Evaluates to 0.0 if either token set is empty.",
        threshold="T1[0] == T2[0]"
    ),
    FeatureSpec(
        name="name_last_token_match",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if last word token matches exactly.",
        missing_value_behavior="Evaluates to 0.0 if either token set is empty.",
        threshold="T1[-1] == T2[-1]"
    ),

    # Legal & Business Semantics
    FeatureSpec(
        name="legal_suffix_agreement",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if both names share identical legal suffix or both have none.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="leg1 == leg2"
    ),
    FeatureSpec(
        name="legal_suffix_mismatch",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if both names have recognized legal suffixes and they conflict.",
        missing_value_behavior="Evaluates to 0.0 if either suffix is missing.",
        threshold="has_leg1 and has_leg2 and leg1 != leg2"
    ),
    FeatureSpec(
        name="business_word_overlap_count",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Count of shared common business words (e.g. enterprises, technologies, logistics).",
        missing_value_behavior="Evaluates to 0.0 if no shared business words.",
        threshold="Non-negative integer"
    ),

    # Script & Alphabet Signals
    FeatureSpec(
        name="name_both_latin",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if both business names contain exclusively Latin script.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="not is_native(s1) and not is_native(cand)"
    ),
    FeatureSpec(
        name="name_cross_script",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if one name is Latin and the other contains native Indic script.",
        missing_value_behavior="Evaluates to 0.0 if either name is missing.",
        threshold="is_native(s1) != is_native(cand)"
    ),
    FeatureSpec(
        name="name_has_non_ascii",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if either business name contains non-ASCII characters.",
        missing_value_behavior="Evaluates to 0.0 if both names are missing.",
        threshold="any(ord > 127 in s1 or cand)"
    ),
    FeatureSpec(
        name="name_missing_either",
        group="Group B: Name Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if either S1 or Candidate business name is empty/null.",
        missing_value_behavior="Evaluates to 1.0 if missing, else 0.0.",
        threshold="s1_name == '' or cand_name == ''"
    ),

    # --------------------------------------------------------------------------
    # GROUP C: ADDRESS & GEOGRAPHIC SIMILARITIES (17 features)
    # --------------------------------------------------------------------------
    FeatureSpec(
        name="address_raw_exact",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if normalized address strings are exactly identical.",
        missing_value_behavior="Evaluates to 0.0 if either address is missing.",
        threshold="s1_addr.lower() == cand_addr.lower()"
    ),
    FeatureSpec(
        name="address_token_jaccard",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Address word token Jaccard similarity (excluding stopwords like street, road).",
        missing_value_behavior="Evaluates to 0.0 if either address is missing.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="address_token_overlap_coef",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Address word token Overlap coefficient: |A1 ∩ A2| / min(|A1|, |A2|).",
        missing_value_behavior="Evaluates to 0.0 if either address is missing.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="address_shared_token_count",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Count of distinctive address word tokens shared between S1 and Candidate.",
        missing_value_behavior="Evaluates to 0.0 if either address is missing.",
        threshold="Non-negative integer"
    ),
    FeatureSpec(
        name="address_token_containment",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if address tokens of one side are a subset of the other.",
        missing_value_behavior="Evaluates to 0.0 if either address is missing.",
        threshold="A1 ⊆ A2 or A2 ⊆ A1"
    ),
    FeatureSpec(
        name="address_token_count_diff",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Absolute difference in address token counts: ||A1| - |A2||.",
        missing_value_behavior="Evaluates to 0.0 if both missing, else count of present tokens.",
        threshold="Non-negative integer"
    ),
    FeatureSpec(
        name="address_first_token_match",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if the first address token matches (e.g. street number).",
        missing_value_behavior="Evaluates to 0.0 if either address is missing.",
        threshold="A1[0] == A2[0]"
    ),
    FeatureSpec(
        name="address_last_token_match",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if the last address token matches (e.g. city/PIN).",
        missing_value_behavior="Evaluates to 0.0 if either address is missing.",
        threshold="A1[-1] == A2[-1]"
    ),
    FeatureSpec(
        name="address_building_num_match",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if primary building/house numbers match exactly.",
        missing_value_behavior="Evaluates to 0.0 if either address lacks a number.",
        threshold="pnum1 != '' and pnum1 == pnum2"
    ),
    FeatureSpec(
        name="address_shared_numeric_count",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Count of shared numeric tokens (house, unit, floor, PIN) in address.",
        missing_value_behavior="Evaluates to 0.0 if either address lacks numbers.",
        threshold="Non-negative integer"
    ),
    FeatureSpec(
        name="address_numeric_token_jaccard",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Jaccard similarity over numeric tokens in address: |N1 ∩ N2| / |N1 ∪ N2|.",
        missing_value_behavior="Evaluates to 0.0 if either address lacks numbers.",
        threshold="Continuous in [0.0, 1.0]"
    ),
    FeatureSpec(
        name="address_has_shared_numeric",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if at least one numeric token is shared.",
        missing_value_behavior="Evaluates to 0.0 if no numeric overlap.",
        threshold="len(N1 ∩ N2) > 0"
    ),
    FeatureSpec(
        name="address_postal_exact_match",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if extracted 5-6 digit postal codes match.",
        missing_value_behavior="Evaluates to 0.0 if either address lacks a postal code.",
        threshold="len(P1 ∩ P2) > 0"
    ),
    FeatureSpec(
        name="address_postal_both_present",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if both addresses contain a 5-6 digit postal code.",
        missing_value_behavior="Evaluates to 0.0 if either lacks postal code.",
        threshold="has_post1 and has_post2"
    ),
    FeatureSpec(
        name="address_postal_mismatch",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if both contain postal codes and they conflict.",
        missing_value_behavior="Evaluates to 0.0 if either lacks postal code.",
        threshold="has_post1 and has_post2 and P1 ∩ P2 == ∅"
    ),
    FeatureSpec(
        name="address_missing_s1",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if Source 1 address is null or empty.",
        missing_value_behavior="Evaluates to 1.0 if missing, else 0.0.",
        threshold="s1_addr == ''"
    ),
    FeatureSpec(
        name="address_missing_cand",
        group="Group C: Address Similarities",
        dtype="float32",
        description="Binary indicator: 1.0 if Candidate address is null or empty.",
        missing_value_behavior="Evaluates to 1.0 if missing, else 0.0.",
        threshold="cand_addr == ''"
    ),

    # --------------------------------------------------------------------------
    # GROUP D: CROSS-FIELD INTERACTION SIGNALS (7 features)
    # --------------------------------------------------------------------------
    FeatureSpec(
        name="exact_name_and_address_number_match",
        group="Group D: Cross-Field Interactions",
        dtype="float32",
        description="1.0 if exact alphanumeric name match AND exact building number match.",
        missing_value_behavior="Evaluates to 0.0 if any component is missing/false.",
        threshold="(name_alphanumeric_exact == 1.0) and (address_building_num_match == 1.0)"
    ),
    FeatureSpec(
        name="high_name_sim_and_address_overlap",
        group="Group D: Cross-Field Interactions",
        dtype="float32",
        description="1.0 if high name similarity AND substantial address token overlap.",
        missing_value_behavior="Evaluates to 0.0 if any component is missing/false.",
        threshold="(name_levenshtein_sim >= 0.85) and (address_token_jaccard >= 0.30)"
    ),
    FeatureSpec(
        name="shared_name_and_shared_number",
        group="Group D: Cross-Field Interactions",
        dtype="float32",
        description="1.0 if at least 1 name word matches AND at least 1 numeric token matches.",
        missing_value_behavior="Evaluates to 0.0 if any component is missing/false.",
        threshold="(name_shared_token_count >= 1.0) and (address_has_shared_numeric == 1.0)"
    ),
    FeatureSpec(
        name="exact_postal_and_strong_name",
        group="Group D: Cross-Field Interactions",
        dtype="float32",
        description="1.0 if exact postal code match AND moderate-to-high name token overlap.",
        missing_value_behavior="Evaluates to 0.0 if any component is missing/false.",
        threshold="(address_postal_exact_match == 1.0) and (name_token_jaccard >= 0.50)"
    ),
    FeatureSpec(
        name="cross_script_and_strong_address",
        group="Group D: Cross-Field Interactions",
        dtype="float32",
        description="1.0 if cross-script pair AND strong address evidence (number or locality).",
        missing_value_behavior="Evaluates to 0.0 if any component is missing/false.",
        threshold="(name_cross_script == 1.0) and ((address_building_num_match == 1.0) or (address_token_jaccard >= 0.35))"
    ),
    FeatureSpec(
        name="weak_name_and_strong_address",
        group="Group D: Cross-Field Interactions",
        dtype="float32",
        description="1.0 if weak name overlap (alias/rebrand) BUT strong address anchors (number + 2 tokens).",
        missing_value_behavior="Evaluates to 0.0 if any component is missing/false.",
        threshold="(name_token_jaccard < 0.20) and (address_building_num_match == 1.0) and (address_shared_token_count >= 2.0)"
    ),
    FeatureSpec(
        name="name_missing_and_strong_address",
        group="Group D: Cross-Field Interactions",
        dtype="float32",
        description="1.0 if name is missing on either side BUT address token overlap is very high.",
        missing_value_behavior="Evaluates to 0.0 if name is present or address overlap is low.",
        threshold="(name_missing_either == 1.0) and (address_token_jaccard >= 0.50)"
    ),

    # --------------------------------------------------------------------------
    # GROUP E: SOURCE-SPECIFIC SIGNALS (4 features)
    # --------------------------------------------------------------------------
    FeatureSpec(
        name="candidate_is_source2",
        group="Group E: Source-Specific Signals",
        dtype="float32",
        description="Binary indicator: 1.0 if candidate record originates from Source 2.",
        missing_value_behavior="Evaluates to 0.0 if candidate is from Source 3.",
        threshold="candidate_entity_id.startswith('S2-')"
    ),
    FeatureSpec(
        name="candidate_is_source3",
        group="Group E: Source-Specific Signals",
        dtype="float32",
        description="Binary indicator: 1.0 if candidate record originates from Source 3.",
        missing_value_behavior="Evaluates to 0.0 if candidate is from Source 2.",
        threshold="candidate_entity_id.startswith('S3-')"
    ),
    FeatureSpec(
        name="cand_name_has_url",
        group="Group E: Source-Specific Signals",
        dtype="float32",
        description="Binary indicator: 1.0 if Candidate business name contains domain/URL patterns.",
        missing_value_behavior="Evaluates to 0.0 if no URL pattern detected.",
        threshold="DOMAIN_REGEX.search(cand_name) is not None"
    ),
    FeatureSpec(
        name="cand_name_has_noise_prefix",
        group="Group E: Source-Specific Signals",
        dtype="float32",
        description="Binary indicator: 1.0 if Candidate name begins with honorific/noise prefix.",
        missing_value_behavior="Evaluates to 0.0 if no prefix detected.",
        threshold="PREFIX_NOISE_REGEX.search(cand_name) is not None"
    ),

    # --------------------------------------------------------------------------
    # GROUP F: BLOCKING PROVENANCE SIGNALS (11 features)
    # --------------------------------------------------------------------------
    FeatureSpec(
        name="fired_chan_a",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel A (Exact Normalized Core Name).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="fired_chan_a2",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel A2 (Prefix-Stripped Core Name).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="fired_chan_b",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel B (Rare Name Tokens DF <= 200).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="fired_chan_c",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel C (Character 4-gram Inverted Index).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="fired_chan_d",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel D (Distinctive Address Tokens DF <= 150).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="fired_chan_e",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel E (Address Number + Name Prefix-3).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="fired_chan_e2",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel E2 (Address Number + Distinctive Locality).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="fired_chan_g",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="1.0 if pair was retrieved by Channel G (1-Edit Initial-Char Typo Key DF <= 150).",
        missing_value_behavior="Evaluates to 0.0 if channel did not retrieve pair.",
        threshold="Candidate generation provenance"
    ),
    FeatureSpec(
        name="channels_fired_count",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="Total number of distinct blocking channels that retrieved this candidate pair.",
        missing_value_behavior="Defaults to 1.0 (at least one channel generated candidate).",
        threshold="Integer in [1, 8]"
    ),
    FeatureSpec(
        name="blocking_priority_score",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="Sum of priority scores across all firing channels during candidate generation.",
        missing_value_behavior="Defaults to 0.0.",
        threshold="Non-negative integer"
    ),
    FeatureSpec(
        name="blocking_rank_order",
        group="Group F: Blocking Provenance",
        dtype="float32",
        description="Deterministic rank order of candidate within S1 candidate set (row_number by priority).",
        missing_value_behavior="Defaults to 1.0.",
        threshold="Integer in [1, 100]"
    ),
]

# Canonical ordered list of feature names
FEATURE_NAMES: List[str] = [spec.name for spec in FEATURE_SPECS]

# Quick lookup map by name
FEATURE_SPEC_MAP: Dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}

# Group counts verification
FEATURE_GROUPS: Dict[str, List[str]] = {}
for spec in FEATURE_SPECS:
    FEATURE_GROUPS.setdefault(spec.group, []).append(spec.name)

def get_feature_count() -> int:
    """Returns the exact total feature count (65)."""
    return len(FEATURE_SPECS)

def get_feature_names() -> List[str]:
    """Returns the ordered feature name list."""
    return list(FEATURE_NAMES)
