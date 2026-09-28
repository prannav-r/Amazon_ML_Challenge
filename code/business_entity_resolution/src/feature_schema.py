"""
Amazon ML Challenge 2026: Business Entity Resolution
Production Feature Schema Definition (Config D1: 76 Features)
"""

from typing import List

CANONICAL_65_FEATURES: List[str] = ['country_exact_match', 'country_missing_either', 'name_raw_exact', 'name_legal_stripped_exact', 'name_domain_stripped_exact', 'name_alphanumeric_exact', 'name_levenshtein_sim', 'name_jaro_winkler_sim', 'name_char_3gram_jaccard', 'name_length_diff', 'name_length_ratio', 'name_token_jaccard', 'name_token_overlap_coef', 'name_token_containment_s1', 'name_token_containment_cand', 'name_shared_token_count', 'name_token_count_diff', 'name_first_token_match', 'name_last_token_match', 'legal_suffix_agreement', 'legal_suffix_mismatch', 'business_word_overlap_count', 'name_both_latin', 'name_cross_script', 'name_has_non_ascii', 'name_missing_either', 'address_raw_exact', 'address_token_jaccard', 'address_token_overlap_coef', 'address_shared_token_count', 'address_token_containment', 'address_token_count_diff', 'address_first_token_match', 'address_last_token_match', 'address_building_num_match', 'address_shared_numeric_count', 'address_numeric_token_jaccard', 'address_has_shared_numeric', 'address_postal_exact_match', 'address_postal_both_present', 'address_postal_mismatch', 'address_missing_s1', 'address_missing_cand', 'exact_name_and_address_number_match', 'high_name_sim_and_address_overlap', 'shared_name_and_shared_number', 'exact_postal_and_strong_name', 'cross_script_and_strong_address', 'weak_name_and_strong_address', 'name_missing_and_strong_address', 'candidate_is_source2', 'candidate_is_source3', 'cand_name_has_url', 'cand_name_has_noise_prefix', 'fired_chan_a', 'fired_chan_a2', 'fired_chan_b', 'fired_chan_c', 'fired_chan_d', 'fired_chan_e', 'fired_chan_e2', 'fired_chan_g', 'channels_fired_count', 'blocking_priority_score', 'blocking_rank_order']

SELECTED_PHASE8_FEATURES: List[str] = ['feat_name_token_jaccard_cand_addr_missing', 'feat_name_3gram_cand_addr_missing', 'feat_name_containment_cand_addr_missing', 'feat_name_lev_cand_addr_missing', 'feat_name_best_token_similarity', 'feat_name_distinctive_jaccard', 'feat_name_char_2gram_jaccard', 'feat_same_postal_weak_name', 'feat_distinctive_name_zero_overlap']

GROUP_K_DISAMBIGUATION_FEATURES: List[str] = ['feat_same_building_weak_name', 'feat_high_addr_low_name_penalty']

FEATURE_NAMES: List[str] = list(CANONICAL_65_FEATURES) + list(SELECTED_PHASE8_FEATURES) + list(GROUP_K_DISAMBIGUATION_FEATURES)

assert len(FEATURE_NAMES) == 76
assert len(set(FEATURE_NAMES)) == 76

def get_feature_names() -> List[str]:
    return list(FEATURE_NAMES)
