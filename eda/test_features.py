"""
Comprehensive Unit Tests & Schema Verification for Pairwise Feature Engineering
Amazon ML Challenge 2026: Business Entity Resolution

Tests:
1. Schema verification: exact 65 features, deterministic ordering, single source of truth.
2. The 9 explicit missing-value & edge-case scenarios:
   - Case 1: Both names missing
   - Case 2: One name missing
   - Case 3: Both addresses missing
   - Case 4: One address missing
   - Case 5: Both postal codes missing
   - Case 6: One postal code missing
   - Case 7: Empty token sets (symbols only)
   - Case 8: Cross-script names with strong matching addresses
   - Case 9: Cross-script names with unrelated addresses
3. Full 65-feature vector inspection on representative positive, native-script, typo, and hard-negative pairs.
4. Blocking provenance feature leakage audit.
"""

import sys
import os
import io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_schema import FEATURE_SPECS, FEATURE_NAMES, FEATURE_GROUPS, get_feature_count, get_feature_names
from src.features import PairwiseFeatureExtractor

extractor = PairwiseFeatureExtractor()

print("=" * 90)
print("COMPREHENSIVE PAIRWISE FEATURE ENGINEERING AUDIT & VERIFICATION")
print("=" * 90)

# ==============================================================================
# 1. AUTHORITATIVE SCHEMA & ORDERING VERIFICATION
# ==============================================================================
print("\n[1/4] Auditing Feature Schema & Reproducible Ordering...")

total_schema_features = get_feature_count()
extractor_features = extractor.get_feature_names()

print(f"  Total Canonical Features : {total_schema_features}")
print(f"  Extractor Feature Count  : {len(extractor_features)}")

assert total_schema_features == 65, f"Expected 65 features, got {total_schema_features}"
assert len(extractor_features) == 65, f"Expected 65 extractor features, got {len(extractor_features)}"
assert extractor_features == FEATURE_NAMES, "Feature ordering in extractor does not match canonical schema!"

print("  Feature Group Breakdown:")
for grp, feats in FEATURE_GROUPS.items():
    print(f"    - {grp:34s} : {len(feats):2d} features")

total_grouped = sum(len(f) for f in FEATURE_GROUPS.values())
assert total_grouped == 65, f"Group sum ({total_grouped}) != 65!"
print(f"  [PASSED] Schema verified: Exactly 65 features across 6 groups with 100% deterministic ordering.")

# ==============================================================================
# 2. AUDIT THE 9 EXPLICIT MISSING-VALUE & EDGE-CASE SCENARIOS
# ==============================================================================
print("\n[2/4] Testing 9 Explicit Missing-Value & Edge-Case Scenarios...")

edge_s1 = {
    "S1-M1": {"country": "US", "business_name": "", "business_address": "100 Main St, Austin, TX"},              # Both names missing
    "S1-M2": {"country": "US", "business_name": "", "business_address": "100 Main St, Austin, TX"},              # One name missing
    "S1-M3": {"country": "US", "business_name": "Apex Corp", "business_address": ""},                            # Both addresses missing
    "S1-M4": {"country": "US", "business_name": "Apex Corp", "business_address": ""},                            # One address missing
    "S1-M5": {"country": "US", "business_name": "Apex Corp", "business_address": "Main Street, Austin, Texas"},   # Both postal missing
    "S1-M6": {"country": "US", "business_name": "Apex Corp", "business_address": "Main Street, Austin 78701"},   # One postal missing
    "S1-M7": {"country": "US", "business_name": "!@#$%^", "business_address": "---"},                            # Empty token sets
    "S1-M8": {"country": "India", "business_name": "Apex Products Pvt Ltd", "business_address": "7 Snehlata Ganj, Indore 452003"}, # Cross-script strong addr
    "S1-M9": {"country": "India", "business_name": "Apex Products Pvt Ltd", "business_address": "7 Snehlata Ganj, Indore 452003"}  # Cross-script unrelated addr
}

edge_cands = {
    "S2-M1": {"country": "US", "business_name": "", "business_address": "100 Main St, Austin, TX"},              # Both names missing
    "S2-M2": {"country": "US", "business_name": "Acme Corp", "business_address": "100 Main St, Austin, TX"},     # One name missing
    "S2-M3": {"country": "US", "business_name": "Apex Corp", "business_address": ""},                            # Both addresses missing
    "S2-M4": {"country": "US", "business_name": "Apex Corp", "business_address": "100 Main St, Austin, TX"},     # One address missing
    "S2-M5": {"country": "US", "business_name": "Apex Corp", "business_address": "Main St, Austin, TX"},         # Both postal missing
    "S2-M6": {"country": "US", "business_name": "Apex Corp", "business_address": "Main St, Austin, Texas"},     # One postal missing
    "S2-M7": {"country": "US", "business_name": "&*()_+", "business_address": "==="},                            # Empty token sets
    "S3-M8": {"country": "India", "business_name": "एपेक्स प्रोडक्ट्स प्राइवेट लिमिटेड", "business_address": "7 Snehlata Ganj, Indore 452003"}, # Cross-script match
    "S3-M9": {"country": "India", "business_name": "एपेक्स प्रोडक्ट्स प्राइवेट लिमिटेड", "business_address": "500 MG Road, Bangalore 560001"}   # Cross-script unrelated
}

edge_pairs_df = pd.DataFrame([
    {"source1_entity_id": f"S1-M{i}", "candidate_entity_id": f"S{'3' if i in (8,9) else '2'}-M{i}",
     "total_priority": 100, "channels_fired": 1, "rank_order": 1,
     "fired_chan_a": 1, "fired_chan_a2": 0, "fired_chan_b": 0, "fired_chan_c": 0,
     "fired_chan_d": 0, "fired_chan_e": 0, "fired_chan_e2": 0, "fired_chan_g": 0}
    for i in range(1, 10)
])

df_edge_feats = extractor.extract_features(edge_pairs_df, edge_s1, edge_cands)

# Assertion 1: Both names missing -> similarity MUST be 0.0, name_missing_either MUST be 1.0
assert df_edge_feats.loc[0, "name_raw_exact"] == 0.0, "Both names missing gave name_raw_exact == 1.0!"
assert df_edge_feats.loc[0, "name_levenshtein_sim"] == 0.0, "Both names missing gave non-zero Levenshtein!"
assert df_edge_feats.loc[0, "name_missing_either"] == 1.0, "name_missing_either was not flagged!"
print("  [Case 1 PASSED] Both names missing: similarities evaluate strictly to 0.0, missing flag is 1.0.")

# Assertion 2: One name missing -> similarity MUST be 0.0
assert df_edge_feats.loc[1, "name_raw_exact"] == 0.0, "One name missing gave name_raw_exact == 1.0!"
assert df_edge_feats.loc[1, "name_levenshtein_sim"] == 0.0, "One name missing gave non-zero Levenshtein!"
assert df_edge_feats.loc[1, "name_missing_either"] == 1.0, "One name missing flag not set!"
print("  [Case 2 PASSED] One name missing: similarities evaluate strictly to 0.0, missing flag is 1.0.")

# Assertion 3: Both addresses missing -> address exact MUST be 0.0, token jaccard MUST be 0.0
assert df_edge_feats.loc[2, "address_raw_exact"] == 0.0, "Both addresses missing gave address_raw_exact == 1.0!"
assert df_edge_feats.loc[2, "address_token_jaccard"] == 0.0, "Both addresses missing gave non-zero Jaccard!"
assert df_edge_feats.loc[2, "address_missing_s1"] == 1.0 and df_edge_feats.loc[2, "address_missing_cand"] == 1.0
print("  [Case 3 PASSED] Both addresses missing: address similarities strictly 0.0, both missing flags 1.0.")

# Assertion 4: One address missing -> similarities MUST be 0.0
assert df_edge_feats.loc[3, "address_raw_exact"] == 0.0, "One address missing gave address_raw_exact == 1.0!"
assert df_edge_feats.loc[3, "address_token_jaccard"] == 0.0, "One address missing gave non-zero Jaccard!"
assert df_edge_feats.loc[3, "address_missing_s1"] == 1.0 and df_edge_feats.loc[3, "address_missing_cand"] == 0.0
print("  [Case 4 PASSED] One address missing: address similarities strictly 0.0, correct missing flags.")

# Assertion 5: Both postal codes missing -> postal_exact MUST be 0.0, both_present MUST be 0.0
assert df_edge_feats.loc[4, "address_postal_exact_match"] == 0.0, "Missing postals gave postal_exact == 1.0!"
assert df_edge_feats.loc[4, "address_postal_both_present"] == 0.0, "postal_both_present should be 0.0!"
print("  [Case 5 PASSED] Both postal codes missing: postal match strictly 0.0, both_present is 0.0.")

# Assertion 6: One postal code missing -> postal_exact MUST be 0.0, mismatch MUST be 0.0
assert df_edge_feats.loc[5, "address_postal_exact_match"] == 0.0, "One postal missing gave postal_exact == 1.0!"
assert df_edge_feats.loc[5, "address_postal_both_present"] == 0.0, "One postal missing gave both_present == 1.0!"
assert df_edge_feats.loc[5, "address_postal_mismatch"] == 0.0, "One postal missing flagged as mismatch!"
print("  [Case 6 PASSED] One postal code missing: postal match strictly 0.0, no artificial mismatch.")

# Assertion 7: Empty token sets (symbols only) -> similarities MUST be 0.0
assert df_edge_feats.loc[6, "name_token_jaccard"] == 0.0, "Symbol-only names gave non-zero token jaccard!"
assert df_edge_feats.loc[6, "address_token_jaccard"] == 0.0, "Symbol-only addresses gave non-zero token jaccard!"
print("  [Case 7 PASSED] Empty token sets: all token similarities evaluate cleanly to 0.0.")

# Assertion 8: Cross-script with strong address -> cross_script == 1.0 AND strong interaction == 1.0
assert df_edge_feats.loc[7, "name_cross_script"] == 1.0, "Cross-script pair not flagged!"
assert df_edge_feats.loc[7, "address_building_num_match"] == 1.0, "Address building number 7 match failed!"
assert df_edge_feats.loc[7, "cross_script_and_strong_address"] == 1.0, "cross_script_and_strong_address interaction failed!"
print("  [Case 8 PASSED] Cross-script with strong address: correctly triggers cross_script_and_strong_address == 1.0.")

# Assertion 9: Cross-script with unrelated address -> cross_script == 1.0 BUT strong interaction MUST be 0.0
assert df_edge_feats.loc[8, "name_cross_script"] == 1.0, "Cross-script pair not flagged!"
assert df_edge_feats.loc[8, "address_building_num_match"] == 0.0, "Unrelated addresses gave building match == 1.0!"
assert df_edge_feats.loc[8, "cross_script_and_strong_address"] == 0.0, "cross_script_and_strong_address fired on unrelated address!"
print("  [Case 9 PASSED] Cross-script with unrelated address: strong address interaction evaluates strictly to 0.0.")

# ==============================================================================
# 3. COMPLETE 65-FEATURE VECTOR PRINT FOR REPRESENTATIVE EXAMPLES
# ==============================================================================
print("\n[3/4] Printing Full 65-Feature Vector for Representative Pairs...")

rep_s1 = {
    "S1-EXACT": {"country": "US", "business_name": "Acme Industrial Technologies Inc.", "business_address": "104 Main Street, Suite 200, Austin, TX 78701"},
    "S1-NATIVE": {"country": "India", "business_name": "Apex Products Private Limited", "business_address": "7 Snehlata Ganj, Indore, Madhya Pradesh 452003"},
    "S1-TYPO": {"country": "US", "business_name": "Willow Co", "business_address": "1046 24th Street, Brooklyn, NY 11210"},
    "S1-FRANCE": {"country": "France", "business_name": "Dubois & Fils SARL", "business_address": "12 Boulevard Haussmann, Paris 75008"},
    "S1-NEG": {"country": "US", "business_name": "Acme Industrial Technologies Inc.", "business_address": "104 Main Street, Suite 200, Austin, TX 78701"}
}

rep_cands = {
    "S2-EXACT": {"country": "US", "business_name": "Acme Industrial Technologies", "business_address": "104 Main St, Ste 200, Austin, Texas 78701"},
    "S3-NATIVE": {"country": "India", "business_name": "एपेक्स प्रोडक्ट्स प्राइवेट लिमिटेड", "business_address": "7 Snehlata Ganj, Indore, MP 452003"},
    "S2-TYPO": {"country": "US", "business_name": "Wsillw Co", "business_address": "1046 24th St, Brooklyn, New York 11210"},
    "S3-FRANCE": {"country": "France", "business_name": "SARL Dubois et Fils", "business_address": "12 Bd Haussmann, 75008 Paris"},
    "S2-NEG": {"country": "US", "business_name": "Texas Solar Roofing LLC", "business_address": "500 Congress Avenue, Austin, TX 78701"}
}

rep_pairs_df = pd.DataFrame([
    {"source1_entity_id": "S1-EXACT", "candidate_entity_id": "S2-EXACT", "total_priority": 250, "channels_fired": 4, "rank_order": 1,
     "fired_chan_a": 1, "fired_chan_a2": 1, "fired_chan_b": 1, "fired_chan_c": 1, "fired_chan_d": 0, "fired_chan_e": 0, "fired_chan_e2": 0, "fired_chan_g": 0},
    
    {"source1_entity_id": "S1-NATIVE", "candidate_entity_id": "S3-NATIVE", "total_priority": 85, "channels_fired": 1, "rank_order": 1,
     "fired_chan_a": 0, "fired_chan_a2": 0, "fired_chan_b": 0, "fired_chan_c": 0, "fired_chan_d": 0, "fired_chan_e": 0, "fired_chan_e2": 1, "fired_chan_g": 0},

    {"source1_entity_id": "S1-TYPO", "candidate_entity_id": "S2-TYPO", "total_priority": 125, "channels_fired": 2, "rank_order": 1,
     "fired_chan_a": 0, "fired_chan_a2": 0, "fired_chan_b": 0, "fired_chan_c": 0, "fired_chan_d": 0, "fired_chan_e": 1, "fired_chan_e2": 0, "fired_chan_g": 1},

    {"source1_entity_id": "S1-FRANCE", "candidate_entity_id": "S3-FRANCE", "total_priority": 180, "channels_fired": 3, "rank_order": 1,
     "fired_chan_a": 1, "fired_chan_a2": 1, "fired_chan_b": 1, "fired_chan_c": 0, "fired_chan_d": 0, "fired_chan_e": 0, "fired_chan_e2": 0, "fired_chan_g": 0},

    {"source1_entity_id": "S1-NEG", "candidate_entity_id": "S2-NEG", "total_priority": 50, "channels_fired": 1, "rank_order": 35,
     "fired_chan_a": 0, "fired_chan_a2": 0, "fired_chan_b": 0, "fired_chan_c": 0, "fired_chan_d": 1, "fired_chan_e": 0, "fired_chan_e2": 0, "fired_chan_g": 0}
])

df_rep_feats = extractor.extract_features(rep_pairs_df, rep_s1, rep_cands)

# Print complete feature table
print(f"\n{'#':2s} | {'Feature Name':36s} | {'Exact (US)':11s} | {'Native (IN)':11s} | {'Typo (US)':11s} | {'France (FR)':11s} | {'Hard Neg (US)':13s}")
print("-" * 98)
for i, feat in enumerate(FEATURE_NAMES, 1):
    val_exact = df_rep_feats.loc[0, feat]
    val_native = df_rep_feats.loc[1, feat]
    val_typo = df_rep_feats.loc[2, feat]
    val_france = df_rep_feats.loc[3, feat]
    val_neg = df_rep_feats.loc[4, feat]
    
    # Format cleanly
    def fmt(v):
        return f"{v:10.4f}" if isinstance(v, (float, np.floating)) else f"{v:10}"
    print(f"{i:2d} | {feat:36s} | {fmt(val_exact)} | {fmt(val_native)} | {fmt(val_typo)} | {fmt(val_france)} | {fmt(val_neg)}")

# ==============================================================================
# 4. BLOCKING PROVENANCE & LEAKAGE AUDIT
# ==============================================================================
print("\n[4/4] Verifying Blocking Provenance Features & Leakage Audit...")

prov_features = [
    "blocking_rank_order", "blocking_priority_score", "channels_fired_count",
    "fired_chan_a", "fired_chan_a2", "fired_chan_b", "fired_chan_c",
    "fired_chan_d", "fired_chan_e", "fired_chan_e2", "fired_chan_g"
]

for pf in prov_features:
    spec = next(s for s in FEATURE_SPECS if s.name == pf)
    assert spec.available_at_test == True, f"{pf} not available at test!"
    assert spec.depends_on_ground_truth == False, f"{pf} depends on ground truth!"

print(f"  All {len(prov_features)} blocking provenance features verified as available at test time with zero label dependency.")

print("\n" + "=" * 90)
print("ALL COMPREHENSIVE TESTS & AUDITS COMPLETED SUCCESSFULLY")
print("=" * 90)
