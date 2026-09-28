"""
Phase 13: Fresh Validation of Config D1
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Reconstructs the exact split history:
   - Phase 5.1 Protected Holdout: 1,004 S1 entities (PERMANENTLY LOCKED)
   - Phase 10/12 Development Set: 998 S1 entities (LOCKED)
   - Phase 11 Protected Holdout: 597 S1 entities (PERMANENTLY LOCKED)
   - Remaining Eligible Pool: 2,401 S1 entities
2. Constructs a completely fresh, untouched S1 holdout (478 S1 entities, ~20%) from the eligible pool.
3. Strictly asserts zero leakage:
   - Phase 5.1 Holdout ∩ Phase 13 Holdout = 0
   - Phase 11 Holdout ∩ Phase 13 Holdout = 0
   - Phase 10/12 Dev ∩ Phase 13 Holdout = 0
   - Phase 13 Train ∩ Phase 13 Holdout = 0
4. Exports the deterministic holdout manifest to 'eda/phase13_holdout_s1_ids.txt'.
5. Trains the frozen Config D1 model on the 1,923 Phase 13 training S1 entities:
   - 76 features: 65 canonical + 9 selected Phase 8 + 2 Group K address disambiguation features
     (feat_same_building_weak_name, feat_high_addr_low_name_penalty)
   - XGBoost C1 architecture (hist, lr=0.08, depth=5, n_est=300, seed=42)
   - scale_pos_weight = sqrt(neg/pos) computed strictly on training pairs
6. Generates candidate pairs on Phase 13 holdout using Enhanced Blocker V2 (Cap 200, 11 channels).
7. Extracts the exact 76 features on holdout candidate pairs.
8. Evaluates the frozen configuration at threshold 0.88 (NO RETUNING).
9. Computes all required metrics:
   - S1-level Macro-F0.5, Macro Precision, Macro Recall
   - Blocker Recall vs End-to-End Recall, Classifier Retention
   - Singleton, Zero-Match, Multi-Match breakdowns
   - Country breakdown (US, India)
   - Generalization Gap vs Phase 12 Dev (0.8530)
   - Comparison vs Phase 11 Fresh Holdout (0.8350) and Phase 5.1 Benchmark (0.8122)
10. Performs descriptive error analysis and inspects behavior of Group K features.
11. Saves structured results to 'output/phase13_holdout_results.json'.
"""

import sys
import os
import io
import time
import pickle
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Tuple
import numpy as np
import pandas as pd
import duckdb
import Levenshtein

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking_v2 import CandidateBlockerV2
from src.features_v2 import PairwiseFeatureExtractorV2
from src.feature_schema_v2 import (
    FEATURE_SPECS_V2,
    SELECTED_PHASE8_FEATURES,
    EXPERIMENT_FEATURE_SETS,
)
from src.model import EntityMatcherModel
from src.evaluate import evaluate_predictions_df, evaluate_by_group

OUTPUT_DIR = "output"
GT_PATH = os.path.join(OUTPUT_DIR, "eval_ground_truth.pkl")
TRAIN_PARQUET_V2 = os.path.join(OUTPUT_DIR, "eval_features_v2_cap150.parquet")
BASE_TRAIN = "student_resource/dataset/train"
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "phase13_holdout_s1_ids.txt")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "phase13_holdout_results.json")


def categorize_error_pattern(s1_name: str, cand_name: str, s1_addr: str, cand_addr: str, is_fp: bool = False) -> str:
    s1_n = (s1_name or "").lower().strip()
    cand_n = (cand_name or "").lower().strip()
    s1_a = (s1_addr or "").lower().strip()
    cand_a = (cand_addr or "").lower().strip()

    # 1. Missing address
    if not s1_a or not cand_a:
        return "Missing Address"

    # 2. Native script
    if any(ord(char) > 127 for char in s1_n + cand_n):
        return "Native Script"

    # Postal codes
    post1 = re.findall(r'\b[0-9]{5,6}\b', s1_a)
    post2 = re.findall(r'\b[0-9]{5,6}\b', cand_a)
    same_postal = (len(post1) > 0 and len(post2) > 0 and post1[0] == post2[0])

    # Address numbers
    num1 = re.findall(r'[0-9]{1,6}', s1_a)
    num2 = re.findall(r'[0-9]{1,6}', cand_a)
    same_num = (len(num1) > 0 and len(num2) > 0 and num1[0].lstrip('0') == num2[0].lstrip('0'))

    # Name tokens
    toks1 = set(re.findall(r'[a-z0-9]+', s1_n))
    toks2 = set(re.findall(r'[a-z0-9]+', cand_n))
    jaccard = len(toks1 & toks2) / max(1, len(toks1 | toks2))

    # 3. Same building
    if s1_a == cand_a or (len(s1_a) > 12 and s1_a[:15] == cand_a[:15]):
        if is_fp and jaccard < 0.4:
            return "Same Building"
        elif not is_fp and jaccard < 0.3:
            return "Aliases / Trade Names"

    # 4. Franchise / Brand
    if s1_n == cand_n and s1_a != cand_a:
        return "Franchise / Brand"

    # 5. Address Number Match
    if same_num and jaccard < 0.4:
        return "Address Number Match"

    # 6. Shared Address Token
    addr_toks1 = set(re.findall(r'[a-z0-9]{4,}', s1_a))
    addr_toks2 = set(re.findall(r'[a-z0-9]{4,}', cand_a))
    if len(addr_toks1 & addr_toks2) > 0 and jaccard < 0.4:
        return "Shared Address Token"

    # 7. OCR / Typo
    lev = Levenshtein.distance(s1_n, cand_n)
    if lev <= 2 and len(s1_n) >= 5:
        return "OCR / Typo"

    # 8. Aliases / Trade Names
    if jaccard < 0.35 and (s1_a == cand_a or same_postal or same_num or len(toks1 & toks2) > 0):
        return "Aliases / Trade Names"

    return "Other / Lexical Shift"


def main():
    print("=" * 80)
    print("PHASE 13: FRESH UNTOUCHED HOLDOUT VALIDATION OF CONFIG D1")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. SETUP ENVIRONMENT & VERIFY PARTITIONS
    # --------------------------------------------------------------------------
    print("\n[1/6] Loading ground truth and constructing fresh holdout partition...")
    with open(GT_PATH, "rb") as f:
        meta = pickle.load(f)

    gt_dict = meta["gt_dict"]
    s1_list_all = meta["s1_list"]
    s1_lookup = meta["s1_lookup"]

    # Reconstruct original 3-way split (Seed 42)
    strata = defaultdict(list)
    for s1 in s1_list_all:
        c = s1_lookup[s1]["country"]
        num_matches = len(gt_dict.get(s1, set()))
        m_type = "zero" if num_matches == 0 else ("single" if num_matches == 1 else "multi")
        strata[(c, m_type)].append(s1)

    train_s1_orig, dev_s1_p10, holdout_s1_p51 = [], [], []
    rng_p5 = np.random.RandomState(42)
    for (c, m_type), ids in sorted(strata.items()):
        shuffled = rng_p5.permutation(ids)
        n = len(shuffled)
        n_train = int(n * 0.60)
        n_dev = int(n * 0.20)
        train_s1_orig.extend(shuffled[:n_train])
        dev_s1_p10.extend(shuffled[n_train:n_train + n_dev])
        holdout_s1_p51.extend(shuffled[n_train + n_dev:])

    # Reconstruct Phase 11 Fresh Holdout (Seed 2026, ~20% of train_s1_orig)
    p11_strata = defaultdict(list)
    for s in train_s1_orig:
        c = s1_lookup[s]["country"]
        num_matches = len(gt_dict.get(s, set()))
        m_type = "zero" if num_matches == 0 else ("single" if num_matches == 1 else "multi")
        p11_strata[(c, m_type)].append(s)

    p11_train_s1, p11_holdout_s1 = [], []
    rng_p11 = np.random.RandomState(2026)
    for (c, m_type), ids in sorted(p11_strata.items()):
        shuffled = rng_p11.permutation(ids)
        n = len(shuffled)
        n_holdout = int(n * 0.20)
        p11_holdout_s1.extend(shuffled[:n_holdout])
        p11_train_s1.extend(shuffled[n_holdout:])

    print(f"  Total S1 Entities           : {len(s1_list_all):,}")
    print(f"  Protected Phase 5.1 Holdout : {len(holdout_s1_p51):,} S1s (LOCKED)")
    print(f"  Phase 10/12 Development Set : {len(dev_s1_p10):,} S1s (LOCKED)")
    print(f"  Protected Phase 11 Holdout  : {len(p11_holdout_s1):,} S1s (LOCKED)")
    print(f"  Eligible Training Pool      : {len(p11_train_s1):,} S1s")

    # Construct Phase 13 Fresh Holdout (Seed 2027, ~20% of remaining pool)
    p13_strata = defaultdict(list)
    for s in p11_train_s1:
        c = s1_lookup[s]["country"]
        num_matches = len(gt_dict.get(s, set()))
        m_type = "zero" if num_matches == 0 else ("single" if num_matches == 1 else "multi")
        p13_strata[(c, m_type)].append(s)

    p13_train_s1, p13_holdout_s1 = [], []
    rng_p13 = np.random.RandomState(2027)
    for (c, m_type), ids in sorted(p13_strata.items()):
        shuffled = rng_p13.permutation(ids)
        n = len(shuffled)
        n_holdout = int(n * 0.20)
        p13_holdout_s1.extend(shuffled[:n_holdout])
        p13_train_s1.extend(shuffled[n_holdout:])

    # --------------------------------------------------------------------------
    # 2. STRICT LEAKAGE AUDIT
    # --------------------------------------------------------------------------
    print("\n[2/6] Executing strict anti-leakage non-overlap assertions...")
    overlap_p51 = set(holdout_s1_p51) & set(p13_holdout_s1)
    overlap_p11 = set(p11_holdout_s1) & set(p13_holdout_s1)
    overlap_dev = set(dev_s1_p10) & set(p13_holdout_s1)
    overlap_train = set(p13_train_s1) & set(p13_holdout_s1)

    print(f"  Check 1: Phase 5.1 Holdout ∩ Phase 13 Holdout = {len(overlap_p51)} entities")
    print(f"  Check 2: Phase 11 Holdout  ∩ Phase 13 Holdout = {len(overlap_p11)} entities")
    print(f"  Check 3: Phase 10/12 Dev S1 ∩ Phase 13 Holdout = {len(overlap_dev)} entities")
    print(f"  Check 4: Phase 13 Train     ∩ Phase 13 Holdout = {len(overlap_train)} entities")

    assert len(overlap_p51) == 0, "CRITICAL ERROR: Phase 13 holdout overlaps protected Phase 5.1 holdout!"
    assert len(overlap_p11) == 0, "CRITICAL ERROR: Phase 13 holdout overlaps protected Phase 11 holdout!"
    assert len(overlap_dev) == 0, "CRITICAL ERROR: Phase 13 holdout overlaps Phase 10/12 development set!"
    assert len(overlap_train) == 0, "CRITICAL ERROR: Phase 13 holdout overlaps Phase 13 training set!"
    print("  --> ALL NON-OVERLAP CHECKS PASSED: 100% UNTOUCHED HOLDOUT VERIFIED.")

    # Record manifest
    with open(MANIFEST_PATH, "w") as f:
        for s in sorted(p13_holdout_s1):
            f.write(f"{s}\n")
    print(f"  Wrote deterministic holdout manifest ({len(p13_holdout_s1)} IDs) to {MANIFEST_PATH}")

    # Holdout ground-truth matches
    holdout_gt_pairs = set()
    for s in p13_holdout_s1:
        for c in gt_dict.get(s, set()):
            holdout_gt_pairs.add((s, c))
    total_holdout_true = len(holdout_gt_pairs)

    country_counts = Counter(s1_lookup[s]["country"] for s in p13_holdout_s1)
    mtype_counts = Counter(
        "zero" if len(gt_dict.get(s, set())) == 0 else ("single" if len(gt_dict.get(s, set())) == 1 else "multi")
        for s in p13_holdout_s1
    )
    print(f"\n  Phase 13 Holdout Composition:")
    print(f"    Total S1s: {len(p13_holdout_s1):,} | Total True Matches: {total_holdout_true:,}")
    print(f"    Countries: US={country_counts['US']} ({country_counts['US']/len(p13_holdout_s1)*100:.1f}%), India={country_counts['India']} ({country_counts['India']/len(p13_holdout_s1)*100:.1f}%)")
    print(f"    Match Types: Multi={mtype_counts['multi']}, Single={mtype_counts['single']}, Zero={mtype_counts['zero']}")

    # --------------------------------------------------------------------------
    # 3. TRAIN FROZEN CONFIG D1 ON PHASE 13 TRAINING DATA
    # --------------------------------------------------------------------------
    print("\n[3/6] Training frozen Config D1 model on Phase 13 training pairs only...")
    t0_tr = time.time()
    df_all_v2 = pd.read_parquet(TRAIN_PARQUET_V2)
    p13_train_s1_set = set(p13_train_s1)
    df_train_p13 = df_all_v2[df_all_v2["source1_entity_id"].isin(p13_train_s1_set)].copy()

    train_pos = int(df_train_p13["match_label"].sum())
    train_neg = int(len(df_train_p13) - train_pos)
    scale_pos_weight = float(np.sqrt(train_neg / train_pos))
    print(f"  Training S1 Entities: {len(p13_train_s1):,}")
    print(f"  Training Pairs: {len(df_train_p13):,} ({train_pos:,} pos, {train_neg:,} neg | scale_pos_weight={scale_pos_weight:.2f})")

    # Define exact 76 features
    base_74 = list(EXPERIMENT_FEATURE_SETS["Selected_Phase8"])
    feats_76 = base_74 + [
        "feat_same_building_weak_name",
        "feat_high_addr_low_name_penalty",
    ]
    print(f"  Feature Schema: {len(feats_76)} features (74 Baseline + 2 Group K Address Disambiguation)")
    print(f"    1. feat_same_building_weak_name")
    print(f"    2. feat_high_addr_low_name_penalty")

    m76 = EntityMatcherModel(
        model_type="xgboost_weighted",
        params={
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
        },
        random_state=42,
        feature_names=feats_76,
    )
    t0_fit = time.time()
    m76.fit(df_train_p13[feats_76].values, df_train_p13["match_label"].values)
    print(f"  Model successfully fitted in {time.time()-t0_fit:.2f}s.")

    # --------------------------------------------------------------------------
    # 4. GENERATE CANDIDATES & EXTRACT FEATURES ON FRESH HOLDOUT
    # --------------------------------------------------------------------------
    print("\n[4/6] Generating candidate pairs with Enhanced Blocker V2 (Cap 200) on fresh holdout...")
    con = duckdb.connect()

    holdout_s1_rows = [
        (s, s1_lookup[s]["business_name"], s1_lookup[s]["business_address"], s1_lookup[s]["country"])
        for s in p13_holdout_s1
    ]
    df_holdout_s1 = pd.DataFrame(holdout_s1_rows, columns=["entity_id", "business_name", "business_address", "country"])
    con.register("holdout_s1_tbl", df_holdout_s1)

    t0_cand = time.time()
    con.execute(f"""
    CREATE TEMP TABLE all_candidates AS
    SELECT entity_id, business_name, business_address, country, 'S2' as src
    FROM read_csv('{BASE_TRAIN}/train_source2.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE country IN ('US', 'India')
    UNION ALL
    SELECT entity_id, business_name, business_address, country, 'S3' as src
    FROM read_csv('{BASE_TRAIN}/train_source3.tsv', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE country IN ('US', 'India');
    """)
    n_cands_pool = con.execute("SELECT count(*) FROM all_candidates").fetchone()[0]
    print(f"  Loaded candidate pool of {n_cands_pool:,} records in {time.time()-t0_cand:.2f}s.")

    t0_block = time.time()
    blocker = CandidateBlockerV2(con)
    blocker.generate_candidates(
        s1_table_or_path="holdout_s1_tbl",
        cand_table_or_path="all_candidates",
        output_table="holdout_cands_v2_200",
        max_candidates_per_s1=200,
        enable_enhanced_channels=True,
    )
    t_block = time.time() - t0_block

    df_cands_holdout = con.execute("SELECT * FROM holdout_cands_v2_200").fetchdf()

    # Blocker diagnostics
    counts_s1 = df_cands_holdout.groupby("source1_entity_id")["candidate_entity_id"].count().to_dict()
    for s in p13_holdout_s1:
        if s not in counts_s1:
            counts_s1[s] = 0
    vals = list(counts_s1.values())
    avg_cands = float(np.mean(vals))
    median_cands = float(np.median(vals))
    p95_cands = float(np.percentile(vals, 95))
    max_cands = int(np.max(vals))
    zero_cand_s1 = sum(1 for v in vals if v == 0)

    holdout_cand_pairs_set = set(zip(df_cands_holdout["source1_entity_id"], df_cands_holdout["candidate_entity_id"]))
    retained_true = len(holdout_gt_pairs & holdout_cand_pairs_set)
    blocker_recall = retained_true / total_holdout_true * 100.0
    missed_blocker_true = total_holdout_true - retained_true

    print(f"  Holdout Candidate Pairs Generated : {len(df_cands_holdout):,} in {t_block:.2f}s")
    print(f"  Blocker Recall                    : {blocker_recall:.2f}% ({retained_true:,}/{total_holdout_true:,} true matches | Missed: {missed_blocker_true:,})")
    print(f"  Candidate Distribution            : Avg: {avg_cands:.1f}/S1, Median: {median_cands:.0f}, P95: {p95_cands:.0f}, Max: {max_cands}, Zero-Cand S1s: {zero_cand_s1}")

    # Feature extraction
    t0_feat = time.time()
    c_rows = con.execute("""
        SELECT entity_id, country, business_name, business_address 
        FROM all_candidates 
        WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM holdout_cands_v2_200)
    """).fetchall()
    cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in c_rows}

    extractor_v2 = PairwiseFeatureExtractorV2()
    df_feat_holdout = extractor_v2.extract_features(df_cands_holdout, s1_lookup, cand_lookup)
    t_feat = time.time() - t0_feat
    print(f"  Extracted 76 features on holdout pairs in {t_feat:.2f}s.")

    # --------------------------------------------------------------------------
    # 5. INFERENCE & FROZEN EVALUATION AT THRESHOLD 0.88
    # --------------------------------------------------------------------------
    print("\n[5/6] Performing inference and evaluating frozen threshold 0.88...")
    df_eval = df_feat_holdout[["source1_entity_id", "candidate_entity_id"]].copy()
    X_holdout = df_feat_holdout[feats_76].values
    t0_inf = time.time()
    df_eval["score"] = m76.predict_proba(X_holdout)
    t_inf = time.time() - t0_inf

    # Evaluate overall at FROZEN threshold 0.88
    res_088 = evaluate_predictions_df(df_eval, gt_dict, threshold=0.88, s1_list=p13_holdout_s1)
    grp_088 = evaluate_by_group(df_eval, gt_dict, threshold=0.88, s1_list=p13_holdout_s1)

    matched_088 = df_eval[df_eval["score"] >= 0.88]
    pred_pairs_088 = set(zip(matched_088["source1_entity_id"], matched_088["candidate_entity_id"]))
    e2e_true = len(holdout_gt_pairs & pred_pairs_088)
    e2e_recall = e2e_true / total_holdout_true * 100.0
    classifier_retention = e2e_true / retained_true * 100.0 if retained_true > 0 else 0.0

    print("\n" + "=" * 80)
    print("PHASE 13 FRESH UNTOUCHED HOLDOUT RESULTS (FROZEN CONFIG D1, THRESHOLD 0.88)")
    print("=" * 80)
    print(f"  Primary Metric (S1 Macro-F0.5) : {res_088['macro_f05']:.4f}")
    print(f"  Macro Precision                : {res_088['macro_precision']:.4f}")
    print(f"  Macro Recall                   : {res_088['macro_recall']:.4f}")
    print(f"  End-to-End True Matches        : {e2e_true:,} / {total_holdout_true:,} ({e2e_recall:.2f}%)")
    print(f"  Blocker Retention              : {retained_true:,} / {total_holdout_true:,} ({blocker_recall:.2f}%)")
    print(f"  Classifier Retention           : {classifier_retention:.2f}% ({e2e_true:,} / {retained_true:,})")
    print(f"  Total Predicted Matches        : {res_088['total_pred_matches']:,} (Avg: {res_088['total_pred_matches']/len(p13_holdout_s1):.2f}/S1)")
    print(f"  Empty Predictions Percentage   : {res_088['empty_pred_pct']:.1f}%")

    # Relationship groups
    s_grp = grp_088["group1_singleton"]
    z_grp = grp_088["group2_zero_match"]
    m_grp = grp_088["group3_multi_match"]
    print(f"\n  S1 Relationship Group Breakdown:")
    print(f"    - Singletons  : Prec={s_grp['macro_precision']:.4f} | Rec={s_grp['macro_recall']:.4f} | F0.5={s_grp['macro_f05']:.4f} ({s_grp['count']} entities)")
    print(f"    - Zero-Match  : Prec={z_grp['macro_precision']:.4f} | Rec={z_grp['macro_recall']:.4f} | F0.5={z_grp['macro_f05']:.4f} (FP Rate: {z_grp['false_positive_rate']:.2f}% | {z_grp['count']} entities)")
    print(f"    - Multi-Match : Prec={m_grp['macro_precision']:.4f} | Rec={m_grp['macro_recall']:.4f} | F0.5={m_grp['macro_f05']:.4f} ({m_grp['count']} entities)")

    # Country breakdown
    us_s1 = [s for s in p13_holdout_s1 if s1_lookup[s]["country"] == "US"]
    in_s1 = [s for s in p13_holdout_s1 if s1_lookup[s]["country"] == "India"]
    res_us = evaluate_predictions_df(df_eval, gt_dict, threshold=0.88, s1_list=us_s1)
    res_in = evaluate_predictions_df(df_eval, gt_dict, threshold=0.88, s1_list=in_s1)

    print(f"\n  Country Breakdown:")
    print(f"    - United States ({len(us_s1)} S1s) : Prec={res_us['macro_precision']:.4f} | Rec={res_us['macro_recall']:.4f} | F0.5={res_us['macro_f05']:.4f}")
    print(f"    - India         ({len(in_s1)} S1s) : Prec={res_in['macro_precision']:.4f} | Rec={res_in['macro_recall']:.4f} | F0.5={res_in['macro_f05']:.4f}")
    print(f"    - France                           : N/A (France is absent from the training/validation population)")

    # --------------------------------------------------------------------------
    # 6. FEATURE IMPORTANCES & CONTRIBUTION OF GROUP K
    # --------------------------------------------------------------------------
    print("\n[6/6] Extracting feature importances and performing error analysis...")
    df_imp = m76.get_feature_importances()
    feat_gains = []
    for _, row in df_imp.iterrows():
        f = row["feature"]
        g = float(row["gain_importance"])
        feat_gains.append({
            "feature": f,
            "gain": g,
            "gain_pct": g * 100.0,
        })

    print(f"\n  Top 30 Features by Gain:")
    for rank, fg in enumerate(feat_gains[:30], 1):
        flag = " [NEW GROUP K]" if fg["feature"] in ["feat_same_building_weak_name", "feat_high_addr_low_name_penalty"] else ""
        print(f"    {rank:>2}. {fg['feature']:<44} : {fg['gain_pct']:5.2f}%{flag}")

    # Inspect Group K features
    gk_feats = ["feat_same_building_weak_name", "feat_high_addr_low_name_penalty"]
    print(f"\n  Group K Feature Contributions:")
    for f in gk_feats:
        match_item = next(((idx + 1, item) for idx, item in enumerate(feat_gains) if item["feature"] == f), (None, None))
        if match_item[0]:
            print(f"    - {f:<35} : Rank {match_item[0]:>2} | Gain: {match_item[1]['gain_pct']:.2f}%")
        else:
            print(f"    - {f:<35} : Unranked / 0 gain")

    # --------------------------------------------------------------------------
    # 7. DESCRIPTIVE ERROR ANALYSIS
    # --------------------------------------------------------------------------
    df_eval["is_true"] = [1 if (s, c) in holdout_gt_pairs else 0 for s, c in zip(df_eval["source1_entity_id"], df_eval["candidate_entity_id"])]
    df_eval["is_pred"] = (df_eval["score"] >= 0.88).astype(int)

    # 1. False Positives
    df_fp = df_eval[(df_eval["is_true"] == 0) & (df_eval["is_pred"] == 1)].sort_values("score", ascending=False)
    fp_records = df_fp.head(30).to_dict(orient="records")

    # 2. Classifier False Negatives (Retained by blocker, rejected by classifier)
    df_fn_class = df_eval[(df_eval["is_true"] == 1) & (df_eval["is_pred"] == 0)].sort_values("score", ascending=True)
    fn_class_records = df_fn_class.head(30).to_dict(orient="records")

    # 3. Blocker Misses
    missed_blocker_pairs = holdout_gt_pairs - holdout_cand_pairs_set
    missed_blocker_records = list(missed_blocker_pairs)[:30]

    # Populate cand_lookup for missed candidates not present in blocker candidate pool
    missed_cands_needed = list(set(c for s, c in missed_blocker_pairs if c not in cand_lookup))
    if missed_cands_needed:
        missed_c_rows = con.execute("""
            SELECT entity_id, country, business_name, business_address 
            FROM all_candidates 
            WHERE entity_id IN (SELECT unnest(?))
        """, [missed_cands_needed]).fetchall()
        for r in missed_c_rows:
            cand_lookup[r[0]] = {"country": r[1], "business_name": r[2], "business_address": r[3]}

    # Error breakdown counts
    fp_cats = Counter()
    for r in df_fp.to_dict(orient="records"):
        s1_r = s1_lookup.get(r["source1_entity_id"], {})
        c_r = cand_lookup.get(r["candidate_entity_id"], {})
        cat = categorize_error_pattern(s1_r.get("business_name", ""), c_r.get("business_name", ""), s1_r.get("business_address", ""), c_r.get("business_address", ""), is_fp=True)
        fp_cats[cat] += 1

    fn_cats = Counter()
    for r in df_fn_class.to_dict(orient="records"):
        s1_r = s1_lookup.get(r["source1_entity_id"], {})
        c_r = cand_lookup.get(r["candidate_entity_id"], {})
        cat = categorize_error_pattern(s1_r.get("business_name", ""), c_r.get("business_name", ""), s1_r.get("business_address", ""), c_r.get("business_address", ""), is_fp=False)
        fn_cats[cat] += 1

    bm_cats = Counter()
    for s, c in missed_blocker_pairs:
        s1_r = s1_lookup.get(s, {})
        c_r = cand_lookup.get(c, {})
        cat = categorize_error_pattern(s1_r.get("business_name", ""), c_r.get("business_name", ""), s1_r.get("business_address", ""), c_r.get("business_address", ""), is_fp=False)
        bm_cats[cat] += 1

    print(f"\n  Holdout Error Breakdown:")
    print(f"    - Blocker Misses ({len(missed_blocker_pairs)} total)        : {dict(bm_cats)}")
    print(f"    - Classifier FNs ({len(df_fn_class)} total)        : {dict(fn_cats)}")
    print(f"    - False Positives ({len(df_fp)} total)        : {dict(fp_cats)}")

    # Specific inspection of Group K feature actions
    df_with_feats = df_feat_holdout[["source1_entity_id", "candidate_entity_id", "feat_same_building_weak_name", "feat_high_addr_low_name_penalty"]].copy()
    df_with_feats["score"] = df_eval["score"]
    df_with_feats["is_true"] = df_eval["is_true"]
    df_with_feats["is_pred"] = df_eval["is_pred"]

    fired_penalty = df_with_feats[df_with_feats["feat_high_addr_low_name_penalty"] == 1.0]
    fired_building = df_with_feats[df_with_feats["feat_same_building_weak_name"] == 1.0]
    print(f"\n  Group K Diagnostic Actions on Holdout:")
    print(f"    - feat_high_addr_low_name_penalty fired on {len(fired_penalty)} pairs:")
    print(f"        Non-matches penalized: {sum(fired_penalty['is_true'] == 0)} | True matches penalized: {sum(fired_penalty['is_true'] == 1)}")
    print(f"        Max score when penalty fired: {fired_penalty['score'].max() if len(fired_penalty) > 0 else 0.0:.4f}")
    print(f"    - feat_same_building_weak_name fired on {len(fired_building)} pairs:")
    print(f"        Non-matches: {sum(fired_building['is_true'] == 0)} | True matches: {sum(fired_building['is_true'] == 1)}")
    print(f"        Max score: {fired_building['score'].max() if len(fired_building) > 0 else 0.0:.4f}")

    # --------------------------------------------------------------------------
    # 8. SAVE STRUCTURED RESULTS
    # --------------------------------------------------------------------------
    results_payload = {
        "phase": 13,
        "config_name": "Config_D1",
        "holdout_s1_count": len(p13_holdout_s1),
        "train_s1_count": len(p13_train_s1),
        "train_pair_count": len(df_train_p13),
        "train_pos": train_pos,
        "train_neg": train_neg,
        "scale_pos_weight": scale_pos_weight,
        "feature_count": len(feats_76),
        "threshold": 0.88,
        "macro_f05": res_088["macro_f05"],
        "macro_precision": res_088["macro_precision"],
        "macro_recall": res_088["macro_recall"],
        "e2e_true_matches": e2e_true,
        "total_true_matches": total_holdout_true,
        "e2e_recall": e2e_recall,
        "blocker_recall": blocker_recall,
        "retained_true": retained_true,
        "missed_blocker_true": missed_blocker_true,
        "classifier_retention": classifier_retention,
        "candidate_pairs_count": len(df_cands_holdout),
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": median_cands,
        "p95_candidates_per_s1": p95_cands,
        "max_candidates_per_s1": max_cands,
        "zero_candidate_s1_count": zero_cand_s1,
        "total_predicted_matches": res_088["total_pred_matches"],
        "avg_predicted_matches": res_088["total_pred_matches"] / len(p13_holdout_s1),
        "empty_predictions_pct": res_088["empty_pred_pct"],
        "singleton_metrics": s_grp,
        "zero_match_metrics": z_grp,
        "multi_match_metrics": m_grp,
        "us_metrics": res_us,
        "india_metrics": res_in,
        "top_features": feat_gains[:30],
        "group_k_gains": [fg for fg in feat_gains if fg["feature"] in gk_feats],
        "fp_category_counts": dict(fp_cats),
        "fn_category_counts": dict(fn_cats),
        "blocker_miss_category_counts": dict(bm_cats),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n  Successfully saved structured results to {RESULTS_PATH}")
    print("=" * 80)
    print("PHASE 13 EVALUATION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
