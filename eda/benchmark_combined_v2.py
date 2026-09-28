"""
Phase 10: Controlled Combination of Enhanced Blocking + Feature Improvements
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Establishes the 3-way S1-level split (Train 60%, Dev 20%, Holdout 20% LOCKED).
2. Trains 3 supervised XGBoost C1 models on the 273,838 training pairs:
   - Model 65: Canonical 65 baseline features
   - Model 74: 65 baseline + 9 selected high-gain Phase 8 features
   - Model 92: Complete 92-feature schema (Baseline + all error-driven features)
3. Generates candidate sets on the Development Set (998 S1s):
   - Set 1: Baseline Blocker, Cap 150 (92,515 pairs)
   - Set 2: Enhanced Blocker V2, Cap 150 (95,682 pairs)
   - Set 3: Enhanced Blocker V2, Cap 200 (109,385 pairs)
4. Evaluates all 7 controlled configurations:
   - Config A: Baseline Blocker Cap 150 + 65 features
   - Config B: Enhanced V2 Cap 150 + 65 features
   - Config C: Enhanced V2 Cap 200 + 65 features
   - Config D: Enhanced V2 Cap 150 + 74 features (selected)
   - Config E: Enhanced V2 Cap 200 + 74 features (selected)
   - Config F: Enhanced V2 Cap 150 + 92 features (all)
   - Config G: Enhanced V2 Cap 200 + 92 features (all)
5. Performs threshold search [0.70 to 0.95] for each configuration:
   - S1 Macro-F0.5, Macro Precision, Macro Recall
   - Blocker Recall vs End-to-End Classifier Recall
   - Singleton, Zero-Match, and Multi-Match diagnostics
6. Performs deep-dive Error Analysis on the winning configuration:
   - Categorizes 30 FPs, 30 FNs, 20 newly recovered true matches, 20 new-channel matches
7. Extracts Feature Importance for the winning configuration:
   - Top 30 features by Gain and aggregate group importance
8. Saves all structured results to 'output/benchmark_phase10_results.json'.
"""

import sys
import os
import io
import time
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np
import pandas as pd
import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.blocking import CandidateBlocker
from src.blocking_v2 import CandidateBlockerV2
from src.features import PairwiseFeatureExtractor
from src.features_v2 import PairwiseFeatureExtractorV2
from src.feature_schema import FEATURE_NAMES as BASELINE_NAMES
from src.feature_schema_v2 import (
    FEATURE_SPECS_V2,
    FEATURE_NAMES_V2,
    SELECTED_PHASE8_FEATURES,
    EXPERIMENT_FEATURE_SETS,
)
from src.model import EntityMatcherModel
from src.evaluate import evaluate_predictions_df, evaluate_by_group

OUTPUT_DIR = "output"
GT_PATH = os.path.join(OUTPUT_DIR, "eval_ground_truth.pkl")
TRAIN_PARQUET_V2 = os.path.join(OUTPUT_DIR, "eval_features_v2_cap150.parquet")
BASE_TRAIN = "student_resource/dataset/train"


def setup_environment_and_models():
    print("=" * 80)
    print("PHASE 10: CONTROLLED COMBINATION OF BLOCKING + FEATURES")
    print("=" * 80)
    print("\n[1/5] Loading ground truth, S1 split, and training supervised models...")

    with open(GT_PATH, "rb") as f:
        meta = pickle.load(f)

    gt_dict = meta["gt_dict"]
    s1_list_all = meta["s1_list"]
    s1_lookup = meta["s1_lookup"]

    # Stratified 3-way split
    strata = defaultdict(list)
    for s1 in s1_list_all:
        c = s1_lookup[s1]["country"]
        num_matches = len(gt_dict.get(s1, set()))
        if num_matches == 0:
            m_type = "zero"
        elif num_matches == 1:
            m_type = "single"
        else:
            m_type = "multi"
        strata[(c, m_type)].append(s1)

    train_s1, dev_s1, holdout_s1 = [], [], []
    rng = np.random.RandomState(42)
    for (c, m_type), ids in sorted(strata.items()):
        shuffled = rng.permutation(ids)
        n = len(shuffled)
        n_train = int(n * 0.60)
        n_dev = int(n * 0.20)
        train_s1.extend(shuffled[:n_train])
        dev_s1.extend(shuffled[n_train:n_train + n_dev])
        holdout_s1.extend(shuffled[n_train + n_dev:])

    # Strictly assert zero holdout leakage
    assert len(set(train_s1) & set(dev_s1)) == 0
    assert len(set(train_s1) & set(holdout_s1)) == 0
    assert len(set(dev_s1) & set(holdout_s1)) == 0

    dev_gt_pairs = set()
    for s in dev_s1:
        for c in gt_dict.get(s, set()):
            dev_gt_pairs.add((s, c))

    total_dev_true = len(dev_gt_pairs)
    print(f"  Train S1s: {len(train_s1):,} | Dev S1s: {len(dev_s1):,} | Dev True Matches: {total_dev_true:,}")

    # Load precomputed V2 features for the training split
    t0_tr = time.time()
    df_all_v2 = pd.read_parquet(TRAIN_PARQUET_V2)
    train_s1_set = set(train_s1)
    df_train = df_all_v2[df_all_v2["source1_entity_id"].isin(train_s1_set)].copy()

    train_pos = int(df_train["match_label"].sum())
    train_neg = int(len(df_train) - train_pos)
    scale_pos_weight = float(np.sqrt(train_neg / train_pos))
    print(f"  Loaded {len(df_train):,} training pairs ({train_pos:,} pos, {train_neg:,} neg | scale_pos_weight={scale_pos_weight:.2f}) in {time.time()-t0_tr:.2f}s.")

    # Define feature lists
    feats_65 = list(BASELINE_NAMES)
    feats_74 = list(EXPERIMENT_FEATURE_SETS["Selected_Phase8"])
    feats_92 = list(FEATURE_NAMES_V2)

    # Train Model 65
    print("\n  Fitting Model 65 (Canonical 65 features)...")
    m65 = EntityMatcherModel(
        model_type="xgboost_weighted",
        params={
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight, "tree_method": "hist",
        },
        random_state=42, feature_names=feats_65,
    )
    t0 = time.time()
    m65.fit(df_train[feats_65].values, df_train["match_label"].values)
    print(f"    Fitted in {time.time()-t0:.2f}s.")

    # Train Model 74
    print("  Fitting Model 74 (65 Baseline + 9 Selected Phase 8 features)...")
    m74 = EntityMatcherModel(
        model_type="xgboost_weighted",
        params={
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight, "tree_method": "hist",
        },
        random_state=42, feature_names=feats_74,
    )
    t0 = time.time()
    m74.fit(df_train[feats_74].values, df_train["match_label"].values)
    print(f"    Fitted in {time.time()-t0:.2f}s.")

    # Train Model 92
    print("  Fitting Model 92 (All 92 features)...")
    m92 = EntityMatcherModel(
        model_type="xgboost_weighted",
        params={
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight, "tree_method": "hist",
        },
        random_state=42, feature_names=feats_92,
    )
    t0 = time.time()
    m92.fit(df_train[feats_92].values, df_train["match_label"].values)
    print(f"    Fitted in {time.time()-t0:.2f}s.")

    # Setup DuckDB tables
    print("\n[2/5] Setting up DuckDB database for Dev candidate generation...")
    con = duckdb.connect()
    dev_s1_rows = [
        (s, s1_lookup[s]["business_name"], s1_lookup[s]["business_address"], s1_lookup[s]["country"])
        for s in dev_s1
    ]
    df_dev_s1 = pd.DataFrame(dev_s1_rows, columns=["entity_id", "business_name", "business_address", "country"])
    con.register("dev_s1_tbl", df_dev_s1)

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
    n_cands = con.execute("SELECT count(*) FROM all_candidates").fetchone()[0]
    print(f"  Loaded {n_cands:,} candidate records in {time.time()-t0_cand:.2f}s.")

    models_dict = {
        "model_65": (m65, feats_65),
        "model_74": (m74, feats_74),
        "model_92": (m92, feats_92),
    }

    return con, models_dict, dev_s1, s1_lookup, gt_dict, dev_gt_pairs, total_dev_true


def generate_and_extract_candidate_set(
    con: duckdb.DuckDBPyConnection,
    s1_lookup: Dict[str, Dict[str, str]],
    blocker_type: str,  # 'baseline' or 'enhanced_v2'
    max_cap: int,
    table_name: str,
) -> Tuple[pd.DataFrame, float, float]:
    """Generates candidate pairs and extracts full 92 V2 features."""
    t0_block = time.time()
    if blocker_type == "baseline":
        blocker = CandidateBlocker(con)
        blocker.generate_candidates(
            s1_table_or_path="dev_s1_tbl",
            cand_table_or_path="all_candidates",
            output_table=table_name,
            max_candidates_per_s1=max_cap,
        )
    else:
        blocker = CandidateBlockerV2(con)
        blocker.generate_candidates(
            s1_table_or_path="dev_s1_tbl",
            cand_table_or_path="all_candidates",
            output_table=table_name,
            max_candidates_per_s1=max_cap,
            enable_enhanced_channels=True,
        )
    t_block = time.time() - t0_block

    df_cands = con.execute(f"SELECT * FROM {table_name}").fetchdf()

    # Extract distinct candidate records
    t0_feat = time.time()
    c_rows = con.execute(f"""
        SELECT entity_id, country, business_name, business_address 
        FROM all_candidates 
        WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM {table_name})
    """).fetchall()
    cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in c_rows}

    # Extract all 92 features
    extractor_v2 = PairwiseFeatureExtractorV2()
    df_feat_v2 = extractor_v2.extract_features(df_cands, s1_lookup, cand_lookup)
    t_feat = time.time() - t0_feat

    return df_feat_v2, t_block, t_feat


def evaluate_configuration(
    config_name: str,
    df_feat: pd.DataFrame,
    model: EntityMatcherModel,
    feature_list: List[str],
    dev_s1: List[str],
    gt_dict: Dict[str, Set[str]],
    dev_gt_pairs: Set[Tuple[str, str]],
    total_dev_true: int,
    t_block: float,
    t_feat: float,
) -> Dict[str, Any]:
    print(f"\n" + "-" * 75)
    print(f"EVALUATING CONFIGURATION: {config_name}")
    print(f"  Features: {len(feature_list)} | Pairs: {len(df_feat):,}")
    print("-" * 75)

    # Blocker stats
    counts_s1 = df_feat.groupby("source1_entity_id")["candidate_entity_id"].count().to_dict()
    for s in dev_s1:
        if s not in counts_s1:
            counts_s1[s] = 0
    vals = list(counts_s1.values())
    avg_cands = float(np.mean(vals))
    median_cands = float(np.median(vals))
    p95_cands = float(np.percentile(vals, 95))
    max_cands = int(np.max(vals))

    cand_pairs_set = set(zip(df_feat["source1_entity_id"], df_feat["candidate_entity_id"]))
    retained_true = len(dev_gt_pairs & cand_pairs_set)
    blocker_recall = retained_true / total_dev_true * 100.0
    missed_count = total_dev_true - retained_true
    cands_per_true = len(df_feat) / retained_true if retained_true > 0 else 0.0

    print(f"  Candidates Generated : {len(df_feat):,} (Avg: {avg_cands:.1f}/S1, P95: {p95_cands:.0f}, Max: {max_cands})")
    print(f"  Blocker Recall       : {blocker_recall:.2f}% ({retained_true:,}/{total_dev_true:,} true matches | Missed: {missed_count:,})")
    print(f"  Cands per True Match : {cands_per_true:.1f} | Block Time: {t_block:.2f}s | Feat Time: {t_feat:.2f}s")

    # Model inference
    df_eval = df_feat[["source1_entity_id", "candidate_entity_id"]].copy()
    X_mat = df_feat[feature_list].values
    t0_inf = time.time()
    df_eval["score"] = model.predict_proba(X_mat)
    t_inf = time.time() - t0_inf

    # 1. Fixed threshold 0.88 evaluation
    res_088 = evaluate_predictions_df(df_eval, gt_dict, threshold=0.88, s1_list=dev_s1)
    grp_088 = evaluate_by_group(df_eval, gt_dict, threshold=0.88, s1_list=dev_s1)
    matched_088 = df_eval[df_eval["score"] >= 0.88]
    pred_pairs_088 = set(zip(matched_088["source1_entity_id"], matched_088["candidate_entity_id"]))
    tp_088 = len(dev_gt_pairs & pred_pairs_088)
    e2e_rec_088 = tp_088 / total_dev_true * 100.0

    print(f"  At Fixed Threshold 0.88:")
    print(f"    Macro-F0.5: {res_088['macro_f05']:.4f} | Prec: {res_088['macro_precision']:.4f} | Rec: {res_088['macro_recall']:.4f}")
    print(f"    Blocker Rec: {blocker_recall:.2f}% -> E2E Model Rec: {e2e_rec_088:.2f}% ({tp_088:,} true matches)")
    print(f"    Pred Matches: {res_088['total_pred_matches']} | % Empty: {res_088['empty_pred_pct']:.1f}%")

    # 2. Optimal threshold sweep [0.70 to 0.95]
    best_thresh = 0.88
    best_f05 = -1.0
    best_res = None
    for t in np.arange(0.70, 0.96, 0.01):
        t_val = round(float(t), 2)
        cur_res = evaluate_predictions_df(df_eval, gt_dict, threshold=t_val, s1_list=dev_s1)
        if cur_res["macro_f05"] > best_f05:
            best_f05 = cur_res["macro_f05"]
            best_thresh = t_val
            best_res = cur_res

    best_grp = evaluate_by_group(df_eval, gt_dict, threshold=best_thresh, s1_list=dev_s1)
    matched_opt = df_eval[df_eval["score"] >= best_thresh]
    pred_pairs_opt = set(zip(matched_opt["source1_entity_id"], matched_opt["candidate_entity_id"]))
    tp_opt = len(dev_gt_pairs & pred_pairs_opt)
    e2e_rec_opt = tp_opt / total_dev_true * 100.0

    print(f"  At Optimal Threshold {best_thresh:.2f}:")
    print(f"    Macro-F0.5: {best_res['macro_f05']:.4f} | Prec: {best_res['macro_precision']:.4f} | Rec: {best_res['macro_recall']:.4f}")
    print(f"    Blocker Rec: {blocker_recall:.2f}% -> E2E Model Rec: {e2e_rec_opt:.2f}% ({tp_opt:,} true matches)")
    print(f"    Pred Matches: {best_res['total_pred_matches']} | % Empty: {best_res['empty_pred_pct']:.1f}%")
    print(f"    - Singletons: Prec {best_grp['group1_singleton']['macro_precision']:.4f}, Rec {best_grp['group1_singleton']['macro_recall']:.4f}, F0.5 {best_grp['group1_singleton']['macro_f05']:.4f}")
    print(f"    - Zero-Match: Prec {best_grp['group2_zero_match']['macro_precision']:.4f}, Rec {best_grp['group2_zero_match']['macro_recall']:.4f}, F0.5 {best_grp['group2_zero_match']['macro_f05']:.4f} (FP: {best_grp['group2_zero_match']['false_positive_rate']:.2f}%)")
    print(f"    - Multi-Match: Prec {best_grp['group3_multi_match']['macro_precision']:.4f}, Rec {best_grp['group3_multi_match']['macro_recall']:.4f}, F0.5 {best_grp['group3_multi_match']['macro_f05']:.4f}")

    return {
        "config_name": config_name,
        "feature_count": len(feature_list),
        "total_cands": len(df_feat),
        "avg_cands": avg_cands,
        "median_cands": median_cands,
        "p95_cands": p95_cands,
        "max_cands": max_cands,
        "blocker_recall": blocker_recall,
        "retained_true": retained_true,
        "missed_true": missed_count,
        "cands_per_true": cands_per_true,
        "runtime_block": t_block,
        "runtime_feat": t_feat,
        "runtime_inf": t_inf,
        "df_eval": df_eval,
        "model": model,
        "canonical_088": {
            "threshold": 0.88,
            "macro_precision": res_088["macro_precision"],
            "macro_recall": res_088["macro_recall"],
            "macro_f05": res_088["macro_f05"],
            "e2e_recall": e2e_rec_088,
            "e2e_matches": tp_088,
            "pred_matches": res_088["total_pred_matches"],
            "empty_pct": res_088["empty_pred_pct"],
            "singleton": grp_088["group1_singleton"],
            "zero_match": grp_088["group2_zero_match"],
            "multi_match": grp_088["group3_multi_match"],
        },
        "optimal": {
            "threshold": best_thresh,
            "macro_precision": best_res["macro_precision"],
            "macro_recall": best_res["macro_recall"],
            "macro_f05": best_res["macro_f05"],
            "e2e_recall": e2e_rec_opt,
            "e2e_matches": tp_opt,
            "pred_matches": best_res["total_pred_matches"],
            "empty_pct": best_res["empty_pred_pct"],
            "singleton": best_grp["group1_singleton"],
            "zero_match": best_grp["group2_zero_match"],
            "multi_match": best_grp["group3_multi_match"],
        }
    }


import re
import Levenshtein


def categorize_error_pattern(s1_name: str, cand_name: str, s1_addr: str, cand_addr: str, is_fp: bool = False) -> str:
    s1_n = (s1_name or "").lower().strip()
    cand_n = (cand_name or "").lower().strip()
    s1_a = (s1_addr or "").lower().strip()
    cand_a = (cand_addr or "").lower().strip()

    # 1. Missing address
    if not s1_a or not cand_a:
        return "missing address"

    # 2. Native script
    if any(ord(char) > 127 for char in s1_n + cand_n):
        return "native script"

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
            return "same building"
        elif not is_fp and jaccard < 0.3:
            return "aliases"

    # 4. Generic business names
    generic_words = {"store", "shop", "restaurant", "hotel", "cafe", "agency", "enterprises", "traders", "mart", "market", "services"}
    if len(toks1 & generic_words) > 0 and len(toks2 & generic_words) > 0 and jaccard < 0.4:
        return "generic business names"

    # 5. Franchise / Brand
    if s1_n == cand_n and s1_a != cand_a:
        return "franchise/brand"

    # 6. Postal / Name-prefix match
    if same_postal and len(s1_n) >= 3 and len(cand_n) >= 3 and s1_n[:3] == cand_n[:3] and jaccard < 0.5:
        return "postal/name-prefix match"

    # 7. Address-number match
    if same_num and jaccard < 0.4:
        return "address-number match"

    # 8. Same postal code
    if same_postal and jaccard < 0.4:
        return "same postal code"

    # 9. OCR / Typo
    lev = Levenshtein.distance(s1_n, cand_n)
    if lev <= 2 and len(s1_n) >= 5:
        return "OCR/typo"

    # 10. Aliases
    if jaccard < 0.35 and (s1_a == cand_a or same_postal or same_num or len(toks1 & toks2) > 0):
        return "aliases"

    return "other"


def perform_detailed_error_analysis(
    best_config: Dict[str, Any],
    base_config: Dict[str, Any],
    con: duckdb.DuckDBPyConnection,
    s1_lookup: Dict[str, Dict[str, str]],
    gt_dict: Dict[str, Set[str]],
    dev_gt_pairs: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("DETAILED ERROR ANALYSIS ON WINNING CONFIGURATION: " + best_config["config_name"])
    print("=" * 80)

    df_eval = best_config["df_eval"]
    opt_thresh = best_config["optimal"]["threshold"]

    df_eval["is_true"] = [
        1 if (s, c) in dev_gt_pairs else 0
        for s, c in zip(df_eval["source1_entity_id"], df_eval["candidate_entity_id"])
    ]
    df_eval["is_pred"] = (df_eval["score"] >= opt_thresh).astype(int)

    # Base config predictions and candidate set
    base_eval = base_config["df_eval"]
    base_opt_thresh = base_config["optimal"]["threshold"]
    base_pred_pairs = set(
        zip(
            base_eval[base_eval["score"] >= base_opt_thresh]["source1_entity_id"],
            base_eval[base_eval["score"] >= base_opt_thresh]["candidate_entity_id"],
        )
    )
    base_cand_pairs = set(zip(base_eval["source1_entity_id"], base_eval["candidate_entity_id"]))

    win_pred_pairs = set(
        zip(
            df_eval[df_eval["score"] >= opt_thresh]["source1_entity_id"],
            df_eval[df_eval["score"] >= opt_thresh]["candidate_entity_id"],
        )
    )

    # 1. False Positives (30 sample)
    df_fp = df_eval[(df_eval["is_true"] == 0) & (df_eval["is_pred"] == 1)].copy()
    df_fp = df_fp.sort_values("score", ascending=False)
    fp_records = df_fp.head(30).to_dict(orient="records")

    # 2. False Negatives (30 sample)
    df_fn = df_eval[(df_eval["is_true"] == 1) & (df_eval["is_pred"] == 0)].copy()
    df_fn = df_fn.sort_values("score", ascending=True)
    fn_records = df_fn.head(30).to_dict(orient="records")

    # 3. Newly Recovered True Matches (Predicted by Winner, but NOT by Baseline Config A) (20 sample)
    newly_rec_pairs = (win_pred_pairs & dev_gt_pairs) - base_pred_pairs
    df_newly_rec = df_eval[
        df_eval.apply(lambda r: (r["source1_entity_id"], r["candidate_entity_id"]) in newly_rec_pairs, axis=1)
    ].copy()
    df_newly_rec = df_newly_rec.sort_values("score", ascending=False)
    newly_rec_records = df_newly_rec.head(20).to_dict(orient="records")

    # 4. Matches recovered ONLY by new blocking channels (Predicted by Winner, but NOT in Baseline Candidate Set) (20 sample)
    new_channel_pairs = (win_pred_pairs & dev_gt_pairs) - base_cand_pairs
    df_new_chan = df_eval[
        df_eval.apply(lambda r: (r["source1_entity_id"], r["candidate_entity_id"]) in new_channel_pairs, axis=1)
    ].copy()
    df_new_chan = df_new_chan.sort_values("score", ascending=False)
    new_chan_records = df_new_chan.head(20).to_dict(orient="records")

    # Fetch candidate lookup for names and addresses
    needed_cand_ids = set()
    for r in fp_records + fn_records + newly_rec_records + new_chan_records:
        needed_cand_ids.add(r["candidate_entity_id"])

    c_rows = con.execute(f"""
        SELECT entity_id, country, business_name, business_address 
        FROM all_candidates 
        WHERE entity_id IN ({','.join([f"'{c}'" for c in needed_cand_ids])})
    """).fetchall()
    c_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in c_rows}

    # Categorize FPs
    fp_categories = defaultdict(int)
    fp_details = []
    for r in fp_records:
        s1_id = r["source1_entity_id"]
        c_id = r["candidate_entity_id"]
        s1_rec = s1_lookup.get(s1_id, {})
        c_rec = c_lookup.get(c_id, {})
        s1_n, c_n = s1_rec.get("business_name", ""), c_rec.get("business_name", "")
        s1_a, c_a = s1_rec.get("business_address", ""), c_rec.get("business_address", "")
        cat = categorize_error_pattern(s1_n, c_n, s1_a, c_a, is_fp=True)
        fp_categories[cat] += 1
        fp_details.append({
            "s1_id": s1_id, "cand_id": c_id, "score": round(float(r["score"]), 4),
            "category": cat, "s1_name": s1_n, "cand_name": c_n, "s1_addr": s1_a, "cand_addr": c_a,
        })

    # Categorize FNs
    fn_categories = defaultdict(int)
    fn_details = []
    for r in fn_records:
        s1_id = r["source1_entity_id"]
        c_id = r["candidate_entity_id"]
        s1_rec = s1_lookup.get(s1_id, {})
        c_rec = c_lookup.get(c_id, {})
        s1_n, c_n = s1_rec.get("business_name", ""), c_rec.get("business_name", "")
        s1_a, c_a = s1_rec.get("business_address", ""), c_rec.get("business_address", "")
        cat = categorize_error_pattern(s1_n, c_n, s1_a, c_a, is_fp=False)
        fn_categories[cat] += 1
        fn_details.append({
            "s1_id": s1_id, "cand_id": c_id, "score": round(float(r["score"]), 4),
            "category": cat, "s1_name": s1_n, "cand_name": c_n, "s1_addr": s1_a, "cand_addr": c_a,
        })

    # Categorize Newly Recovered Matches
    newly_rec_categories = defaultdict(int)
    newly_rec_details = []
    for r in newly_rec_records:
        s1_id = r["source1_entity_id"]
        c_id = r["candidate_entity_id"]
        s1_rec = s1_lookup.get(s1_id, {})
        c_rec = c_lookup.get(c_id, {})
        s1_n, c_n = s1_rec.get("business_name", ""), c_rec.get("business_name", "")
        s1_a, c_a = s1_rec.get("business_address", ""), c_rec.get("business_address", "")
        cat = categorize_error_pattern(s1_n, c_n, s1_a, c_a, is_fp=False)
        newly_rec_categories[cat] += 1
        newly_rec_details.append({
            "s1_id": s1_id, "cand_id": c_id, "score": round(float(r["score"]), 4),
            "category": cat, "s1_name": s1_n, "cand_name": c_n, "s1_addr": s1_a, "cand_addr": c_a,
        })

    # Categorize New-Channel Matches
    new_chan_categories = defaultdict(int)
    new_chan_details = []
    for r in new_chan_records:
        s1_id = r["source1_entity_id"]
        c_id = r["candidate_entity_id"]
        s1_rec = s1_lookup.get(s1_id, {})
        c_rec = c_lookup.get(c_id, {})
        s1_n, c_n = s1_rec.get("business_name", ""), c_rec.get("business_name", "")
        s1_a, c_a = s1_rec.get("business_address", ""), c_rec.get("business_address", "")
        cat = categorize_error_pattern(s1_n, c_n, s1_a, c_a, is_fp=False)
        new_chan_categories[cat] += 1
        new_chan_details.append({
            "s1_id": s1_id, "cand_id": c_id, "score": round(float(r["score"]), 4),
            "category": cat, "s1_name": s1_n, "cand_name": c_n, "s1_addr": s1_a, "cand_addr": c_a,
        })

    print(f"\n--- FALSE POSITIVE CATEGORIZATION (TOP 30 FPs) ---")
    for cat, cnt in sorted(fp_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<32}: {cnt} ({cnt/30*100:.1f}%)")

    print(f"\n--- FALSE NEGATIVE CATEGORIZATION (TOP 30 FNs) ---")
    for cat, cnt in sorted(fn_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<32}: {cnt} ({cnt/30*100:.1f}%)")

    print(f"\n--- NEWLY RECOVERED TRUE MATCHES CATEGORIZATION (20 SAMPLE) ---")
    print(f"  Total Newly Recovered Matches vs Config A: {len(newly_rec_pairs):,}")
    for cat, cnt in sorted(newly_rec_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<32}: {cnt} ({cnt/len(newly_rec_records)*100:.1f}%)")

    print(f"\n--- NEW BLOCKING CHANNEL BREAKTHROUGH MATCHES (20 SAMPLE) ---")
    print(f"  Total Matches from New Channels: {len(new_channel_pairs):,}")
    for cat, cnt in sorted(new_chan_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<32}: {cnt} ({cnt/len(new_chan_records)*100:.1f}%)")

    # Check for systematic false positive patterns from new blocker
    fp_new_channels = [r for r in fp_records if (r["source1_entity_id"], r["candidate_entity_id"]) not in base_cand_pairs]
    print(f"\nNew Blocker Systematic FP Check:")
    print(f"  FPs introduced exclusively by new blocker channels: {len(fp_new_channels)} / 30 ({len(fp_new_channels)/30*100:.1f}%)")

    return {
        "fp_categories": dict(fp_categories),
        "fn_categories": dict(fn_categories),
        "newly_rec_categories": dict(newly_rec_categories),
        "new_chan_categories": dict(new_chan_categories),
        "total_newly_recovered": len(newly_rec_pairs),
        "total_new_channel_matches": len(new_channel_pairs),
        "fp_from_new_channels_count": len(fp_new_channels),
        "fp_sample": fp_details,
        "fn_sample": fn_details,
        "newly_recovered_sample": newly_rec_details,
        "new_channel_sample": new_chan_details,
    }


def main():
    con, models_dict, dev_s1, s1_lookup, gt_dict, dev_gt_pairs, total_dev_true = setup_environment_and_models()

    # Generate Candidate Sets on Dev
    print("\n[3/5] Generating candidate sets and extracting features on Development Set...")

    print("  Generating Candidate Set 1: Baseline Blocker (Cap 150)...")
    df_base150, t_b1, t_f1 = generate_and_extract_candidate_set(con, s1_lookup, "baseline", 150, "dev_cands_base150")
    print(f"    Set 1: {len(df_base150):,} pairs (Blocker: {t_b1:.2f}s, Feat: {t_f1:.2f}s)")

    print("  Generating Candidate Set 2: Enhanced Blocker V2 (Cap 150)...")
    df_v2_150, t_b2, t_f2 = generate_and_extract_candidate_set(con, s1_lookup, "enhanced_v2", 150, "dev_cands_v2_150")
    print(f"    Set 2: {len(df_v2_150):,} pairs (Blocker: {t_b2:.2f}s, Feat: {t_f2:.2f}s)")

    print("  Generating Candidate Set 3: Enhanced Blocker V2 (Cap 200)...")
    df_v2_200, t_b3, t_f3 = generate_and_extract_candidate_set(con, s1_lookup, "enhanced_v2", 200, "dev_cands_v2_200")
    print(f"    Set 3: {len(df_v2_200):,} pairs (Blocker: {t_b3:.2f}s, Feat: {t_f3:.2f}s)")

    # --------------------------------------------------------------------------
    # EVALUATE 7 CONTROLLED CONFIGURATIONS
    # --------------------------------------------------------------------------
    print("\n[4/5] Running end-to-end evaluation across 7 controlled configurations...")

    configs_to_run = [
        ("Config A: Baseline Blocker Cap 150 + 65 feats", df_base150, "model_65", t_b1, t_f1),
        ("Config B: Enhanced V2 Cap 150 + 65 feats", df_v2_150, "model_65", t_b2, t_f2),
        ("Config C: Enhanced V2 Cap 200 + 65 feats", df_v2_200, "model_65", t_b3, t_f3),
        ("Config D: Enhanced V2 Cap 150 + 74 feats (selected)", df_v2_150, "model_74", t_b2, t_f2),
        ("Config E: Enhanced V2 Cap 200 + 74 feats (selected)", df_v2_200, "model_74", t_b3, t_f3),
        ("Config F: Enhanced V2 Cap 150 + 92 feats (all)", df_v2_150, "model_92", t_b2, t_f2),
        ("Config G: Enhanced V2 Cap 200 + 92 feats (all)", df_v2_200, "model_92", t_b3, t_f3),
    ]

    results = []
    for cfg_name, df_cand, m_key, tb, tf in configs_to_run:
        model_obj, feat_cols = models_dict[m_key]
        res = evaluate_configuration(
            config_name=cfg_name,
            df_feat=df_cand,
            model=model_obj,
            feature_list=feat_cols,
            dev_s1=dev_s1,
            gt_dict=gt_dict,
            dev_gt_pairs=dev_gt_pairs,
            total_dev_true=total_dev_true,
            t_block=tb,
            t_feat=tf,
        )
        results.append(res)

    # --------------------------------------------------------------------------
    # SUMMARY TABLES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 10 CONTROLLED EXPERIMENT SUMMARY (OPTIMAL THRESHOLD)")
    print("=" * 115)
    base_f05 = results[0]["optimal"]["macro_f05"]
    rows_summary = []
    for r in results:
        o = r["optimal"]
        delta = o["macro_f05"] - base_f05
        rows_summary.append({
            "Configuration": r["config_name"],
            "Feats": r["feature_count"],
            "Pairs": f"{r['total_cands']:,}",
            "Opt Thresh": f"{o['threshold']:.2f}",
            "Macro Prec": f"{o['macro_precision']:.4f}",
            "Macro Rec": f"{o['macro_recall']:.4f}",
            "Macro F0.5": f"{o['macro_f05']:.4f}",
            "Delta F0.5": f"{delta:+.4f}",
            "Blocker Rec": f"{r['blocker_recall']:.2f}%",
            "E2E Rec": f"{o['e2e_recall']:.2f}%",
            "Single F0.5": f"{o['singleton']['macro_f05']:.4f}",
            "Zero F0.5": f"{o['zero_match']['macro_f05']:.4f}",
            "Multi F0.5": f"{o['multi_match']['macro_f05']:.4f}",
        })
    df_sum = pd.DataFrame(rows_summary)
    print(df_sum.to_string(index=False))

    # --------------------------------------------------------------------------
    # IDENTIFY WINNER & PERFORM ERROR ANALYSIS
    # --------------------------------------------------------------------------
    best_config = max(results, key=lambda r: r["optimal"]["macro_f05"])
    print("\n" + "=" * 80)
    print(f"WINNING CONFIGURATION: {best_config['config_name']}")
    print(f"Optimal Macro-F0.5: {best_config['optimal']['macro_f05']:.4f} at threshold {best_config['optimal']['threshold']:.2f}")
    print("=" * 80)

    # Detailed Error Analysis
    err_res = perform_detailed_error_analysis(best_config, results[0], con, s1_lookup, gt_dict, dev_gt_pairs)

    # Feature Importance for Winning Model
    print("\n--- FEATURE IMPORTANCES FOR WINNING MODEL ---")
    winning_model = best_config["model"]
    df_imp = winning_model.get_feature_importances()
    spec_dict = {s.name: s.group for s in FEATURE_SPECS_V2}
    df_imp["group"] = df_imp["feature"].map(lambda f: spec_dict.get(f, "Unknown"))
    df_imp["gain_pct"] = df_imp["gain_importance"] * 100.0

    print("\nTop 30 Features by Gain:")
    top30 = df_imp.head(30)[["feature", "gain_pct", "group"]].copy()
    top30["gain_pct"] = top30["gain_pct"].map(lambda x: f"{x:.2f}%")
    print(top30.to_string(index=True))

    print("\nAggregated Feature Group Gain:")
    grp_gain = df_imp.groupby("group")["gain_pct"].sum().reset_index()
    grp_gain = grp_gain.sort_values("gain_pct", ascending=False).reset_index(drop=True)
    grp_gain["gain_pct"] = grp_gain["gain_pct"].map(lambda x: f"{x:.2f}%")
    print(grp_gain.to_string(index=False))

    # Save to JSON
    json_path = os.path.join(OUTPUT_DIR, "benchmark_phase10_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "summary_table": rows_summary,
            "best_config": best_config["config_name"],
            "best_optimal_f05": best_config["optimal"]["macro_f05"],
            "best_optimal_thresh": best_config["optimal"]["threshold"],
            "top30_features": top30.to_dict(orient="records"),
            "group_gain": grp_gain.to_dict(orient="records"),
            "error_analysis": {
                "fp_categories": err_res["fp_categories"],
                "fn_categories": err_res["fn_categories"],
                "newly_rec_categories": err_res["newly_rec_categories"],
                "new_chan_categories": err_res["new_chan_categories"],
                "total_newly_recovered": err_res["total_newly_recovered"],
                "total_new_channel_matches": err_res["total_new_channel_matches"],
                "fp_from_new_channels_count": err_res["fp_from_new_channels_count"],
                "fp_sample": err_res["fp_sample"],
                "fn_sample": err_res["fn_sample"],
                "newly_recovered_sample": err_res["newly_recovered_sample"],
                "new_channel_sample": err_res["new_channel_sample"],
            }
        }, f, indent=2)
    print(f"\nSaved Phase 10 benchmark results to {json_path}")


if __name__ == "__main__":
    main()
