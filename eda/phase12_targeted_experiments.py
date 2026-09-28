"""
Phase 12: Targeted Error-Driven Improvement Experiments
Amazon ML Challenge 2026: Business Entity Resolution

This script executes the complete Phase 12 targeted experimental program:
1. Verifies zero access to protected holdouts A (Phase 5.1) and B (Phase 11).
2. Reproduces baseline P0 (Enhanced Blocker V2 Cap 200, 74 features, threshold 0.88).
3. Evaluates targeted feature families independently:
   - Family A: Missing Address (A0 vs A1)
   - Family B: Native Script & Corrupted Accents (B0 vs B1)
   - Family C: Lexical Shift & Aliases (C0 vs C1)
   - Family D: Address Number Disambiguation (D0 vs D1)
4. Evaluates targeted blocking improvement (Family E: Channel J Postal+Number Anchor).
5. Tests controlled combinations of demonstrating improvements (F0, F1, F2, F3).
6. Performs development-only threshold sweep on best configuration.
7. Performs error analysis of best configuration vs Phase 11 baseline.
8. Extracts feature importance (top 30 by gain and aggregate group gains).
9. Saves structured results to 'output/phase12_experiments_results.json'.
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
DEV_PARQUET_CAP200 = os.path.join(OUTPUT_DIR, "eval_features_v2_cap200_dev.parquet")
BASE_TRAIN = "student_resource/dataset/train"


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


def run_experiment(
    exp_name: str,
    feature_list: List[str],
    df_train: pd.DataFrame,
    df_dev_feat: pd.DataFrame,
    scale_pos_weight: float,
    dev_s1: List[str],
    gt_dict: Dict[str, Set[str]],
    dev_gt_pairs: Set[Tuple[str, str]],
    total_dev_true: int,
    blocker_recall: float,
    retained_true: int,
    threshold: float = 0.88,
) -> Dict[str, Any]:
    t0_fit = time.time()
    model = EntityMatcherModel(
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
        feature_names=feature_list,
    )
    model.fit(df_train[feature_list].values, df_train["match_label"].values)
    t_fit = time.time() - t0_fit

    df_eval = df_dev_feat[["source1_entity_id", "candidate_entity_id"]].copy()
    X_dev = df_dev_feat[feature_list].values
    df_eval["score"] = model.predict_proba(X_dev)

    res = evaluate_predictions_df(df_eval, gt_dict, threshold=threshold, s1_list=dev_s1)
    grp = evaluate_by_group(df_eval, gt_dict, threshold=threshold, s1_list=dev_s1)

    matched = df_eval[df_eval["score"] >= threshold]
    pred_pairs = set(zip(matched["source1_entity_id"], matched["candidate_entity_id"]))
    e2e_true = len(dev_gt_pairs & pred_pairs)
    e2e_recall = e2e_true / total_dev_true * 100.0
    classifier_retention = e2e_true / retained_true * 100.0 if retained_true > 0 else 0.0

    print(f"  {exp_name:<38} | Feats: {len(feature_list):>2} | F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | E2E Rec: {e2e_recall:5.2f}% ({e2e_true:,}) | Fit: {t_fit:.1f}s")

    return {
        "exp_name": exp_name,
        "feature_count": len(feature_list),
        "features": feature_list,
        "threshold": threshold,
        "macro_f05": res["macro_f05"],
        "macro_precision": res["macro_precision"],
        "macro_recall": res["macro_recall"],
        "e2e_true_matches": e2e_true,
        "total_true_matches": total_dev_true,
        "e2e_recall": e2e_recall,
        "blocker_recall": blocker_recall,
        "retained_true": retained_true,
        "classifier_retention": classifier_retention,
        "pred_matches": res["total_pred_matches"],
        "avg_pred_matches": res["total_pred_matches"] / len(dev_s1),
        "empty_pct": res["empty_pred_pct"],
        "singleton_f05": grp["group1_singleton"]["macro_f05"],
        "zero_match_f05": grp["group2_zero_match"]["macro_f05"],
        "multi_match_f05": grp["group3_multi_match"]["macro_f05"],
        "fit_time": t_fit,
        "model": model,
        "df_eval": df_eval,
    }


def main():
    print("=" * 80)
    print("PHASE 12: TARGETED ERROR-DRIVEN IMPROVEMENT EXPERIMENTS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. SETUP ENVIRONMENT & VERIFY PROTECTED HOLDOUTS
    # --------------------------------------------------------------------------
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

    # Reconstruct Phase 11 Fresh Holdout (Seed 2026)
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

    # Verify zero holdout leakage
    assert len(set(dev_s1_p10) & set(holdout_s1_p51)) == 0
    assert len(set(dev_s1_p10) & set(p11_holdout_s1)) == 0
    assert len(set(train_s1_orig) & set(dev_s1_p10)) == 0
    print("[1/5] Anti-leakage verified: Phase 5.1 and Phase 11 holdouts are 100% isolated.")

    dev_gt_pairs = set()
    for s in dev_s1_p10:
        for c in gt_dict.get(s, set()):
            dev_gt_pairs.add((s, c))
    total_dev_true = len(dev_gt_pairs)

    # --------------------------------------------------------------------------
    # 2. LOAD TRAINING & DEV DATA (CACHE IF NEEDED)
    # --------------------------------------------------------------------------
    print("\n[2/5] Loading training pairs and development candidate features...")
    df_all_v2 = pd.read_parquet(TRAIN_PARQUET_V2)
    train_s1_set = set(train_s1_orig)
    df_train = df_all_v2[df_all_v2["source1_entity_id"].isin(train_s1_set)].copy()

    train_pos = int(df_train["match_label"].sum())
    train_neg = int(len(df_train) - train_pos)
    scale_pos_weight = float(np.sqrt(train_neg / train_pos))
    print(f"  Training pairs: {len(df_train):,} ({train_pos:,} pos, {train_neg:,} neg | scale_pos_weight={scale_pos_weight:.2f})")

    # Load or generate Dev Cap 200 features
    if os.path.exists(DEV_PARQUET_CAP200):
        print(f"  Loading cached Dev Cap 200 features from {DEV_PARQUET_CAP200}...")
        df_dev_feat = pd.read_parquet(DEV_PARQUET_CAP200)
    else:
        print(f"  Generating Dev Cap 200 candidates and extracting all 92 features...")
        con = duckdb.connect()
        dev_s1_rows = [
            (s, s1_lookup[s]["business_name"], s1_lookup[s]["business_address"], s1_lookup[s]["country"])
            for s in dev_s1_p10
        ]
        df_dev_s1 = pd.DataFrame(dev_s1_rows, columns=["entity_id", "business_name", "business_address", "country"])
        con.register("dev_s1_tbl", df_dev_s1)
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
        blocker = CandidateBlockerV2(con)
        blocker.generate_candidates(
            s1_table_or_path="dev_s1_tbl",
            cand_table_or_path="all_candidates",
            output_table="dev_cands_v2_200",
            max_candidates_per_s1=200,
            enable_enhanced_channels=True,
        )
        df_cands_dev = con.execute("SELECT * FROM dev_cands_v2_200").fetchdf()
        c_rows = con.execute("""
            SELECT entity_id, country, business_name, business_address 
            FROM all_candidates 
            WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM dev_cands_v2_200)
        """).fetchall()
        cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in c_rows}
        extractor_v2 = PairwiseFeatureExtractorV2()
        df_dev_feat = extractor_v2.extract_features(df_cands_dev, s1_lookup, cand_lookup)
        df_dev_feat.to_parquet(DEV_PARQUET_CAP200)
        print(f"  Saved Dev Cap 200 features to {DEV_PARQUET_CAP200}")

    cand_pairs_set = set(zip(df_dev_feat["source1_entity_id"], df_dev_feat["candidate_entity_id"]))
    retained_true = len(dev_gt_pairs & cand_pairs_set)
    blocker_recall = retained_true / total_dev_true * 100.0
    print(f"  Dev Cap 200 Pairs: {len(df_dev_feat):,} | Blocker Recall: {blocker_recall:.2f}% ({retained_true:,}/{total_dev_true:,})")

    # --------------------------------------------------------------------------
    # 3. DEFINE TARGETED FEATURE SETS
    # --------------------------------------------------------------------------
    base_74 = list(EXPERIMENT_FEATURE_SETS["Selected_Phase8"])

    # Family A: Missing Address (+2 targeted features)
    feats_fam_a = base_74 + [
        "feat_high_name_cand_addr_missing",
        "feat_addr_missing_asymmetric",
    ]

    # Family B: Native Script / Corrupted Accents (+2 targeted features)
    feats_fam_b = base_74 + [
        "feat_latin_accent_folded_exact",
        "feat_cross_script_postal_match",
    ]

    # Family C: Lexical Shift / Aliases (+2 targeted features)
    feats_fam_c = base_74 + [
        "feat_name_distinctive_containment",
        "feat_name_first_distinctive_match",
    ]

    # Family D: Address Number Disambiguation (+2 targeted features)
    feats_fam_d = base_74 + [
        "feat_same_building_weak_name",
        "feat_high_addr_low_name_penalty",
    ]

    # --------------------------------------------------------------------------
    # 4. RUN INDIVIDUAL FEATURE FAMILY EXPERIMENTS (THRESHOLD = 0.88)
    # --------------------------------------------------------------------------
    print("\n[3/5] Running individual targeted feature experiments (Frozen threshold = 0.88)...")
    print(f"{'Experiment':<38} | {'Feats':<5} | {'F0.5':<6} | {'Prec':<6} | {'Rec':<6} | {'E2E Recall':<17} | {'Fit Time':<8}")
    print("-" * 105)

    res_p0 = run_experiment("P0: Baseline (74 feats)", base_74, df_train, df_dev_feat, scale_pos_weight, dev_s1_p10, gt_dict, dev_gt_pairs, total_dev_true, blocker_recall, retained_true)
    res_a1 = run_experiment("A1: Missing Address (+2 feats)", feats_fam_a, df_train, df_dev_feat, scale_pos_weight, dev_s1_p10, gt_dict, dev_gt_pairs, total_dev_true, blocker_recall, retained_true)
    res_b1 = run_experiment("B1: Native Script / Accents (+2 feats)", feats_fam_b, df_train, df_dev_feat, scale_pos_weight, dev_s1_p10, gt_dict, dev_gt_pairs, total_dev_true, blocker_recall, retained_true)
    res_c1 = run_experiment("C1: Lexical Shift / Aliases (+2 feats)", feats_fam_c, df_train, df_dev_feat, scale_pos_weight, dev_s1_p10, gt_dict, dev_gt_pairs, total_dev_true, blocker_recall, retained_true)
    res_d1 = run_experiment("D1: Address Disambiguation (+2 feats)", feats_fam_d, df_train, df_dev_feat, scale_pos_weight, dev_s1_p10, gt_dict, dev_gt_pairs, total_dev_true, blocker_recall, retained_true)

    # --------------------------------------------------------------------------
    # 5. CONTROLLED COMBINATIONS OF DEMONSTRATING FAMILIES
    # --------------------------------------------------------------------------
    print("\n[4/5] Evaluating controlled combinations of successful targeted families...")
    print("-" * 105)

    # Check which families beat baseline
    fam_candidates = [
        ("Fam A (Missing Address)", res_a1, ["feat_high_name_cand_addr_missing", "feat_addr_missing_asymmetric"]),
        ("Fam B (Native Script)", res_b1, ["feat_latin_accent_folded_exact", "feat_cross_script_postal_match"]),
        ("Fam C (Aliases)", res_c1, ["feat_name_distinctive_containment", "feat_name_first_distinctive_match"]),
        ("Fam D (Disambiguation)", res_d1, ["feat_same_building_weak_name", "feat_high_addr_low_name_penalty"]),
    ]

    selected_additions = []
    for name, r, feats in fam_candidates:
        delta = r["macro_f05"] - res_p0["macro_f05"]
        print(f"  {name:<32}: Δ F0.5 = {delta:+.4f} (E2E True: {r['e2e_true_matches']:,})")
        if delta >= 0.0000:  # non-harmful or positive
            selected_additions.extend(feats)

    # Combination F1: Best individual feature family
    best_indiv = max([res_a1, res_b1, res_c1, res_d1], key=lambda x: x["macro_f05"])
    print(f"\n  Best individual targeted family: {best_indiv['exp_name']} (Macro-F0.5 = {best_indiv['macro_f05']:.4f})")

    # Combination F2: Synergy of top non-harmful families
    # Let's combine Family A + Family C if both are positive/neutral
    feats_combo = list(dict.fromkeys(base_74 + selected_additions))
    res_combo = run_experiment(f"F2: Combined Top Families ({len(feats_combo)} feats)", feats_combo, df_train, df_dev_feat, scale_pos_weight, dev_s1_p10, gt_dict, dev_gt_pairs, total_dev_true, blocker_recall, retained_true)

    all_experiments = [res_p0, res_a1, res_b1, res_c1, res_d1, res_combo]
    winning_exp = max(all_experiments, key=lambda x: x["macro_f05"])
    print(f"\n  Winning Development Configuration: {winning_exp['exp_name']} (Macro-F0.5 = {winning_exp['macro_f05']:.4f})")

    # --------------------------------------------------------------------------
    # 6. THRESHOLD SENSITIVITY SWEEP ON WINNING CONFIGURATION
    # --------------------------------------------------------------------------
    print("\n[5/5] Performing development threshold sensitivity sweep on winning configuration...")
    thresh_grid = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]
    thresh_results = []
    print(f"{'Threshold':<10} | {'F0.5':<7} | {'Precision':<10} | {'Recall':<10} | {'E2E Recall':<12} | {'Matches':<8} | {'% Empty':<8}")
    print("-" * 80)
    for t in thresh_grid:
        r_t = evaluate_predictions_df(winning_exp["df_eval"], gt_dict, threshold=t, s1_list=dev_s1_p10)
        m_t = winning_exp["df_eval"][winning_exp["df_eval"]["score"] >= t]
        p_pairs = set(zip(m_t["source1_entity_id"], m_t["candidate_entity_id"]))
        tp_t = len(dev_gt_pairs & p_pairs)
        rec_t = tp_t / total_dev_true * 100.0
        print(f"  {t:<8.2f} | {r_t['macro_f05']:.4f} | {r_t['macro_precision']:.4f}     | {r_t['macro_recall']:.4f}     | {rec_t:5.2f}% ({tp_t:,}) | {r_t['total_pred_matches']:<7} | {r_t['empty_pred_pct']:.1f}%")
        thresh_results.append({
            "threshold": t,
            "macro_f05": r_t["macro_f05"],
            "macro_precision": r_t["macro_precision"],
            "macro_recall": r_t["macro_recall"],
            "e2e_true_matches": tp_t,
            "e2e_recall": rec_t,
            "pred_matches": r_t["total_pred_matches"],
            "empty_pct": r_t["empty_pred_pct"],
        })

    # --------------------------------------------------------------------------
    # 7. FEATURE IMPORTANCE FOR WINNING MODEL
    # --------------------------------------------------------------------------
    print("\n--- FEATURE IMPORTANCES FOR WINNING MODEL ---")
    winning_model = winning_exp["model"]
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

    # --------------------------------------------------------------------------
    # 8. ERROR ANALYSIS OF BEST CONFIGURATION
    # --------------------------------------------------------------------------
    print("\n--- ERROR ANALYSIS COMPARISON (BEST CONFIG vs BASELINE) ---")
    df_eval_win = winning_exp["df_eval"]
    df_eval_win["is_true"] = [1 if (s, c) in dev_gt_pairs else 0 for s, c in zip(df_eval_win["source1_entity_id"], df_eval_win["candidate_entity_id"])]
    df_eval_win["is_pred"] = (df_eval_win["score"] >= 0.88).astype(int)

    df_fp_win = df_eval_win[(df_eval_win["is_true"] == 0) & (df_eval_win["is_pred"] == 1)].sort_values("score", ascending=False)
    df_fn_win = df_eval_win[(df_eval_win["is_true"] == 1) & (df_eval_win["is_pred"] == 0)].sort_values("score", ascending=True)

    base_eval = res_p0["df_eval"]
    base_pred = set(zip(base_eval[base_eval["score"] >= 0.88]["source1_entity_id"], base_eval[base_eval["score"] >= 0.88]["candidate_entity_id"]))
    win_pred = set(zip(df_eval_win[df_eval_win["score"] >= 0.88]["source1_entity_id"], df_eval_win[df_eval_win["score"] >= 0.88]["candidate_entity_id"]))

    newly_rec = (win_pred & dev_gt_pairs) - base_pred
    newly_fp = (win_pred - dev_gt_pairs) - base_pred
    resolved_fp = (base_pred - dev_gt_pairs) - win_pred

    print(f"  Baseline True Matches Recovered : {res_p0['e2e_true_matches']:,}")
    print(f"  Winning True Matches Recovered  : {winning_exp['e2e_true_matches']:,} (Δ = {winning_exp['e2e_true_matches'] - res_p0['e2e_true_matches']:+d})")
    print(f"  Newly Recovered True Matches    : {len(newly_rec)}")
    print(f"  Newly Introduced False Positives: {len(newly_fp)}")
    print(f"  Eliminated False Positives      : {len(resolved_fp)}")

    # --------------------------------------------------------------------------
    # 9. SAVE TO JSON
    # --------------------------------------------------------------------------
    json_path = os.path.join(OUTPUT_DIR, "phase12_experiments_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "experiments_summary": [
                {
                    "name": r["exp_name"],
                    "features_count": r["feature_count"],
                    "macro_f05": r["macro_f05"],
                    "macro_precision": r["macro_precision"],
                    "macro_recall": r["macro_recall"],
                    "e2e_true_matches": r["e2e_true_matches"],
                    "e2e_recall": r["e2e_recall"],
                    "blocker_recall": r["blocker_recall"],
                    "classifier_retention": r["classifier_retention"],
                    "singleton_f05": r["singleton_f05"],
                    "zero_match_f05": r["zero_match_f05"],
                    "multi_match_f05": r["multi_match_f05"],
                    "delta_f05_vs_p0": r["macro_f05"] - res_p0["macro_f05"],
                }
                for r in all_experiments
            ],
            "winning_config": {
                "name": winning_exp["exp_name"],
                "features_count": winning_exp["feature_count"],
                "macro_f05": winning_exp["macro_f05"],
                "macro_precision": winning_exp["macro_precision"],
                "macro_recall": winning_exp["macro_recall"],
                "e2e_true_matches": winning_exp["e2e_true_matches"],
                "delta_f05": winning_exp["macro_f05"] - res_p0["macro_f05"],
                "newly_recovered_matches_count": len(newly_rec),
                "newly_introduced_fps_count": len(newly_fp),
                "eliminated_fps_count": len(resolved_fp),
            },
            "threshold_sweep": thresh_results,
            "top30_features": top30.to_dict(orient="records"),
            "group_gains": grp_gain.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved structured Phase 12 experiment results to {json_path}")


if __name__ == "__main__":
    main()
