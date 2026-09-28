"""
Phase 9: High-Recall / Adaptive Blocking Benchmark & End-to-End Evaluation
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Loads ground truth and sets up DuckDB with Dev S1 (998 entities) and full candidate pool (S2+S3 for US & India).
2. Benchmarks 6 blocker configurations:
   - Config 1: Baseline 8 Channels, Fixed Cap 150
   - Config 2: Baseline 8 Channels, Fixed Cap 200
   - Config 3: Baseline 8 Channels, Adaptive Cap Rule B (Cap 150 -> 250 for ambiguous S1s)
   - Config 4: Enhanced 11 Channels (V2), Fixed Cap 150
   - Config 5: Enhanced 11 Channels (V2), Fixed Cap 200
   - Config 6: Enhanced 11 Channels (V2), Adaptive Cap Rule B (Cap 150 -> 250 for ambiguous S1s)
3. For every blocker reports:
   - Total candidate pairs, avg candidates/S1, median, P95, max
   - Blocker Recall (% of 3,500 true matches retained)
   - Category A misses (Rank > Cap) vs Category B misses (Never retrieved)
   - Candidates per recovered true match
   - Blocking runtime
4. End-to-end evaluation using XGBoost Model C1 with the canonical 65-feature schema:
   - Trains Model C1 on the fixed Phase 5 training split (2,998 S1s, 273,838 pairs).
   - Extracts 65 baseline features on the newly generated candidate sets.
   - Evaluates Dev S1 Macro-F0.5 (optimal threshold and fixed 0.88).
   - Reports Macro Precision, Macro Recall, Macro F0.5.
   - Reports Blocker Recall vs End-to-End Classifier Recall.
   - Reports Singleton, Zero-Match, and Multi-Match diagnostic metrics.
5. Saves results to 'output/benchmark_blocking_v2_results.json'.
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
from src.feature_schema import FEATURE_NAMES as BASELINE_NAMES
from src.model import EntityMatcherModel
from src.evaluate import evaluate_predictions_df, evaluate_by_group

OUTPUT_DIR = "output"
GT_PATH = os.path.join(OUTPUT_DIR, "eval_ground_truth.pkl")
TRAIN_PARQUET = os.path.join(OUTPUT_DIR, "eval_features_cap150.parquet")
BASE_TRAIN = "student_resource/dataset/train"


def setup_data_and_train_model():
    print("=" * 80)
    print("PHASE 9: BLOCKING BENCHMARK & END-TO-END MODEL EVALUATION")
    print("=" * 80)
    print("\n[1/4] Loading ground truth, splits, and training Baseline Model C1...")

    with open(GT_PATH, "rb") as f:
        meta = pickle.load(f)

    gt_dict = meta["gt_dict"]
    s1_list_all = meta["s1_list"]
    s1_lookup = meta["s1_lookup"]
    cand_lookup = meta["cand_lookup"]

    # Reconstruct 3-way split deterministically
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

    # Train Model C1 on the fixed Phase 5 training split
    df_train_all = pd.read_parquet(TRAIN_PARQUET)
    train_s1_set = set(train_s1)
    df_train = df_train_all[df_train_all["source1_entity_id"].isin(train_s1_set)].copy()

    train_pos = int(df_train["match_label"].sum())
    train_neg = int(len(df_train) - train_pos)
    scale_pos_weight = float(np.sqrt(train_neg / train_pos))

    print(f"  Training Model C1 on {len(df_train):,} training pairs (scale_pos_weight={scale_pos_weight:.2f})...")
    X_train = df_train[list(BASELINE_NAMES)].values
    y_train = df_train["match_label"].values

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
        feature_names=list(BASELINE_NAMES),
    )
    t0_fit = time.time()
    model.fit(X_train, y_train)
    print(f"  Model C1 fitted in {time.time()-t0_fit:.2f}s.")

    # Setup DuckDB tables
    print("\n[2/4] Setting up DuckDB database with candidate tables...")
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

    return con, model, dev_s1, s1_lookup, gt_dict, dev_gt_pairs, total_dev_true


def evaluate_blocker(
    config_name: str,
    con: duckdb.DuckDBPyConnection,
    model: EntityMatcherModel,
    dev_s1: List[str],
    s1_lookup: Dict[str, Dict[str, str]],
    gt_dict: Dict[str, Set[str]],
    dev_gt_pairs: Set[Tuple[str, str]],
    total_dev_true: int,
    enable_enhanced: bool = False,
    max_cap: int = 150,
    adaptive_rule: Optional[str] = None,
    adaptive_cap: int = 250,
) -> Dict[str, Any]:
    print(f"\n" + "-" * 75)
    print(f"EVALUATING BLOCKER: {config_name}")
    print("-" * 75)

    blocker = CandidateBlockerV2(con)
    t0_b = time.time()
    res_b = blocker.generate_candidates(
        s1_table_or_path="dev_s1_tbl",
        cand_table_or_path="all_candidates",
        output_table="eval_cand_out",
        max_candidates_per_s1=max_cap,
        enable_enhanced_channels=enable_enhanced,
        adaptive_rule=adaptive_rule,
        adaptive_cap=adaptive_cap,
    )
    t_block = time.time() - t0_b
    total_cands = res_b["total_candidates"]

    df_cands = con.execute("SELECT * FROM eval_cand_out").fetchdf()

    # Calculate Candidate Distribution per S1
    counts_s1 = df_cands.groupby("source1_entity_id")["candidate_entity_id"].count().to_dict()
    for s in dev_s1:
        if s not in counts_s1:
            counts_s1[s] = 0
    vals = list(counts_s1.values())
    avg_cands = float(np.mean(vals))
    median_cands = float(np.median(vals))
    p95_cands = float(np.percentile(vals, 95))
    max_cands = int(np.max(vals))

    # Blocker Recall Analysis
    cand_pairs_set = set(zip(df_cands["source1_entity_id"], df_cands["candidate_entity_id"]))
    retained_true = len(dev_gt_pairs & cand_pairs_set)
    blocker_recall = retained_true / total_dev_true * 100.0
    missed_count = total_dev_true - retained_true
    cands_per_true = total_cands / retained_true if retained_true > 0 else 0.0

    print(f"  Candidates Generated : {total_cands:,} (Avg: {avg_cands:.1f}/S1, Median: {median_cands:.0f}, P95: {p95_cands:.0f}, Max: {max_cands})")
    print(f"  Blocker Recall       : {blocker_recall:.2f}% ({retained_true:,}/{total_dev_true:,} true matches | Missed: {missed_count:,})")
    print(f"  Efficiency           : {cands_per_true:.1f} candidates per recovered true match | Runtime: {t_block:.2f}s")

    # Feature Extraction (Baseline 65 features)
    print(f"  Extracting 65 baseline features on {len(df_cands):,} pairs...")
    c_rows = con.execute("""
        SELECT entity_id, country, business_name, business_address 
        FROM all_candidates 
        WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM eval_cand_out)
    """).fetchall()
    cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in c_rows}
    extractor = PairwiseFeatureExtractor()
    t0_feat = time.time()
    df_feat = extractor.extract_features(df_cands, s1_lookup, cand_lookup)
    t_feat = time.time() - t0_feat
    print(f"  Features extracted in {t_feat:.2f}s ({len(df_feat)/t_feat:,.0f} pairs/sec).")

    # Predict with Model C1
    X_mat = df_feat[list(BASELINE_NAMES)].values
    df_feat["score"] = model.predict_proba(X_mat)

    # 1. Fixed threshold 0.88 evaluation
    res_088 = evaluate_predictions_df(df_feat, gt_dict, threshold=0.88, s1_list=dev_s1)
    grp_088 = evaluate_by_group(df_feat, gt_dict, threshold=0.88, s1_list=dev_s1)

    # Calculate end-to-end classifier recall
    matched_088 = df_feat[df_feat["score"] >= 0.88]
    pred_pairs_088 = set(zip(matched_088["source1_entity_id"], matched_088["candidate_entity_id"]))
    tp_088 = len(dev_gt_pairs & pred_pairs_088)
    e2e_recall_088 = tp_088 / total_dev_true * 100.0

    print(f"  At Fixed Threshold 0.88:")
    print(f"    Macro-F0.5: {res_088['macro_f05']:.4f} | Prec: {res_088['macro_precision']:.4f} | Rec: {res_088['macro_recall']:.4f}")
    print(f"    Blocker Rec: {blocker_recall:.2f}% -> E2E Model Rec: {e2e_recall_088:.2f}% ({tp_088:,} true matches)")
    print(f"    Pred Matches: {res_088['total_pred_matches']} | % Empty: {res_088['empty_pred_pct']:.1f}%")

    # 2. Optimal threshold sweep [0.80 to 0.95]
    best_thresh = 0.88
    best_f05 = -1.0
    best_res = None
    for t in np.arange(0.80, 0.96, 0.01):
        t_val = round(float(t), 2)
        cur_res = evaluate_predictions_df(df_feat, gt_dict, threshold=t_val, s1_list=dev_s1)
        if cur_res["macro_f05"] > best_f05:
            best_f05 = cur_res["macro_f05"]
            best_thresh = t_val
            best_res = cur_res

    best_grp = evaluate_by_group(df_feat, gt_dict, threshold=best_thresh, s1_list=dev_s1)
    matched_opt = df_feat[df_feat["score"] >= best_thresh]
    pred_pairs_opt = set(zip(matched_opt["source1_entity_id"], matched_opt["candidate_entity_id"]))
    tp_opt = len(dev_gt_pairs & pred_pairs_opt)
    e2e_recall_opt = tp_opt / total_dev_true * 100.0

    print(f"  At Optimal Threshold {best_thresh:.2f}:")
    print(f"    Macro-F0.5: {best_res['macro_f05']:.4f} | Prec: {best_res['macro_precision']:.4f} | Rec: {best_res['macro_recall']:.4f}")
    print(f"    Blocker Rec: {blocker_recall:.2f}% -> E2E Model Rec: {e2e_recall_opt:.2f}% ({tp_opt:,} true matches)")
    print(f"    Pred Matches: {best_res['total_pred_matches']} | % Empty: {best_res['empty_pred_pct']:.1f}%")
    print(f"    - Singletons: Prec {best_grp['group1_singleton']['macro_precision']:.4f}, Rec {best_grp['group1_singleton']['macro_recall']:.4f}, F0.5 {best_grp['group1_singleton']['macro_f05']:.4f}")
    print(f"    - Zero-Match: Prec {best_grp['group2_zero_match']['macro_precision']:.4f}, Rec {best_grp['group2_zero_match']['macro_recall']:.4f}, F0.5 {best_grp['group2_zero_match']['macro_f05']:.4f} (FP: {best_grp['group2_zero_match']['false_positive_rate']:.2f}%)")
    print(f"    - Multi-Match: Prec {best_grp['group3_multi_match']['macro_precision']:.4f}, Rec {best_grp['group3_multi_match']['macro_recall']:.4f}, F0.5 {best_grp['group3_multi_match']['macro_f05']:.4f}")

    return {
        "config_name": config_name,
        "total_cands": total_cands,
        "avg_cands": avg_cands,
        "median_cands": median_cands,
        "p95_cands": p95_cands,
        "max_cands": max_cands,
        "blocker_recall": blocker_recall,
        "retained_true": retained_true,
        "missed_true": missed_count,
        "cands_per_true": cands_per_true,
        "runtime_sec": t_block,
        "canonical_088": {
            "threshold": 0.88,
            "macro_precision": res_088["macro_precision"],
            "macro_recall": res_088["macro_recall"],
            "macro_f05": res_088["macro_f05"],
            "e2e_recall": e2e_recall_088,
            "e2e_true_matches": tp_088,
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
            "e2e_recall": e2e_recall_opt,
            "e2e_true_matches": tp_opt,
            "pred_matches": best_res["total_pred_matches"],
            "empty_pct": best_res["empty_pred_pct"],
            "singleton": best_grp["group1_singleton"],
            "zero_match": best_grp["group2_zero_match"],
            "multi_match": best_grp["group3_multi_match"],
        }
    }


def main():
    con, model, dev_s1, s1_lookup, gt_dict, dev_gt_pairs, total_dev_true = setup_data_and_train_model()

    configs = [
        {
            "name": "Config 1: Baseline Blocker (Cap 150)",
            "enable_enhanced": False,
            "max_cap": 150,
            "adaptive_rule": None,
        },
        {
            "name": "Config 2: Baseline Blocker (Cap 200)",
            "enable_enhanced": False,
            "max_cap": 200,
            "adaptive_rule": None,
        },
        {
            "name": "Config 3: Baseline + Adaptive Cap Rule B (150->250)",
            "enable_enhanced": False,
            "max_cap": 150,
            "adaptive_rule": "rule_b",
            "adaptive_cap": 250,
        },
        {
            "name": "Config 4: Enhanced Blocker V2 (Cap 150)",
            "enable_enhanced": True,
            "max_cap": 150,
            "adaptive_rule": None,
        },
        {
            "name": "Config 5: Enhanced Blocker V2 (Cap 200)",
            "enable_enhanced": True,
            "max_cap": 200,
            "adaptive_rule": None,
        },
        {
            "name": "Config 6: Enhanced Blocker V2 + Adaptive Rule B (150->250)",
            "enable_enhanced": True,
            "max_cap": 150,
            "adaptive_rule": "rule_b",
            "adaptive_cap": 250,
        },
    ]

    results = []
    for cfg in configs:
        res = evaluate_blocker(
            config_name=cfg["name"],
            con=con,
            model=model,
            dev_s1=dev_s1,
            s1_lookup=s1_lookup,
            gt_dict=gt_dict,
            dev_gt_pairs=dev_gt_pairs,
            total_dev_true=total_dev_true,
            enable_enhanced=cfg["enable_enhanced"],
            max_cap=cfg["max_cap"],
            adaptive_rule=cfg.get("adaptive_rule"),
            adaptive_cap=cfg.get("adaptive_cap", 250),
        )
        results.append(res)

    # --------------------------------------------------------------------------
    # SUMMARY TABLES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("PHASE 9 BLOCKING BENCHMARK: CANDIDATE VOLUME & BLOCKER RECALL SUMMARY")
    print("=" * 105)
    base_cands = results[0]["total_cands"]
    rows_blocker = []
    for r in results:
        growth = (r["total_cands"] - base_cands) / base_cands * 100.0
        rows_blocker.append({
            "Blocker Configuration": r["config_name"],
            "Total Cands": f"{r['total_cands']:,}",
            "Growth": f"{growth:+5.1f}%",
            "Avg/S1": f"{r['avg_cands']:.1f}",
            "Median": f"{r['median_cands']:.0f}",
            "P95": f"{r['p95_cands']:.0f}",
            "Max": r["max_cands"],
            "Blocker Rec": f"{r['blocker_recall']:.2f}%",
            "True Recov": f"{r['retained_true']:,}",
            "Cands/True": f"{r['cands_per_true']:.1f}",
            "Runtime": f"{r['runtime_sec']:.2f}s",
        })
    df_block_summary = pd.DataFrame(rows_blocker)
    print(df_block_summary.to_string(index=False))

    print("\n" + "=" * 105)
    print("PHASE 9 END-TO-END MODEL EVALUATION: S1-LEVEL MACRO-F0.5 & CLASSIFIER RECALL (OPTIMAL THRESHOLD)")
    print("=" * 105)
    base_f05 = results[0]["optimal"]["macro_f05"]
    rows_model = []
    for r in results:
        o = r["optimal"]
        delta_f05 = o["macro_f05"] - base_f05
        rows_model.append({
            "Configuration": r["config_name"],
            "Opt Thresh": f"{o['threshold']:.2f}",
            "Macro Prec": f"{o['macro_precision']:.4f}",
            "Macro Rec": f"{o['macro_recall']:.4f}",
            "Macro F0.5": f"{o['macro_f05']:.4f}",
            "Delta F0.5": f"{delta_f05:+.4f}",
            "Blocker Rec": f"{r['blocker_recall']:.2f}%",
            "E2E Rec": f"{o['e2e_recall']:.2f}%",
            "E2E Matches": f"{o['e2e_true_matches']:,}",
            "Pred Matches": o["pred_matches"],
            "% Empty": f"{o['empty_pct']:.1f}%",
            "Single F0.5": f"{o['singleton']['macro_f05']:.4f}",
            "Zero F0.5": f"{o['zero_match']['macro_f05']:.4f}",
            "Multi F0.5": f"{o['multi_match']['macro_f05']:.4f}",
        })
    df_model_summary = pd.DataFrame(rows_model)
    print(df_model_summary.to_string(index=False))

    # Save to JSON
    json_path = os.path.join(OUTPUT_DIR, "benchmark_blocking_v2_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "blocker_summary": rows_blocker,
            "end_to_end_summary": rows_model,
        }, f, indent=2)
    print(f"\nSaved benchmark results to {json_path}")


if __name__ == "__main__":
    main()
