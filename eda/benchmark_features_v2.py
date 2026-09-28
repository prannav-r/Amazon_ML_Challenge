"""
Phase 8: Error-Driven Feature Engineering Benchmark & Ablation Study
Amazon ML Challenge 2026: Business Entity Resolution

This script:
1. Loads the precomputed 92-feature dataset ('output/eval_features_v2_cap150.parquet').
2. Uses strictly the Phase 5 stratified split:
   - Training Split (60%): 2,998 S1 entities (273,838 pairs).
   - Development Split (20%): 998 S1 entities (92,515 pairs).
   - Untouched Holdout (20%): 1,004 S1 entities -> LOCKED / NEVER TOUCHED.
3. Benchmarks 7 ablation configurations using XGBoost Model C1 (scale_pos_weight=5.84, seed=42):
   - Baseline: Original 65 canonical features
   - Experiment A: Baseline + Missing-Address Recovery (72 features)
   - Experiment B: Baseline + Alias & Trade Names (70 features)
   - Experiment C: Baseline + OCR & Typo Robustness (70 features)
   - Experiment D: Baseline + Cross-Script & Transliteration (69 features)
   - Experiment E: Baseline + Address Disambiguation & False Positives (71 features)
   - Experiment F: All New Error-Driven Features (92 features)
4. For every experiment reports:
   - Number of features, training pairs, validation S1 count
   - Macro Precision, Macro Recall, Macro F0.5
   - Threshold used (both at fixed 0.88 and optimal threshold)
   - Total predicted matches, % empty predictions
   - Singleton performance (P, R, F0.5)
   - Zero-match performance (P, R, F0.5, % perfect)
   - Multi-match performance (P, R, F0.5)
5. Extracts and analyzes Feature Importance for the best experiment:
   - Top 30 features by Gain
   - Gain importance per feature
   - Feature-group aggregated importance
"""

import sys
import os
import io
import time
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple
import numpy as np
import pandas as pd

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import EntityMatcherModel
from src.evaluate import evaluate_predictions_df, evaluate_by_group, evaluate_s1_macro
from src.feature_schema import FEATURE_NAMES as BASELINE_NAMES
from src.feature_schema_v2 import (
    FEATURE_SPECS_V2,
    FEATURE_NAMES_V2,
    EXPERIMENT_FEATURE_SETS,
    get_experiment_features,
)

OUTPUT_DIR = "output"
PARQUET_PATH = os.path.join(OUTPUT_DIR, "eval_features_v2_cap150.parquet")
GT_PATH = os.path.join(OUTPUT_DIR, "eval_ground_truth.pkl")


def load_data_and_splits():
    print("=" * 80)
    print("PHASE 8: ERROR-DRIVEN FEATURE ENGINEERING BENCHMARK (ABLATION STUDY)")
    print("=" * 80)
    print("\n[1/4] Loading precomputed V2 features and establishing S1-level split...")

    t0 = time.time()
    df_all = pd.read_parquet(PARQUET_PATH)
    with open(GT_PATH, "rb") as f:
        meta = pickle.load(f)
    print(f"  Loaded {len(df_all):,} candidate pairs with {df_all.shape[1]} columns in {time.time()-t0:.2f}s.")

    gt_dict = meta["gt_dict"]
    s1_list_all = meta["s1_list"]
    s1_lookup = meta["s1_lookup"]

    # Stratified 3-way split: exactly as in Phase 5.1
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

    train_s1 = []
    dev_s1 = []
    holdout_s1 = []

    rng = np.random.RandomState(42)
    for (c, m_type), ids in sorted(strata.items()):
        shuffled = rng.permutation(ids)
        n = len(shuffled)
        n_train = int(n * 0.60)
        n_dev = int(n * 0.20)
        train_s1.extend(shuffled[:n_train])
        dev_s1.extend(shuffled[n_train:n_train + n_dev])
        holdout_s1.extend(shuffled[n_train + n_dev:])

    # Strictly assert zero leakage and verify exact partition counts
    assert len(set(train_s1) & set(dev_s1)) == 0, "Leakage between Train and Dev!"
    assert len(set(train_s1) & set(holdout_s1)) == 0, "Leakage between Train and Holdout!"
    assert len(set(dev_s1) & set(holdout_s1)) == 0, "Leakage between Dev and Holdout!"
    assert len(train_s1) == 2998, f"Expected 2998 train S1s, got {len(train_s1)}"
    assert len(dev_s1) == 998, f"Expected 998 dev S1s, got {len(dev_s1)}"
    assert len(holdout_s1) == 1004, f"Expected 1004 holdout S1s, got {len(holdout_s1)}"

    print(f"  Train S1 Entities       : {len(train_s1):,} (60%)")
    print(f"  Development S1 Entities : {len(dev_s1):,} (20%)")
    print(f"  Holdout S1 Entities     : {len(holdout_s1):,} (20%) [LOCKED - NEVER EVALUATED]")

    train_s1_set = set(train_s1)
    dev_s1_set = set(dev_s1)

    df_train = df_all[df_all["source1_entity_id"].isin(train_s1_set)].copy()
    df_dev = df_all[df_all["source1_entity_id"].isin(dev_s1_set)].copy()

    train_pos = int(df_train["match_label"].sum())
    train_neg = int(len(df_train) - train_pos)
    imbalance = train_neg / train_pos
    mild_scale_pos_weight = float(np.sqrt(imbalance))

    print(f"  Training Pairs          : {len(df_train):,} ({train_pos:,} pos, {train_neg:,} neg | Imbalance: {imbalance:.2f}:1)")
    print(f"  Development Pairs       : {len(df_dev):,}")
    print(f"  scale_pos_weight        : {mild_scale_pos_weight:.2f}")

    return df_train, df_dev, train_s1, dev_s1, gt_dict, mild_scale_pos_weight


def run_experiment(
    exp_name: str,
    feature_list: List[str],
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    dev_s1: List[str],
    gt_dict: Dict[str, Set[str]],
    scale_pos_weight: float,
) -> Dict[str, Any]:
    print(f"\n" + "-" * 70)
    print(f"RUNNING EXPERIMENT: {exp_name} ({len(feature_list)} features)")
    print("-" * 70)

    X_train = df_train[feature_list].values
    y_train = df_train["match_label"].values
    X_dev = df_dev[feature_list].values

    # Initialize Model C1
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

    t0_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0_fit
    print(f"  Model trained in {fit_time:.2f}s.")

    # Predict probabilities on Dev
    df_dev_scored = df_dev[["source1_entity_id", "candidate_entity_id", "match_label"]].copy()
    probs = model.predict_proba(X_dev)
    df_dev_scored["score"] = probs

    # 1. Evaluate at canonical baseline threshold 0.88
    res_canonical = evaluate_predictions_df(df_dev_scored, gt_dict, threshold=0.88, s1_list=dev_s1)
    grp_canonical = evaluate_by_group(df_dev_scored, gt_dict, threshold=0.88, s1_list=dev_s1)

    print(f"  At Fixed Threshold 0.88:")
    print(f"    Macro-F0.5: {res_canonical['macro_f05']:.4f} | Prec: {res_canonical['macro_precision']:.4f} | Rec: {res_canonical['macro_recall']:.4f} | Pred Matches: {res_canonical['total_pred_matches']} | % Empty: {res_canonical['empty_pred_pct']:.1f}%")

    # 2. Sweep thresholds [0.80 to 0.95] to find optimal threshold
    best_thresh = 0.88
    best_f05 = -1.0
    best_res = None
    best_grp = None

    for t in np.arange(0.80, 0.96, 0.01):
        t_val = round(float(t), 2)
        cur_res = evaluate_predictions_df(df_dev_scored, gt_dict, threshold=t_val, s1_list=dev_s1)
        if cur_res["macro_f05"] > best_f05:
            best_f05 = cur_res["macro_f05"]
            best_thresh = t_val
            best_res = cur_res

    best_grp = evaluate_by_group(df_dev_scored, gt_dict, threshold=best_thresh, s1_list=dev_s1)

    print(f"  At Optimal Threshold {best_thresh:.2f}:")
    print(f"    Macro-F0.5: {best_res['macro_f05']:.4f} | Prec: {best_res['macro_precision']:.4f} | Rec: {best_res['macro_recall']:.4f} | Pred Matches: {best_res['total_pred_matches']} | % Empty: {best_res['empty_pred_pct']:.1f}%")
    print(f"    - Singletons: Prec {best_grp['group1_singleton']['macro_precision']:.4f}, Rec {best_grp['group1_singleton']['macro_recall']:.4f}, F0.5 {best_grp['group1_singleton']['macro_f05']:.4f}")
    print(f"    - Zero-Match: Prec {best_grp['group2_zero_match']['macro_precision']:.4f}, Rec {best_grp['group2_zero_match']['macro_recall']:.4f}, F0.5 {best_grp['group2_zero_match']['macro_f05']:.4f} (FP Rate: {best_grp['group2_zero_match']['false_positive_rate']:.2f}%)")
    print(f"    - Multi-Match: Prec {best_grp['group3_multi_match']['macro_precision']:.4f}, Rec {best_grp['group3_multi_match']['macro_recall']:.4f}, F0.5 {best_grp['group3_multi_match']['macro_f05']:.4f}")

    # Extract feature importances
    df_imp_raw = model.get_feature_importances()
    df_importance = pd.DataFrame({
        "feature_name": df_imp_raw["feature"],
        "gain": df_imp_raw["gain_importance"],
        "gain_pct": df_imp_raw["gain_importance"] * 100.0,
    })

    return {
        "exp_name": exp_name,
        "num_features": len(feature_list),
        "train_pairs": len(df_train),
        "val_s1_count": len(dev_s1),
        "canonical_088": {
            "threshold": 0.88,
            "macro_precision": res_canonical["macro_precision"],
            "macro_recall": res_canonical["macro_recall"],
            "macro_f05": res_canonical["macro_f05"],
            "predicted_matches": res_canonical["total_pred_matches"],
            "empty_pct": res_canonical["empty_pred_pct"],
            "singleton": grp_canonical["group1_singleton"],
            "zero_match": grp_canonical["group2_zero_match"],
            "multi_match": grp_canonical["group3_multi_match"],
        },
        "optimal": {
            "threshold": best_thresh,
            "macro_precision": best_res["macro_precision"],
            "macro_recall": best_res["macro_recall"],
            "macro_f05": best_res["macro_f05"],
            "predicted_matches": best_res["total_pred_matches"],
            "empty_pct": best_res["empty_pred_pct"],
            "singleton": best_grp["group1_singleton"],
            "zero_match": best_grp["group2_zero_match"],
            "multi_match": best_grp["group3_multi_match"],
        },
        "fit_time": fit_time,
        "importance_df": df_importance,
        "model": model,
    }


def main():
    df_train, df_dev, train_s1, dev_s1, gt_dict, scale_pos_weight = load_data_and_splits()

    experiments = [
        ("Baseline", list(BASELINE_NAMES)),
        ("Exp_A_MissingAddr", get_experiment_features("Exp_A_MissingAddr")),
        ("Exp_B_Aliases", get_experiment_features("Exp_B_Aliases")),
        ("Exp_C_TypoOCR", get_experiment_features("Exp_C_TypoOCR")),
        ("Exp_D_CrossScript", get_experiment_features("Exp_D_CrossScript")),
        ("Exp_E_AddressDisambig", get_experiment_features("Exp_E_AddressDisambig")),
        ("Exp_F_AllNew", get_experiment_features("Exp_F_AllNew")),
    ]

    results = {}
    for exp_name, feat_list in experiments:
        res = run_experiment(
            exp_name=exp_name,
            feature_list=feat_list,
            df_train=df_train,
            df_dev=df_dev,
            dev_s1=dev_s1,
            gt_dict=gt_dict,
            scale_pos_weight=scale_pos_weight,
        )
        results[exp_name] = res

    # --------------------------------------------------------------------------
    # SUMMARY REPORT TABLES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PHASE 8 ABLATION EXPERIMENT SUMMARY (EVALUATED AT CANONICAL THRESHOLD 0.88)")
    print("=" * 90)
    rows_088 = []
    for exp_name, res in results.items():
        c = res["canonical_088"]
        rows_088.append({
            "Experiment": exp_name,
            "Features": res["num_features"],
            "Threshold": c["threshold"],
            "Macro Prec": f"{c['macro_precision']:.4f}",
            "Macro Rec": f"{c['macro_recall']:.4f}",
            "Macro F0.5": f"{c['macro_f05']:.4f}",
            "Delta F0.5": f"{c['macro_f05'] - results['Baseline']['canonical_088']['macro_f05']:+.4f}",
            "Pred Matches": c["predicted_matches"],
            "% Empty": f"{c['empty_pct']:.1f}%",
            "Single F0.5": f"{c['singleton']['macro_f05']:.4f}",
            "Zero F0.5": f"{c['zero_match']['macro_f05']:.4f}",
            "Multi F0.5": f"{c['multi_match']['macro_f05']:.4f}",
        })
    df_summary_088 = pd.DataFrame(rows_088)
    print(df_summary_088.to_string(index=False))

    print("\n" + "=" * 90)
    print("PHASE 8 ABLATION EXPERIMENT SUMMARY (EVALUATED AT OPTIMAL THRESHOLD)")
    print("=" * 90)
    rows_opt = []
    for exp_name, res in results.items():
        o = res["optimal"]
        rows_opt.append({
            "Experiment": exp_name,
            "Features": res["num_features"],
            "Opt Thresh": f"{o['threshold']:.2f}",
            "Macro Prec": f"{o['macro_precision']:.4f}",
            "Macro Rec": f"{o['macro_recall']:.4f}",
            "Macro F0.5": f"{o['macro_f05']:.4f}",
            "Delta F0.5": f"{o['macro_f05'] - results['Baseline']['optimal']['macro_f05']:+.4f}",
            "Pred Matches": o["predicted_matches"],
            "% Empty": f"{o['empty_pct']:.1f}%",
            "Single F0.5": f"{o['singleton']['macro_f05']:.4f}",
            "Zero F0.5": f"{o['zero_match']['macro_f05']:.4f}",
            "Multi F0.5": f"{o['multi_match']['macro_f05']:.4f}",
        })
    df_summary_opt = pd.DataFrame(rows_opt)
    print(df_summary_opt.to_string(index=False))

    # --------------------------------------------------------------------------
    # BEST EXPERIMENT SELECTION & FEATURE IMPORTANCE
    # --------------------------------------------------------------------------
    best_exp_name = max(results.keys(), key=lambda k: results[k]["optimal"]["macro_f05"])
    best_exp = results[best_exp_name]
    print("\n" + "=" * 90)
    print(f"BEST PERFORMING CONFIGURATION: {best_exp_name}")
    print(f"Optimal Macro-F0.5: {best_exp['optimal']['macro_f05']:.4f} at threshold {best_exp['optimal']['threshold']:.2f}")
    print("=" * 90)

    # Feature spec lookup
    spec_dict = {s.name: s for s in FEATURE_SPECS_V2}

    df_imp = best_exp["importance_df"].copy()
    df_imp["group"] = df_imp["feature_name"].map(lambda f: spec_dict[f].group if f in spec_dict else "Unknown")

    print("\n--- TOP 30 FEATURES BY GAIN IMPORTANCE (BEST EXPERIMENT: " + best_exp_name + ") ---")
    top30 = df_imp.head(30)[["feature_name", "gain", "gain_pct", "group"]].copy()
    top30["gain"] = top30["gain"].map(lambda x: f"{x:.2f}")
    top30["gain_pct"] = top30["gain_pct"].map(lambda x: f"{x:.2f}%")
    print(top30.to_string(index=True))

    # Group aggregated importance
    print("\n--- FEATURE GROUP AGGREGATED GAIN IMPORTANCE ---")
    group_imp = df_imp.groupby("group")["gain_pct"].sum().reset_index()
    group_imp = group_imp.sort_values("gain_pct", ascending=False).reset_index(drop=True)
    group_imp["gain_pct"] = group_imp["gain_pct"].map(lambda x: f"{x:.2f}%")
    print(group_imp.to_string(index=False))

    # Save benchmark results to json for reporting
    save_data = {
        "summary_canonical_088": rows_088,
        "summary_optimal": rows_opt,
        "best_experiment": best_exp_name,
        "top30_features": top30.to_dict(orient="records"),
        "group_importance": group_imp.to_dict(orient="records"),
    }
    with open(os.path.join(OUTPUT_DIR, "benchmark_v2_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved benchmark results to {os.path.join(OUTPUT_DIR, 'benchmark_v2_results.json')}")


if __name__ == "__main__":
    main()
