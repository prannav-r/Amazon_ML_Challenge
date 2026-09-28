"""
Train Frozen Config D1 Model on Full Permitted Training Data
Amazon ML Challenge 2026: Business Entity Resolution

Trains Model C1 (76 Features, scale_pos_weight=5.8307, Hist, Depth=5, LR=0.08, n_estimators=300)
on all 5,000 permitted training S1 records (456,432 candidate pairs).
Saves to output/frozen_model_d1.pkl.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb

sys.path.insert(0, ".")
from src.feature_schema import FEATURE_NAMES as CANONICAL_65
from src.feature_schema_v2 import SELECTED_PHASE8_FEATURES
from src.model import EntityMatcherModel

def main():
    print("=" * 80)
    print("TRAINING FROZEN MODEL D1 ON FULL TRAINING DATA")
    print("=" * 80)
    
    t0 = time.time()
    
    # 1. Load full training dataset
    parquet_path = "output/eval_features_v2_cap150.parquet"
    print(f"Loading {parquet_path}...")
    df_train = pd.read_parquet(parquet_path)
    
    gt_path = "output/eval_ground_truth.pkl"
    with open(gt_path, "rb") as f:
        gt_meta = pickle.load(f)
    s1_list = gt_meta["s1_list"]
    s1_lookup = gt_meta["s1_lookup"]
    gt_dict = gt_meta["gt_dict"]
    
    # 2. Features
    group_k = ["feat_same_building_weak_name", "feat_high_addr_low_name_penalty"]
    features_76 = list(CANONICAL_65) + list(SELECTED_PHASE8_FEATURES) + group_k
    assert len(features_76) == 76
    
    pos_pairs = int(df_train["match_label"].sum())
    neg_pairs = int(len(df_train) - pos_pairs)
    spw = float(np.sqrt(neg_pairs / pos_pairs))
    
    print(f"Total S1 entities   : {len(s1_list):,}")
    print(f"Total candidate pairs: {len(df_train):,}")
    print(f"Positive pairs      : {pos_pairs:,}")
    print(f"Negative pairs      : {neg_pairs:,}")
    print(f"scale_pos_weight    : {spw:.4f}")
    
    countries = Counter(s1_lookup[s]["country"] for s in s1_list)
    print(f"Countries           : {dict(countries)}")
    
    mtypes = Counter(
        "zero" if len(gt_dict.get(s, set())) == 0 else ("single" if len(gt_dict.get(s, set())) == 1 else "multi")
        for s in s1_list
    )
    print(f"Match types         : {dict(mtypes)}")
    
    X = df_train[features_76].values
    y = df_train["match_label"].values
    
    # 3. Model parameters
    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 5,
        "gamma": 0.10,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "random_state": 42,
    }
    
    print("\nFitting XGBoost C1 on full training dataset...")
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y)
    
    model_wrapper = EntityMatcherModel(
        model_type="xgboost_weighted",
        params=params,
        random_state=42,
        feature_names=features_76,
    )
    model_wrapper.model = clf
    model_wrapper.is_fitted = True
    model_wrapper.training_time_sec = time.time() - t0
    
    # Save to output/frozen_model_d1.pkl and code/business_entity_resolution/output/frozen_model_d1.pkl
    out_paths = [
        "output/frozen_model_d1.pkl",
        "code/business_entity_resolution/output/frozen_model_d1.pkl"
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        model_wrapper.save(p)
        print(f"Saved frozen model checkpoint to: {p}")
        
    print(f"\nTraining completed in {time.time() - t0:.2f}s.")
    print("=" * 80)

if __name__ == "__main__":
    main()
