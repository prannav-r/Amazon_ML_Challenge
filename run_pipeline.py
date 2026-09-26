"""
Amazon ML Challenge 2026: Business Entity Resolution
Production Inference & Training Pipeline (End-to-End Reproducibility)

Executes the approved frozen architecture:
1. [Optional / Auto-Fallback] Model Training:
   - If output/frozen_model_c1.pkl is missing or --train is passed, trains Model C1 on dataset/train.
2. Ingests raw test files (test_source1.tsv, test_source2.tsv, test_source3.tsv).
3. Multi-Channel Candidate Blocking (Channels A, A2, B, C, D, E, E2, G; Cap 150; src/blocking.py).
4. Canonical 65-Feature Extraction (src/feature_schema.py, src/features.py).
5. Frozen Model Scoring (Model C1: XGBoost 3.2.0, scale_pos_weight=5.84, seed=42; src/model.py).
6. Multi-Candidate Thresholding (tau* = 0.88).
7. Submission Output Formatting:
   - output/matching_results.tsv
   - output/candidate_pairs.tsv

Usage:
    # Standard inference (loads pre-trained frozen_model_c1.pkl):
    python run_pipeline.py --data-dir dataset/test --output-dir output

    # Full train-and-infer reproduction:
    python run_pipeline.py --train --train-dir dataset/train --data-dir dataset/test --output-dir output

    # Fast deterministic smoke test:
    python run_pipeline.py --smoke-test --limit 50
"""

import argparse
import os
import sys
import time
import pickle
from collections import defaultdict
import duckdb
import numpy as np
import pandas as pd

# Add local path for src module resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.blocking import CandidateBlocker
from src.features import PairwiseFeatureExtractor
from src.feature_schema import FEATURE_NAMES, get_feature_names
from src.model import EntityMatcherModel


def resolve_path(path: str, candidates: list) -> str:
    """Finds first existing path from candidates if path doesn't exist."""
    if os.path.exists(path):
        return path
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/test",
        help="Path to directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="dataset/train",
        help="Path to training data directory (used if --train is passed or model is missing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write matching_results.tsv and candidate_pairs.tsv",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="output/frozen_model_c1.pkl",
        help="Path to serialized Model C1 checkpoint (.pkl)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force retraining Model C1 from training data before running inference",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.88,
        help="Frozen decision threshold (default: 0.88)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=150,
        help="Priority blocker candidate cap per S1 (default: 150)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of S1 entities for smoke testing (e.g. 50)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run in smoke-test mode (deterministic 50 entities, safe verification)",
    )
    return parser.parse_args()


def train_frozen_model(train_dir: str, model_save_path: str, cap: int = 150):
    """
    Trains the frozen Model C1 (XGBoost 3.2.0, scale_pos_weight=5.84, seed=42)
    using the approved Phase 5 training pipeline.
    """
    print("\n" + "=" * 80)
    print("TRAINING FROZEN MODEL C1 FROM RAW TRAINING DATA")
    print("=" * 80)
    print(f"Training Data Directory : {train_dir}")
    print(f"Model Artifact Save Path: {model_save_path}")

    s1_train_path = os.path.join(train_dir, "train_source1.tsv").replace("\\", "/")
    s2_train_path = os.path.join(train_dir, "train_source2.tsv").replace("\\", "/")
    s3_train_path = os.path.join(train_dir, "train_source3.tsv").replace("\\", "/")
    gt_train_path = os.path.join(train_dir, "train_ground_truth.tsv").replace("\\", "/")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")

    print("[1/4] Loading training ground truth and sampling 5,000 S1 records (3k US, 2k India)...")
    con.execute(f"""
    CREATE OR REPLACE TABLE eval_s1 AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s1_train_path}', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE country = 'US'
    LIMIT 3000;
    """)
    con.execute(f"""
    INSERT INTO eval_s1
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s1_train_path}', delim='\\t', header=true, quote='', all_varchar=true)
    WHERE country = 'India'
    LIMIT 2000;
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE eval_cand_pool AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s2_train_path}', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s3_train_path}', delim='\\t', header=true, quote='', all_varchar=true);
    """)

    # Parse ground truth
    df_gt = pd.read_csv(gt_train_path, sep="\t")
    gt_dict = {}
    for _, row in df_gt.iterrows():
        s1 = str(row["source1_entity_id"])
        mids = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        gt_dict[s1] = set(mids.split(",")) if mids else set()

    # Blocking
    print(f"[2/4] Generating candidate pairs with Priority Cap = {cap}...")
    blocker = CandidateBlocker(con)
    block_res = blocker.generate_candidates(
        s1_table_or_path="eval_s1",
        cand_table_or_path="eval_cand_pool",
        output_table="train_candidates",
        max_candidates_per_s1=cap,
    )
    df_cands = con.execute("SELECT * FROM train_candidates").fetchdf()
    print(f"  Generated {len(df_cands):,} candidate pairs.")

    # Build lookups
    s1_rows = con.execute("SELECT entity_id, country, business_name, business_address FROM eval_s1").fetchall()
    s1_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in s1_rows}

    cand_rows = con.execute("""
    SELECT entity_id, country, business_name, business_address 
    FROM eval_cand_pool 
    WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM train_candidates);
    """).fetchall()
    cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in cand_rows}

    # Extract 65 features
    print("[3/4] Extracting canonical 65 features...")
    extractor = PairwiseFeatureExtractor()
    df_features = extractor.extract_features(df_cands, s1_lookup, cand_lookup)

    # Attach labels
    labels = []
    for _, r in df_features.iterrows():
        s1 = r["source1_entity_id"]
        cid = r["candidate_entity_id"]
        labels.append(1 if cid in gt_dict.get(s1, set()) else 0)
    df_features["match_label"] = labels

    # S1 stratified train split (60% train)
    strata = defaultdict(list)
    for s1, info in s1_lookup.items():
        c = info["country"]
        m_cnt = len(gt_dict.get(s1, set()))
        m_type = "zero" if m_cnt == 0 else ("single" if m_cnt == 1 else "multi")
        strata[(c, m_type)].append(s1)

    train_s1 = []
    rng = np.random.RandomState(42)
    for _, ids in sorted(strata.items()):
        shuffled = rng.permutation(ids)
        train_s1.extend(shuffled[: int(len(shuffled) * 0.60)])
    train_s1_set = set(train_s1)

    df_train = df_features[df_features["source1_entity_id"].isin(train_s1_set)].copy()
    X_train = df_train[list(FEATURE_NAMES)].values
    y_train = df_train["match_label"].values

    train_pos = int(df_train["match_label"].sum())
    train_neg = int(len(df_train) - train_pos)
    imbalance = train_neg / max(1, train_pos)
    scale_weight = float(np.sqrt(imbalance))

    # Fit Model C1
    print(f"[4/4] Fitting Model C1 (n_estimators=300, max_depth=5, lr=0.08, scale_pos_weight={scale_weight:.2f})...")
    model = EntityMatcherModel(
        model_type="xgboost_weighted",
        params={
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 5,
            "gamma": 0.10,
            "reg_alpha": 0.10,
            "reg_lambda": 1.00,
            "scale_pos_weight": scale_weight,
        },
        random_state=42,
        feature_names=list(FEATURE_NAMES),
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    model.save(model_save_path)
    print(f"Model successfully saved to {model_save_path}!")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    start_total = time.time()

    # Resolve data dir
    data_dir = resolve_path(
        args.data_dir,
        [
            os.path.join(CURRENT_DIR, args.data_dir),
            os.path.join(CURRENT_DIR, "student_resource", "dataset", "test"),
            os.path.join(CURRENT_DIR, "dataset", "test"),
            "student_resource/dataset/test",
        ],
    )

    # Resolve train dir
    train_dir = resolve_path(
        args.train_dir,
        [
            os.path.join(CURRENT_DIR, args.train_dir),
            os.path.join(CURRENT_DIR, "student_resource", "dataset", "train"),
            os.path.join(CURRENT_DIR, "dataset", "train"),
            "student_resource/dataset/train",
        ],
    )

    # Resolve model path
    model_path = resolve_path(
        args.model_path,
        [
            os.path.join(CURRENT_DIR, args.model_path),
            os.path.join(CURRENT_DIR, "output", "frozen_model_c1.pkl"),
            "output/frozen_model_c1.pkl",
        ],
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    limit = 50 if args.smoke_test and args.limit is None else args.limit

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 80)
    print(f"Data Directory     : {data_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Model Path         : {model_path}")
    print(f"Decision Threshold : tau* = {args.threshold}")
    print(f"Blocker Cap        : {args.cap}")
    print(f"Limit (S1 Records) : {limit if limit else 'ALL (Full Test Set)'}")
    print("=" * 80)

    # Check if model exists or training requested
    if args.train or not os.path.exists(model_path):
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"Model checkpoint '{model_path}' not found and training directory '{train_dir}' is missing!"
            )
        train_frozen_model(train_dir=train_dir, model_save_path=model_path, cap=args.cap)

    # 1. Connect DuckDB in-memory engine
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")

    s1_path = os.path.join(data_dir, "test_source1.tsv").replace("\\", "/")
    s2_path = os.path.join(data_dir, "test_source2.tsv").replace("\\", "/")
    s3_path = os.path.join(data_dir, "test_source3.tsv").replace("\\", "/")

    print("[1/5] Ingesting source files into DuckDB...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    con.execute(f"""
    CREATE OR REPLACE TABLE cur_s1 AS 
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('{s1_path}', delim='\\t', header=true, quote='', all_varchar=true)
    {limit_clause};
    """)
    n_s1 = con.execute("SELECT count(*) FROM cur_s1").fetchone()[0]
    print(f"  Loaded {n_s1:,} Source 1 records.")

    con.execute(f"""
    CREATE OR REPLACE TABLE cur_cand_pool AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s2_path}', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s3_path}', delim='\\t', header=true, quote='', all_varchar=true);
    """)
    n_cands_pool = con.execute("SELECT count(*) FROM cur_cand_pool").fetchone()[0]
    print(f"  Loaded {n_cands_pool:,} Source 2 and Source 3 candidate records.")

    # 2. Candidate Blocking
    print("[2/5] Executing multi-channel blocking (Channels A, A2, B, C, D, E, E2, G; Cap 150)...")
    blocker = CandidateBlocker(con)
    t0_block = time.time()
    block_res = blocker.generate_candidates(
        s1_table_or_path="cur_s1",
        cand_table_or_path="cur_cand_pool",
        output_table="cur_candidates",
        max_candidates_per_s1=args.cap,
    )
    t_block = time.time() - t0_block
    n_candidates = block_res["total_candidates"]
    print(f"  Blocking generated {n_candidates:,} candidate pairs in {t_block:.2f}s (avg {n_candidates/max(1, n_s1):.1f}/S1).")

    # 3. Feature Extraction
    print("[3/5] Extracting canonical 65 features...")
    df_cands = con.execute("SELECT * FROM cur_candidates").fetchdf()

    s1_rows = con.execute("SELECT entity_id, country, business_name, business_address FROM cur_s1").fetchall()
    s1_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in s1_rows}

    cand_rows = con.execute("""
    SELECT entity_id, country, business_name, business_address 
    FROM cur_cand_pool 
    WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM cur_candidates);
    """).fetchall()
    cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in cand_rows}

    extractor = PairwiseFeatureExtractor()
    t0_feat = time.time()
    if len(df_cands) > 0:
        df_features = extractor.extract_features(df_cands, s1_lookup, cand_lookup)
    else:
        df_features = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + list(FEATURE_NAMES))
    t_feat = time.time() - t0_feat
    print(f"  Extracted 65 features for {len(df_features):,} pairs in {t_feat:.2f}s.")

    # 4. Supervised Model Scoring
    print("[4/5] Loading frozen XGBoost C1 model and scoring...")
    model = EntityMatcherModel.load(model_path)
    if len(df_features) > 0:
        X = df_features[list(FEATURE_NAMES)].values
        probs = model.predict_proba(X)
        df_features["score"] = probs
        df_features["is_match"] = (probs >= args.threshold).astype(int)
        n_matches = int(df_features["is_match"].sum())
    else:
        df_features["score"] = []
        df_features["is_match"] = []
        n_matches = 0

    print(f"  Scored {len(df_features):,} pairs | Predicted matches: {n_matches:,} (threshold >= {args.threshold})")

    # 5. Export Submission TSVs
    print("[5/5] Formatting and exporting submission TSV files...")
    all_s1_ids = [r[0] for r in con.execute("SELECT entity_id FROM cur_s1 ORDER BY entity_id").fetchall()]
    
    cand_grouped = df_features.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict() if len(df_features) > 0 else {}
    match_grouped = df_features[df_features["is_match"] == 1].groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict() if len(df_features) > 0 else {}

    matching_rows = []
    candidate_rows = []
    for s1 in all_s1_ids:
        c_str = ",".join(cand_grouped.get(s1, []))
        m_str = ",".join(match_grouped.get(s1, []))
        matching_rows.append((s1, m_str))
        candidate_rows.append((s1, c_str))

    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    pd.DataFrame(matching_rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(matching_file, sep="\t", index=False)
    pd.DataFrame(candidate_rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(candidate_file, sep="\t", index=False)

    print(f"  Wrote: {matching_file} ({len(matching_rows):,} rows)")
    print(f"  Wrote: {candidate_file} ({len(candidate_rows):,} rows)")
    print(f"Pipeline completed in {time.time() - start_total:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
