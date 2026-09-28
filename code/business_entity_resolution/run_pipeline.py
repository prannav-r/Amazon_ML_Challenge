"""
Amazon ML Challenge 2026: Business Entity Resolution
Production Inference & Training Pipeline (Config D1: 76 Features, Cap 200, Tau=0.88)

Executes the frozen Config D1 architecture:
1. Model Training (Optional / Auto-Fallback):
   - Trains Model C1 on full training data using 76 features and scale_pos_weight=5.83.
   - Serializes model to output/frozen_model_d1.pkl.
2. Ingests raw test files (test_source1.tsv, test_source2.tsv, test_source3.tsv).
3. Multi-Channel Candidate Blocking (Enhanced Blocker V2: 11 channels; Cap 200).
4. Frozen 76-Feature Extraction (65 Canonical + 9 Selected Phase 8 + 2 Group K Disambiguation).
5. Supervised Model Scoring (XGBoost C1, hist, depth=5, lr=0.08, n_est=300).
6. Multi-Candidate Thresholding (tau* = 0.88).
7. Submission Output Formatting:
   - matching_results.tsv
   - candidate_pairs.tsv

Usage:
    # Full inference (loads pre-trained frozen_model_d1.pkl):
    python run_pipeline.py --data-dir dataset/test --output-dir output

    # Full train-and-infer reproduction:
    python run_pipeline.py --train --train-dir dataset/train --data-dir dataset/test --output-dir output

    # Fast deterministic smoke test:
    python run_pipeline.py --smoke-test --limit 50
"""

import argparse
import os
import sys
import io
import time
import pickle
from collections import defaultdict
import duckdb
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

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
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline (Config D1)")
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
        default="output/frozen_model_d1.pkl",
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
        default=200,
        help="Priority blocker candidate cap per S1 (default: 200)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test on first 50 S1 records",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit on test S1 records for debugging",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of S1 entities to process per inference chunk to conserve RAM (default: 50,000)",
    )
    return parser.parse_args()


def train_frozen_model(train_dir: str, model_save_path: str, cap: int = 200) -> EntityMatcherModel:
    """Trains the frozen Config D1 model on full training data with exact 76 features."""
    print("=" * 80)
    print("TRAINING FROZEN CONFIG D1 (76 FEATURES, ENHANCED BLOCKER V2 CAP 200)")
    print("=" * 80)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")

    s1_path = os.path.join(train_dir, "train_source1.tsv").replace("\\", "/")
    s2_path = os.path.join(train_dir, "train_source2.tsv").replace("\\", "/")
    s3_path = os.path.join(train_dir, "train_source3.tsv").replace("\\", "/")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv").replace("\\", "/")

    print("[1/4] Ingesting training files...")
    con.execute(f"""
    CREATE OR REPLACE TABLE tr_s1 AS 
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('{s1_path}', delim='\\t', header=true, quote='', all_varchar=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE tr_cand_pool AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s2_path}', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s3_path}', delim='\\t', header=true, quote='', all_varchar=true);
    """)

    print("[2/4] Generating training candidates with Enhanced Blocker V2 (11 channels, Cap 200)...")
    blocker = CandidateBlocker(con)
    blocker.generate_candidates(
        s1_table_or_path="tr_s1",
        cand_table_or_path="tr_cand_pool",
        output_table="tr_candidates",
        max_candidates_per_s1=cap,
        enable_enhanced_channels=True,
    )
    df_cands = con.execute("SELECT * FROM tr_candidates").fetchdf()

    print("[3/4] Extracting 76 pairwise features...")
    s1_rows = con.execute("SELECT entity_id, country, business_name, business_address FROM tr_s1").fetchall()
    s1_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in s1_rows}

    cand_rows = con.execute("""
    SELECT entity_id, country, business_name, business_address 
    FROM tr_cand_pool 
    WHERE entity_id IN (SELECT DISTINCT candidate_entity_id FROM tr_candidates);
    """).fetchall()
    cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in cand_rows}

    extractor = PairwiseFeatureExtractor()
    df_features = extractor.extract_features(df_cands, s1_lookup, cand_lookup)

    # Attach ground truth labels
    gt_rows = con.execute(f"SELECT source1_entity_id, matched_entity_ids FROM read_csv('{gt_path}', delim='\\t', header=true, quote='', all_varchar=true);").fetchall()
    gt_dict = {}
    for s1, m_str in gt_rows:
        gt_dict[s1] = set(m_str.split(",")) if m_str and str(m_str).strip() else set()

    labels = [1 if c in gt_dict.get(s, set()) else 0 for s, c in zip(df_features["source1_entity_id"], df_features["candidate_entity_id"])]
    df_features["match_label"] = labels

    train_pos = int(sum(labels))
    train_neg = int(len(labels) - train_pos)
    scale_weight = float(np.sqrt(train_neg / max(1, train_pos)))

    print(f"[4/4] Fitting XGBoost C1 on {len(df_features):,} pairs (pos={train_pos:,}, neg={train_neg:,}, scale_pos_weight={scale_weight:.2f})...")
    feats_76 = get_feature_names()
    X = df_features[feats_76].values
    y = df_features["match_label"].values

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
            "tree_method": "hist",
        },
        random_state=42,
        feature_names=feats_76,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    model.save(model_save_path)
    print(f"Model successfully saved to {model_save_path}!")
    return model


def run_inference(
    data_dir: str,
    output_dir: str,
    model_path: str,
    threshold: float = 0.88,
    cap: int = 200,
    limit: int = None,
    chunk_size: int = 50000,
):
    """
    Executes streaming inference across test Source 1, Source 2, and Source 3.
    Processes S1 entities in chunks to maintain strict memory bounds (<12 GB RAM).
    """
    print("=" * 80)
    print("RUNNING INFERENCE WITH FROZEN CONFIG D1")
    print(f"Data Directory     : {data_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Model Checkpoint   : {model_path}")
    print(f"Decision Threshold : tau* = {threshold}")
    print(f"Blocker Cap        : {cap}")
    print(f"Chunk Size         : {chunk_size:,} S1 entities/batch")
    print(f"Row Limit          : {limit if limit else 'ALL (Full Test Set)'}")
    print("=" * 80)

    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(data_dir, "test_source1.tsv").replace("\\", "/")
    s2_path = os.path.join(data_dir, "test_source2.tsv").replace("\\", "/")
    s3_path = os.path.join(data_dir, "test_source3.tsv").replace("\\", "/")

    if not (os.path.exists(s1_path) and os.path.exists(s2_path) and os.path.exists(s3_path)):
        raise FileNotFoundError(f"Missing required test files in {data_dir}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='12GB';")

    print("[1/3] Loading candidate pool (Source 2 + Source 3)...")
    con.execute(f"""
    CREATE TEMP TABLE test_cand_pool AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s2_path}', delim='\\t', header=true, quote='', all_varchar=true)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{s3_path}', delim='\\t', header=true, quote='', all_varchar=true);
    """)
    n_cand_pool = con.execute("SELECT count(*) FROM test_cand_pool").fetchone()[0]
    print(f"  Candidate pool loaded: {n_cand_pool:,} records.")

    print(f"[2/3] Loading Model C1 from {model_path}...")
    model = EntityMatcherModel.load(model_path)
    extractor = PairwiseFeatureExtractor()
    blocker = CandidateBlocker(con)
    features_76 = get_feature_names()

    print("[2.5/3] Pre-indexing candidate pool across 11 channels...")
    blocker.preindex_candidate_pool("test_cand_pool", enable_enhanced_channels=True)

    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    f_match = open(matching_tsv, "w", encoding="utf-8")
    f_cand = open(candidate_tsv, "w", encoding="utf-8")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    print(f"[3/3] Streaming test S1 entities in batches of {chunk_size:,}...")
    s1_reader = pd.read_csv(
        s1_path,
        sep="\t",
        chunksize=chunk_size,
        nrows=limit,
        dtype=str,
        keep_default_na=False,
    )

    total_s1 = 0
    total_candidates = 0
    total_matches = 0

    for chunk_idx, df_s1 in enumerate(s1_reader, start=1):
        t_chunk = time.time()
        con.register("cur_s1", df_s1)

        # 1. Blocking
        blocker.generate_candidates(
            s1_table_or_path="cur_s1",
            cand_table_or_path="test_cand_pool",
            output_table="cur_cands",
            max_candidates_per_s1=cap,
            enable_enhanced_channels=True,
        )
        df_cands = con.execute("SELECT * FROM cur_cands").fetchdf()

        # 2. Lookups & Feature extraction
        s1_lookup = {
            r["entity_id"]: {
                "country": r["country"],
                "business_name": r["business_name"],
                "business_address": r["business_address"],
            }
            for _, r in df_s1.iterrows()
        }

        cand_rows = con.execute("""
        SELECT c.entity_id, c.country, c.business_name, c.business_address 
        FROM (SELECT DISTINCT candidate_entity_id FROM cur_cands) u
        JOIN test_cand_pool c ON u.candidate_entity_id = c.entity_id;
        """).fetchall()
        cand_lookup = {r[0]: {"country": r[1], "business_name": r[2], "business_address": r[3]} for r in cand_rows}

        if len(df_cands) > 0:
            df_feats = extractor.extract_features(df_cands, s1_lookup, cand_lookup)
            probs = model.predict_proba(df_feats[features_76].values)
            df_feats["prob"] = probs
            df_feats["is_match"] = (probs >= threshold).astype(int)

            cands_map = df_cands.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
            matches_map = df_feats[df_feats["is_match"] == 1].groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
        else:
            cands_map = {}
            matches_map = {}

        for s1_id in df_s1["entity_id"]:
            c_list = cands_map.get(s1_id, [])
            m_list = matches_map.get(s1_id, [])
            f_cand.write(f"{s1_id}\t{','.join(c_list)}\n")
            f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
            total_candidates += len(c_list)
            total_matches += len(m_list)

        total_s1 += len(df_s1)
        f_cand.flush()
        f_match.flush()
        con.execute("DROP TABLE IF EXISTS cur_cands;")
        con.unregister("cur_s1")
        print(f"  Batch {chunk_idx:3d} ({len(df_s1):,} S1) processed in {time.time() - t_chunk:.2f}s | Cumulative S1: {total_s1:,}")

    f_match.close()
    f_cand.close()

    print("=" * 80)
    print(f"Inference complete in {time.time() - t_start:.2f}s!")
    print(f"  Total S1 Entities Processed: {total_s1:,}")
    print(f"  Total Candidate Pairs       : {total_candidates:,}")
    print(f"  Total Predicted Matches     : {total_matches:,}")
    print(f"  Matching Results Written to : {matching_tsv}")
    print(f"  Candidate Pairs Written to  : {candidate_tsv}")
    print("=" * 80)


def main():
    args = parse_args()
    start_total = time.time()

    data_dir = resolve_path(
        args.data_dir,
        [
            os.path.join(CURRENT_DIR, args.data_dir),
            os.path.join(CURRENT_DIR, "..", "..", "dataset", "test"),
            os.path.join(CURRENT_DIR, "..", "..", "student_resource", "dataset", "test"),
            "dataset/test",
            "student_resource/dataset/test",
        ],
    )

    train_dir = resolve_path(
        args.train_dir,
        [
            os.path.join(CURRENT_DIR, args.train_dir),
            os.path.join(CURRENT_DIR, "..", "..", "dataset", "train"),
            os.path.join(CURRENT_DIR, "..", "..", "student_resource", "dataset", "train"),
            "dataset/train",
            "student_resource/dataset/train",
        ],
    )

    model_path = resolve_path(
        args.model_path,
        [
            os.path.join(CURRENT_DIR, args.model_path),
            os.path.join(CURRENT_DIR, "output", "frozen_model_d1.pkl"),
            "output/frozen_model_d1.pkl",
        ],
    )

    output_dir = args.output_dir
    if args.smoke_test and output_dir == "output":
        output_dir = "scratch/smoke_test_output"

    os.makedirs(output_dir, exist_ok=True)

    limit = 50 if args.smoke_test and args.limit is None else args.limit

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE (CONFIG D1)")
    print("=" * 80)
    print(f"Data Directory     : {data_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Model Path         : {model_path}")
    print(f"Decision Threshold : tau* = {args.threshold}")
    print(f"Blocker Cap        : {args.cap}")
    print(f"Features           : 76 Features (Config D1)")
    print(f"Limit (S1 Records) : {limit if limit else 'ALL (Full Test Set)'}")
    print(f"Smoke Test Mode    : {args.smoke_test}")
    print("=" * 80)

    if args.train or not os.path.exists(model_path):
        train_frozen_model(train_dir=train_dir, model_save_path=model_path, cap=args.cap)

    if args.smoke_test:
        print("\nExecuting dry-run smoke test (first 50 S1 records)...")
        limit = 50 if limit is None else limit
        run_inference(
            data_dir=data_dir,
            output_dir=output_dir,
            model_path=model_path,
            threshold=args.threshold,
            cap=args.cap,
            limit=limit,
            chunk_size=args.chunk_size,
        )
    else:
        print("\nExecuting full test inference...")
        run_inference(
            data_dir=data_dir,
            output_dir=output_dir,
            model_path=model_path,
            threshold=args.threshold,
            cap=args.cap,
            limit=limit,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
