# Amazon ML Challenge 2026: Business Entity Resolution

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/XGBoost-3.2.0-orange.svg" alt="XGBoost 3.2.0" />
  <img src="https://img.shields.io/badge/DuckDB-1.4.3-yellow.svg" alt="DuckDB 1.4.3" />
  <img src="https://img.shields.io/badge/Holdout_Macro--F0.5-0.8351-success.svg" alt="Holdout Macro-F0.5: 0.8351" />
  <img src="https://img.shields.io/badge/Precision-0.9024-brightgreen.svg" alt="Precision: 0.9024" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License Apache 2.0" />
</p>

This repository contains the complete, production-grade winning solution developed by team **BrawlDevs** for the **Amazon ML Challenge 2026: Business Entity Resolution**.

The objective is to resolve unlinked, noisy business entity records from three distinct data sources (`Source 1`, `Source 2`, and `Source 3`) to reference entities in `Source 1`, evaluated under entity-level **Macro-$F_{0.5}$** across deduplicated target identities.

---

## 1. Executive Summary

| Metric / Dimension | Production Result | Notes |
| :--- | :---: | :--- |
| **Untouched Holdout Macro-$F_{0.5}$** | **0.8351** | Official entity-level competition metric across S1 (Config D1) |
| **Holdout Macro Precision** | **0.9024** | Precision-skewed weighting ($\beta = 0.5$) |
| **Holdout Macro Recall** | **0.7206** | High recall under strict multi-channel blocking |
| **End-to-End Recall** | **71.33%** | Combined blocker + classifier true match recovery |
| **Blocker Recall** | **80.42%** | Enhanced Blocker V2 (11 channels, Priority Cap = 200) |
| **Singleton Entity $F_{0.5}$** | **0.7422** | Unlinked / zero-match entities protected from false merges |
| **Total Test S1 Entities** | **1,732,544** | Full unseen test corpus |
| **Total Test Candidate Pairs** | **226,529,981** | Mean 130.75 pairs/S1 across India, US, and France |
| **Total Predicted Matches** | **6,994,107** | Calibrated probability threshold $\tau^* = 0.88$ |
| **Official Submission Validator** | **PASS (Exit Code 0)** | Zero format, integrity, cap, or subset violations |

---

## 2. Solution Architecture

```text
               Raw Test TSVs (Source 1, Source 2, Source 3)
                                    │
                                    ▼
                 Data Preprocessing & Normalization
          - Open-set Unicode NFKC cleaning & mojibake repair
          - Legal suffix standardization (Pvt Ltd, LLC, SARL, SAS, EURL)
          - Alphanumeric, character n-grams, and postal/numeric tokens
                                    │
                                    ▼
         Enhanced Blocker V2: 11-Channel SQL Engine (Cap = 200)
          - Country Partition Invariant (candidate.country == s1.country)
          - Channels: A, A2, E2, B, E, E3, H, G, I, D, C
          - Mean 130.75 candidate pairs/S1 (226.53M test pairs total)
                                    │
                                    ▼
                Frozen 76-Feature Extraction Engine
          - 65 Canonical Features (names, addresses, numerics, provenance)
          - 9 Selected Robustness Features (missing-address recovery, aliases)
          - 2 Group K Address Disambiguation & Co-Location Penalties
                                    │
                                    ▼
               Calibrated XGBoost Model C1 Inference
          - Tree Method: hist, Depth: 5, Estimators: 300, LR: 0.08
          - Asymmetric class weight: scale_pos_weight = 5.83, Seed: 42
                                    │
                                    ▼
             Multi-Match Decision Thresholding (tau* = 0.88)
          - Predict match iff P(match) >= 0.88; empty string if none qualify
                                    │
                                    ▼
                    Submission Deliverables Generated
          - output/matching_results.tsv (1,732,544 rows)
          - output/candidate_pairs.tsv (1,732,544 rows)
```

---

## 3. Core Technical Innovations

1. **Multi-Match Cardinality:**  
   Commercial business entities often possess multiple legal representations, trade registrations, or branch listings across independent databases. Unlike naive approaches that enforce a 1-to-1 argmax constraint, our pipeline independently evaluates every blocked candidate pair against a calibrated threshold ($\tau^* = 0.88$). A Source 1 entity may match zero records (7.11% in test), one record, or multiple records (up to 149 matches).

2. **Hard Country Invariant & Open-Set Generalization:**  
   Empirical analysis across all ground-truth pairs proved that 100.0% of matching entities share identical country labels. Enforcing `candidate.country == s1.country` reduces the comparison space by orders of magnitude with zero recall penalty. The architecture treats countries as open-set strings, enabling native support for `France` alongside `India` and `US` without hardcoded branch logic.

3. **High-Recall DuckDB SQL Blocking Engine (Enhanced V2):**  
   Candidate generation pairs an 11-channel union blocker (exact core name, prefix-stripped name, locality anchors, rare token index, address numeric anchors, postal anchors, 1-edit typo keys, distinctive token pairs, and character n-grams) with DuckDB in-memory SQL execution, streaming 226.5M test pairs at over 1.2M pairs/sec.

4. **Group K Address Disambiguation & Co-Location Suppression:**  
   Dense commercial districts often have distinct tenants sharing building numbers and street names. Our 76-feature taxonomy incorporates Group K penalties (`feat_same_building_weak_name` and `feat_high_addr_low_name_penalty`) that actively penalize co-located distinct businesses, boosting precision to **0.9024** and singleton $F_{0.5}$ to **0.7422**.

5. **Asymmetric Precision Optimization ($\beta = 0.5$):**  
   Because the Macro-$F_{0.5}$ metric penalizes false positives twice as heavily as false negatives, threshold calibration on held-out validation data yielded $\tau^* = 0.88$, paired with `scale_pos_weight = 5.83` during training to combat candidate pair class imbalance (~1:13).

---

## 4. Repository Structure

```text
├── README.md                           # Main solution documentation and reference guide
├── requirements.txt                    # Pinned Python production dependencies
├── run_pipeline.py                     # Root inference and reproducibility entry point (Config D1)
├── Documentation_template.md           # Official competition submission document (BrawlDevs)
├── Problem_Statement.pdf               # Competition problem statement
│
├── code/                               # Official submission package structure
│   └── business_entity_resolution/
│       ├── README.md                   # Submission guide
│       ├── requirements.txt            # Minimal dependencies
│       ├── run_pipeline.py             # Self-contained pipeline script
│       └── src/                        # Complete 76-feature source modules
│           ├── __init__.py
│           ├── preprocessing.py        # Text normalization & legal suffix stripping
│           ├── blocking.py             # Enhanced Blocker V2 (11 channels, Cap 200)
│           ├── features.py             # 76-feature extraction engine
│           ├── feature_schema.py       # Feature taxonomy definition
│           ├── model.py                # XGBoost wrapper
│           └── evaluate.py             # Official S1-level Macro-F0.5 evaluator
│
├── src/                                # Synchronized root production modules
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── features.py
│   ├── feature_schema.py
│   ├── model.py
│   └── evaluate.py
│
├── eda/                                # Research notebooks, ablations, and phase holdout audits
│   ├── phase11_fresh_holdout.py        # Phase 11 holdout benchmark
│   ├── phase12_targeted_experiments.py # Phase 12 address disambiguation ablations
│   ├── phase13_fresh_holdout.py        # Phase 13 holdout verification (0.8351 benchmark)
│   ├── phase14_pre_inference_audit.py  # Phase 14 27/27 invariant audit
│   └── benchmark_blocking_v2.py        # 11-channel blocker benchmarks
│
├── utils/                              # Official competition utilities
│   └── validate_submission.py          # Memory-optimized submission validator
│
└── output/                             # Trained models and benchmark summaries
    ├── .gitkeep
    ├── frozen_model_d1.pkl             # Winning Config D1 XGBoost model (722 KB)
    ├── frozen_model_c1.pkl             # Baseline Model C1 checkpoint (685 KB)
    └── *.json                          # Benchmark and test output statistics
```

---

## 5. Quickstart & Reproduction

### Prerequisites
- Python 3.11+ (tested on Python 3.12)
- 8 GB+ RAM (16 GB+ recommended for full test dataset)

### Installation
```bash
git clone https://github.com/prannav-r/Amazon_ML_Challenge.git
cd Amazon_ML_Challenge
pip install -r requirements.txt
```

### Fast Smoke Test (~5 seconds)
Runs the complete end-to-end pipeline (ingestion, normalization, blocking, 76-feature extraction, model loading, scoring, and output formatting) on a deterministic sample of 50 test entities:

```bash
python run_pipeline.py --smoke-test --limit 50 --output-dir output
```

### Full Test Inference (Pre-Trained Frozen Model D1)
To run inference on the full test dataset using the pre-trained model:

```bash
python run_pipeline.py \
    --data-dir dataset/test \
    --output-dir output \
    --model-path output/frozen_model_d1.pkl \
    --threshold 0.88 \
    --cap 200 \
    --chunk-size 50000
```

### Validate Submission Deliverables
Validate outputs using the memory-optimized official competition validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Exit code `0` confirms complete adherence to all formatting, integrity, and cardinality requirements.

---

## 6. License & Fair-Play Compliance

- **Model License:** Apache License 2.0 (`XGBClassifier`).
- **Parameter Footprint:** < 10,000 decision tree nodes (~722 KB serialized model size), strictly satisfying the competition parameter limit of $\le 8\text{B}$ parameters.
- **Fair Play:** 100% offline, self-contained pipeline. Zero external business lookup APIs, zero geocoding services, and zero online commercial enrichment.
