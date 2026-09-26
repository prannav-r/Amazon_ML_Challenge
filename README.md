# Amazon ML Challenge 2026: Business Entity Resolution

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/XGBoost-3.2.0-orange.svg" alt="XGBoost 3.2.0" />
  <img src="https://img.shields.io/badge/DuckDB-1.4.3-yellow.svg" alt="DuckDB 1.4.3" />
  <img src="https://img.shields.io/badge/Holdout_Macro--F0.5-0.8122-success.svg" alt="Holdout Macro-F0.5: 0.8122" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License Apache 2.0" />
</p>

This repository contains the complete, production-grade solution developed by team **BrawlDevs** for the **Amazon ML Challenge 2026: Business Entity Resolution**.

The objective is to resolve unlinked, noisy business entity records from three distinct data sources (`Source 1`, `Source 2`, and `Source 3`) to reference entities in `Source 1`, evaluated using entity-level **Macro-$F_{0.5}$**.

---

## 1. Executive Summary

| Metric / Dimension | Production Result | Notes |
| :--- | :---: | :--- |
| **Untouched Holdout Macro-$F_{0.5}$** | **0.8122** | Official entity-level competition metric across S1 |
| **Holdout Macro Precision** | **0.8848** | Precision-weighted ($\beta = 0.5$) |
| **Holdout Macro Recall** | **0.6927** | Bounded candidate blocking recall |
| **Zero-Match Entity $F_{0.5}$** | **0.9412** | Valid singletons (12.38% of test set) |
| **Non-Empty Entity $F_{0.5}$** | **0.7891** | Multi-match enterprise profiles |
| **Total Test S1 Entities** | **1,732,544** | Full unseen test dataset |
| **Total Candidate Pairs Blocked** | **154,577,946** | Priority Cap = 150 (mean 89.2 pairs/S1) |
| **Total Predicted Matches** | **4,909,130** | Calibrated threshold $\tau^* = 0.88$ |
| **Official Submission Validator** | **PASS (Exit Code 0)** | Zero format or integrity violations |

---

## 2. Solution Architecture

```text
               Raw Test TSVs (Source 1, Source 2, Source 3)
                                    │
                                    ▼
                 Data Preprocessing & Normalization
          - Open-set Unicode NFKD cleaning & lowercasing
          - Legal suffix stripping & standardization (Pvt Ltd, LLC, SARL, SAS)
          - Postal code extraction & token-level normalization
                                    │
                                    ▼
         Multi-Channel Candidate Blocking (Priority Cap = 150)
          - Country Equality Invariant (candidate.country == s1.country)
          - Channels: A, A2, B, C, D, E, E2, G
          - Generates ~89.2 candidate pairs/S1 (154.58M total)
                                    │
                                    ▼
                   Canonical 65-Feature Generation
          - Group A: Country Integrity (2)
          - Group B: Business Name Similarities (22)
          - Group C: Address & Postal Similarities (17)
          - Group D: Cross-Field Interactions (7)
          - Group E: Source-Specific Indicators (4)
          - Group F: Blocking Provenance & Priority (11)
                                    │
                                    ▼
               Frozen XGBoost Model C1 Inference
          - Tree Method: hist, Depth: 5, Estimators: 300
          - Learning Rate: 0.08, scale_pos_weight: 5.84, Seed: 42
                                    │
                                    ▼
             Multi-Match Decision Thresholding (tau* = 0.88)
          - Accept candidate iff P(match) >= 0.88
                                    │
                                    ▼
                    Submission Artifact Generation
          - output/matching_results.tsv (1,732,544 rows)
          - output/candidate_pairs.tsv (1,732,544 rows)
```

---

## 3. Core Technical Innovations

1. **Multi-Match Cardinality (Zero, One, or Many):**  
   Commercial business entities often possess multiple legal representations, trade registrations, or branch listings across independent databases. Unlike naive approaches that enforce a 1-to-1 argmax constraint, our pipeline independently evaluates every blocked candidate pair against a calibrated threshold ($\tau^* = 0.88$). A Source 1 entity may match zero records (12.38% in test), one record, or multiple records (up to 62 matches).

2. **Hard Country Invariant & Open-Set Generalization:**  
   Empirical analysis across all ground-truth pairs proved that 100.0% of matching entities share identical country labels. Enforcing `candidate.country == s1.country` reduces the comparison space by orders of magnitude with zero recall penalty. The architecture treats countries as open-set strings, enabling native support for `France` alongside `India` and `US` without hardcoded branch logic.

3. **High-Recall DuckDB SQL Blocking Engine:**  
   Candidate generation pairs an 8-channel union blocker (exact core name, prefix-stripped name, rare token index, address numeric anchors, locality anchors, character 4-grams, and typo keys) with DuckDB in-memory SQL execution, processing 154.58M pairs at over 1.2M pairs/sec.

4. **Provenance-Aware 65-Feature Taxonomy:**  
   Features capture multi-dimensional alignment across names, standardized street addresses, numeric building/unit identifiers, postal/PIN codes, script detection, cross-field agreements, and blocking provenance channels (`src/feature_schema.py`).

5. **Asymmetric Precision Optimization ($\beta = 0.5$):**  
   Because the Macro-$F_{0.5}$ metric penalizes false positives $4\times$ more heavily than false negatives, threshold calibration on held-out validation data yielded $\tau^* = 0.88$, paired with `scale_pos_weight = 5.84` during training to combat candidate pair class imbalance (~1:13).

---

## 4. Repository Structure

```text
├── README.md                           # Project documentation and reproduction guide
├── requirements.txt                    # Pinned Python production dependencies
├── run_pipeline.py                     # End-to-end inference and smoke-test entry point
├── Documentation_template.md           # Official methodology and technical write-up
├── Problem_Statement.pdf               # Competition problem statement
│
├── src/                                # Core production modules
│   ├── __init__.py
│   ├── preprocessing.py                # Text cleaning, legal suffixes, normalization
│   ├── blocking.py                     # DuckDB SQL multi-channel candidate blocker
│   ├── features.py                     # 65-feature extraction engine
│   ├── feature_schema.py               # Canonical feature schema and taxonomy
│   ├── model.py                        # XGBoost estimator wrapper
│   └── evaluate.py                     # S1-level Macro-F0.5 evaluation engine
│
├── eda/                                # Research, exploratory analysis, and benchmarks
│   ├── benchmark_blocking_channels.py
│   ├── benchmark_models.py
│   ├── phase5_1_holdout.py
│   ├── tune_threshold.py
│   └── run_test_inference.py
│
├── student_resource/                   # Challenge dataset structure and utilities
│   ├── utils/validate_submission.py    # Official submission validator
│   └── Documentation_template.md
│
└── output/                             # Generated predictions and models
    ├── .gitkeep
    └── frozen_model_c1.pkl             # Serialized Model C1 weights (685 KB)
```

---

## 5. Quickstart & Reproduction

### Prerequisites
- Python 3.12 (compatible with Python 3.9+)
- 8 GB+ RAM (16 GB+ recommended for full test dataset)

### Installation
```bash
git clone https://github.com/prannav-r/Amazon_ML_Challenge.git
cd Amazon_ML_Challenge
pip install -r requirements.txt
```

### Run Smoke Test (~40 seconds)
Runs the complete end-to-end pipeline (ingestion, normalization, blocking, 65-feature extraction, model loading, scoring, and output generation) on a deterministic sample of 50 test entities:

```bash
python run_pipeline.py --smoke-test --limit 50 --output-dir output
```

### Full Test Inference
To run inference on the full test dataset:

```bash
python run_pipeline.py \
    --data-dir student_resource/dataset/test \
    --output-dir output \
    --threshold 0.88 \
    --cap 150
```

### Validate Submission Deliverables
Validate outputs using the official competition validator:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

Exit code `0` confirms complete adherence to all formatting, integrity, and cardinality requirements.

---

## 6. License & Fair-Play Compliance

- **Model License:** Apache License 2.0 (`XGBClassifier`).
- **Parameter Footprint:** ~6,000 decision tree nodes (~685 KB serialized model size), strictly satisfying the competition parameter limit of $\le 8\text{B}$ parameters.
- **Fair Play:** 100% offline, self-contained pipeline. Zero external business lookup APIs, zero geocoding services, and zero online commercial enrichment.
