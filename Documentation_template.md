# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** BrawlDevs  
**Team Members:**  
Sai Pranav S R  
Prannav R  
Priyajit Biswal  
Pretham Kumar K  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Business Entity Resolution system designed for the Amazon ML Challenge 2026. Addressing heterogeneous, noisy entity fragments across three unlinked data sources (`Source 1`, `Source 2`, and `Source 3`), our architecture combines an open-set multi-channel candidate blocker (Enhanced Blocker V2, 11 channels, Priority Cap = 200) with an XGBoost gradient-boosted decision tree classifier (Model C1) operating on a frozen 76-feature pairwise schema. Evaluated under the competition metric—Macro-$F_{0.5}$ across deduplicated Source 1 entities—our solution achieves an untouched holdout score of **Macro-$F_{0.5} = \mathbf{0.8351}$** (Macro Precision: **0.9024**, Macro Recall: **0.7206**, End-to-End Recall: **71.33%**, Singleton Macro-$F_{0.5}$: **0.7422**) without reliance on external web APIs, deep neural models, or commercial knowledge graphs.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory Data Analysis across the training and test corpuses revealed critical properties of commercial identity records:

1. **Source Asymmetry & Noise:** Source 1 is deduplicated and clean. Sources 2 and 3 exhibit high noise, including legal suffix variations (`Pvt Ltd`, `LLC`, `SARL`, `SAS`), abbreviations (`Rd` vs. `Road`, `St` vs. `Street`), phonetic transliterations (especially across Indic scripts and Western European accents), and missing address fields (~2.66% missing in S2/S3).
2. **Cardinality Invariant (Multi-Match Semantics):** A Source 1 entity may match zero records (12.38% in test), one record, or multiple records (up to 62 matches for large enterprise profiles). Formulating resolution as a 1-to-1 matching task is fundamentally flawed; each candidate must be independently evaluated via calibrated probability thresholding.
3. **Hard Country Invariant:** 100.0% of ground-truth matches share identical country labels. Enforcing `candidate.country == s1.country` eliminates cross-country negative candidates with zero recall loss.
4. **Open-Set Country Distribution:** The test set introduces `France` (259,452 S1 records) alongside `India` (809,986) and `US` (663,106). All normalization, blocking keys, and feature extractors were engineered to operate generically on open-set strings without hardcoded country filters.

### 2.2 Solution Strategy

Our architecture employs a two-stage **Blocking + Supervised Classification** framework:

- **Stage 1 (Candidate Generation):** An 11-channel DuckDB-powered SQL blocking engine (Enhanced Blocker V2) applies multi-signal candidate generation bounded by a strict Priority Cap = 200 candidates per S1 entity.
- **Stage 2 (Supervised Pairwise Matching):** Candidate pairs are transformed into 76 pairwise features capturing string similarity, token overlap, numeric address alignment, script detection, blocking provenance, missing-address recovery, and address disambiguation. A calibrated XGBoost model predicts match probability, and all pairs exceeding $\tau^* = 0.88$ are accepted.

**Approach Type:** Multi-Channel Blocking + Gradient-Boosted Tabular Classifier with Multi-Match Thresholding  
**Core Innovation:** A provenance-aware 76-feature taxonomy combined with Group K address-disambiguation penalties, an asymmetric loss-weighting strategy (`scale_pos_weight = 5.83`), and an exact S1-level threshold calibration ($\tau^* = 0.88$) that directly maximizes Macro-$F_{0.5}$ while maintaining 100% offline license compliance.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 17.2\text{ trillion}$ pair comparison space into a computationally feasible candidate set, we developed an 11-channel blocker in `src/blocking.py`:

- **Blocking channels used:**
  1. **Channel A (Priority 100):** Exact normalized core name (strips unambiguous legal entity designators and web domains).
  2. **Channel A2 (Priority 95):** Prefix-stripped core name (normalizes prefixes like `The `, `Dr `, `M/s `, `DBA: `).
  3. **Channel E2 (Priority 85):** Address number + locality anchor (recovers Indic trade aliases and non-Latin native script pairs).
  4. **Channel B (Priority 80):** Rare name tokens (inverted index capped at country-level document frequency $\le 200$).
  5. **Channel E (Priority 75):** Address numeric anchors (PIN / street number + 3-character name prefix).
  6. **Channel E3 (Priority 70):** Address number + street token anchor (street token length $\ge 4$, DF $\le 50$).
  7. **Channel H (Priority 65):** Postal code + 3-character name prefix anchor.
  8. **Channel G (Priority 60):** Approximate 1-edit initial-character typo key (recovers OCR errors like `6nni` vs. `Gnni`, `0` vs. `O`).
  9. **Channel I (Priority 55):** Distinctive name token pairs (two distinctive tokens from name, DF $\le 50$).
  10. **Channel D (Priority 50):** Distinctive address tokens (locality and street tokens, DF $\le 150$).
  11. **Channel C (Priority 40):** Character 4-gram prefix + suffix index.

- **Candidate pairs generated:**
  - Priority Candidate Cap = 200 candidates per S1 entity.
  - Achieves **80.42% candidate recall** on fresh holdout validation.

---

## 4. Matching Model

### 4.1 Feature Engineering (76 Features)

All features are defined with strict ordering in `src/feature_schema.py` and extracted in `src/features.py`:

- **Group A: Country Features (2):** `country_exact_match`, `country_missing_either`.
- **Group B: Name Similarities (24):** Exact clean match, Levenshtein ratio, token sort ratio, token set ratio, Jaro-Winkler similarity, length difference, token overlap ratio, Jaccard token similarity, prefix character match, legal suffix match, script match indicator, char 3-gram containment, etc.
- **Group C: Address & Postal Similarities (14):** Exact address match, address token Jaccard similarity, numeric token overlap ratio, numeric token exact match, postal code exact match, building number match, address token length delta, etc.
- **Group D: Cross-Field Interactions (7):** High name + high address agreement, exact name + exact address agreement, exact name + address number match, shared name + shared number, cross-script + strong address, address overlap but low name sim.
- **Group E: Source-Specific Indicators (4):** Source 2 indicator, Source 3 indicator, missing address in candidate, URL in candidate name.
- **Group F: Blocking Provenance & Priority (14):** Individual channel flags (A, A2, B, C, D, E, E2, G), cumulative priority score, total channels fired, candidate rank order.
- **Group G: Missing-Address Recovery (4):** Name token Jaccard, 3-gram, containment, and Levenshtein scaled by candidate address missingness.
- **Group H: Alias & Trade Names (2):** Pairwise best token Levenshtein similarity, distinctive token Jaccard.
- **Group I: OCR & Typo Robustness (1):** Character 2-gram overlap coefficient.
- **Group K: Address Disambiguation & False Positive Suppression (4):**
  - `feat_same_postal_weak_name`: Postal match with name Jaccard $< 0.20$.
  - `feat_distinctive_name_zero_overlap`: Zero overlap on non-generic business tokens.
  - `feat_same_building_weak_name`: Building number match with name Jaccard $< 0.20$.
  - `feat_high_addr_low_name_penalty`: Triggered when address Jaccard $\ge 0.60$ but name Jaccard $\le 0.25$ (penalizes co-located distinct tenants in shopping centers/complexes).

### 4.2 Model Type & Hyperparameters

- **Estimator:** `XGBClassifier` (XGBoost 3.2.0, Hist Tree Method).
- **Hyperparameters:**
  - `n_estimators = 300`
  - `max_depth = 5`
  - `learning_rate = 0.08`
  - `subsample = 0.80`, `colsample_bytree = 0.80`
  - `min_child_weight = 5`
  - `gamma = 0.10`, `reg_alpha = 0.10`, `reg_lambda = 1.00`
  - `scale_pos_weight = 5.83` (calibrated to training negative-to-positive ratio)
  - `random_state = 42`
  - Model footprint: ~700 KB serialized (< 10,000 tree nodes total; << 8B parameters).

### 4.3 Threshold Selection Method

Because Macro-$F_{0.5}$ weights Precision twice as heavily as Recall, traditional 0.50 classification thresholds yield catastrophic false-positive penalties. We evaluated thresholds $\tau \in [0.80, 0.92]$:
- $\tau = 0.80 \implies \text{Macro-}F_{0.5} = 0.8497$ (excess false positives).
- $\tau^* = 0.88 \implies \text{Macro-}F_{0.5} = 0.8530$ (optimal trade-off on dev set).
- Every candidate with $P(\text{match}) \ge 0.88$ is predicted as a match.

---

## 5. Results & Validation

### 5.1 Fresh S1 Holdout Validations
Across three independent, non-overlapping validation holdouts:

| Holdout Partition | S1 Entities | Configuration | Macro-$F_{0.5}$ | Precision | Recall | Singleton $F_{0.5}$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 5.1 Holdout A** | 1,004 | Baseline (65 feats, Cap 150) | **0.8122** | 0.8754 | 0.7103 | 0.6650 | Locked |
| **Phase 11 Fresh Holdout B** | 597 | Config E (74 feats, Cap 200) | **0.8350** | 0.8966 | 0.7300 | 0.6706 | Locked |
| **Phase 13 Fresh Holdout C** | 478 | **Config D1 (76 feats, Cap 200)**| **0.8351** | **0.9024** | 0.7206 | **0.7422** | **Validated** |

### 5.2 Key Insights on Config D1
1. **Precision Stability:** Macro precision reached **0.9024**, demonstrating that the Group K address-disambiguation penalty actively suppresses false merges.
2. **Singleton Surge:** Singleton Macro-$F_{0.5}$ surged to **0.7422** (+0.0716 lift over Phase 11), proving that false-merge suppression unblocks true singletons.
3. **Generalization Stability:** The generalization gap from development (0.8530) to fresh holdout (0.8351) is 1.79 percentage points, closely matching the Phase 11 gap (1.62 points).

---

## 6. Conclusion

We demonstrated that combining domain-specific multi-channel blocking with conservative gradient-boosted decision trees, Group K address-disambiguation penalties, and precision-skewed threshold calibration produces an exceptionally robust entity resolution system. By respecting multi-candidate cardinality, enforcing open-set country invariance, and utilizing a leakage-free 76-feature taxonomy, our pipeline achieved Macro-$F_{0.5} = 0.8351$ on fresh untouched holdout data with 100% determinism.

---

## Appendix: Code Artefacts

All runnable code is organized under `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multi-representation text and address cleaning.
- `src/blocking.py`: DuckDB SQL-based candidate blocking engine (Cap 200, 11 channels).
- `src/features.py` & `src/feature_schema.py`: 76-feature extraction.
- `src/model.py`: XGBoost 3.2.0 wrapper with deterministic serialization.
- `src/evaluate.py`: Official S1-level Macro-$F_{0.5}$ evaluation engine.
- `run_pipeline.py`: End-to-end reproducible inference script.
- `requirements.txt`: Pinned environment specifications.

**Reproducibility Entry Point:**
```bash
# Fast smoke test (50 S1 entities, ~3s runtime)
python code/business_entity_resolution/run_pipeline.py --smoke-test --limit 50

# Full end-to-end inference
python code/business_entity_resolution/run_pipeline.py --data-dir dataset/test --output-dir output
```
