# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** BrawlDevs  
**Team Members:** <br> Sai Pranav S R\
 Prannav R\
 Priyajit Biswal\
 Pretham Kumar K\
**Submission Date:** September 26, 2026

---

## 1. Executive Summary

We present an end-to-end, high-precision Business Entity Resolution system designed for the Amazon ML Challenge 2026. Addressing heterogeneous, noisy entity fragments across three unlinked data sources (`Source 1`, `Source 2`, and `Source 3`), our architecture combines an open-set multi-channel candidate blocker with an XGBoost gradient-boosted decision tree classifier (Model C1) operating on a canonical 65-feature pairwise schema. Evaluated under the competition metric—Macro-$F_{0.5}$ across deduplicated Source 1 entities—our solution achieves an untouched holdout score of **Macro-$F_{0.5} = 0.8122$** (Macro Precision: **0.8848**, Macro Recall: **0.6927**) without reliance on external web APIs, deep neural models, or commercial knowledge graphs.

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

- **Stage 1 (Candidate Generation):** An 8-channel DuckDB-powered SQL blocking engine applies multi-signal candidate generation bounded by a strict Priority Cap = 150 candidates per S1 entity.
- **Stage 2 (Supervised Pairwise Matching):** Candidate pairs are transformed into 65 pairwise features capturing string similarity, token overlap, numeric address alignment, script detection, and blocking provenance. A calibrated XGBoost model predicts match probability, and all pairs exceeding $\tau^* = 0.88$ are accepted.

**Approach Type:** Multi-Channel Blocking + Gradient-Boosted Tabular Classifier with Multi-Match Thresholding  
**Core Innovation:** A provenance-aware 65-feature taxonomy combined with an asymmetric loss-weighting strategy (`scale_pos_weight = 5.84`) and an exact S1-level threshold calibration that directly maximizes Macro-$F_{0.5}$ while maintaining 100% offline license compliance.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 17.2\text{ trillion}$ pair comparison space into a computationally feasible candidate set, we developed an 8-channel blocker in `src/blocking.py`:

- **Blocking channels used:**
  1. **Channel A (Priority 100):** Exact normalized core name (strips unambiguous legal entity designators and web domains).
  2. **Channel A2 (Priority 95):** Prefix-stripped core name (normalizes prefixes like `The `, `Dr `, `M/s `, `DBA: `).
  3. **Channel E2 (Priority 85):** Address number + locality anchor (recovers Indic trade aliases and non-Latin native script pairs).
  4. **Channel B (Priority 80):** Rare name tokens (inverted index capped at country-level document frequency $\le 200$).
  5. **Channel E (Priority 75):** Address numeric anchors (PIN / street number + 3-character name prefix).
  6. **Channel G (Priority 60):** Approximate 1-edit initial-character typo key (recovers OCR errors like `6nni` vs. `Gnni`, `0` vs. `O`).
  7. **Channel D (Priority 50):** Distinctive address tokens (locality and street tokens, DF $\le 150$).
  8. **Channel C (Priority 40):** Character 4-gram prefix + suffix index.

- **Candidate pairs generated:**
  - **Full Test Set:** 154,577,946 candidate pairs across 1,732,544 test S1 entities.
  - **Candidate Distribution:** Mean = 89.2 pairs/S1, Median = 96.0 pairs/S1, Max = 150 pairs/S1 (strictly enforced), Zero-candidate S1s = 5,929 (0.34%).
- **How true matches were preserved:**
  - Independent channel audit demonstrated that Channels A, A2, and B capture 78.4% of matches, while address-anchored Channels E, E2, and G recover the remaining 9.7% of challenging trade-name and native-script matches, achieving **88.1% candidate recall** on the held-out development set.

---

## 4. Matching Model

### 4.1 Feature Engineering (65 Canonical Features)

All features are defined with strict ordering in `src/feature_schema.py` and extracted in `src/features.py`:

- **Group A: Country Features (2):** `country_exact_match`, `country_missing_either`.
- **Group B: Name Similarities (22):** Exact clean match, Levenshtein ratio, token sort ratio, token set ratio, partial ratio, Jaro-Winkler similarity, length difference, token overlap ratio, Jaccard token similarity, prefix character match, legal suffix match, script match indicator, char 3-gram containment, etc.
- **Group C: Address & Postal Similarities (17):** Exact address match, address Levenshtein ratio, address token Jaccard similarity, numeric token overlap ratio, numeric token exact match, postal code exact match, postal code missing either, standardized street name match, address token length delta, etc.
- **Group D: Cross-Field Interactions (7):** High name + high address agreement, exact name + exact postal agreement, name mismatch + address mismatch penalty, combined weighted score.
- **Group E: Source-Specific Indicators (4):** Source 2 indicator, Source 3 indicator, missing address in candidate, missing name in candidate.
- **Group F: Blocking Provenance & Priority (11):** Individual channel flags (A, A2, B, C, D, E, E2, G), cumulative priority score, total channels fired, candidate rank order.

### 4.2 Model Type & Hyperparameters

- **Estimator:** `XGBClassifier` (XGBoost 3.2.0, Hist Tree Method).
- **Hyperparameters:**
  - `n_estimators = 300`
  - `max_depth = 5`
  - `learning_rate = 0.08`
  - `subsample = 0.80`, `colsample_bytree = 0.80`
  - `min_child_weight = 5`
  - `gamma = 0.10`, `reg_alpha = 0.10`, `reg_lambda = 1.00`
  - `scale_pos_weight = 5.84` (calibrated to training negative-to-positive ratio)
  - `random_state = 42`
  - Model footprint: 685 KB serialized (~6,000 tree nodes total; << 8B parameters).

### 4.3 Threshold Selection Method

Because Macro-$F_{0.5}$ weights Precision $4\times$ higher than Recall, traditional 0.50 classification thresholds yield catastrophic false-positive penalties. We evaluated thresholds $\tau \in [0.50, 0.95]$ in increments of 0.01 on a segregated S1-level development set:

- $\tau = 0.50 \implies \text{Macro-}F_{0.5} = 0.6421$ (excess false positives).
- $\tau = 0.80 \implies \text{Macro-}F_{0.5} = 0.7934$.
- $\tau^* = 0.88 \implies \text{Macro-}F_{0.5} = 0.8164$ (optimal trade-off on dev set).
- Every candidate with $P(\text{match}) \ge 0.88$ is predicted as a match.

---

## 5. Results & Error Analysis

### 5.1 Untouched S1 Holdout Evaluation

Validation was performed on a strictly held-out partition of 5,000 Source 1 entities completely excluded from model training and threshold tuning:

- **Macro-$F_{0.5}$ Score:** **0.8122**
- **Macro Precision:** **0.8848**
- **Macro Recall:** **0.6927**
- **Zero-Match Entity $F_{0.5}$:** **0.9412**
- **Non-Empty Entity $F_{0.5}$:** **0.7891**

_(Note: Test set ground truth is unreleased; 0.8122 represents our unbiased offline holdout validation benchmark.)_

### 5.2 Test Inference Output Statistics

Running the frozen pipeline on the complete test dataset yielded:

- **Test S1 Entities:** 1,732,544
- **Total Predicted Matches:** 4,909,130
- **Entities with $\ge 1$ Matches:** 1,518,084 (87.62%)
- **Entities Predicted Empty:** 214,460 (12.38%)
- **Source Breakdown:** Source 2 = 2,523,111 (51.40%), Source 3 = 2,386,019 (48.60%).

### 5.3 Error Analysis

1. **Common False Positives (Wrong Merges):**
   - **Co-Located Franchise / Chain Stores:** Entities sharing identical corporate brand names and street addresses (e.g., separate legal entities or departments within a single commercial complex).
   - **Sub-Unit Number Variations:** Buildings with multiple independent suites where the suite/floor number was omitted or corrupted in one source.
2. **Common False Negatives (Missed Matches):**
   - **Severe Phonetic / Script Discrepancies:** Vernacular Indic business names transliterated into English with non-standard phonetic spellings where neither token overlap nor character 4-grams bridged the distance.
   - **Completely Missing Addresses:** Records with blank address fields in Source 2 or 3, where the name alone was insufficient to surpass the strict 0.88 threshold.

---

## 6. Conclusion

We demonstrated that combining domain-specific multi-channel blocking with conservative gradient-boosted decision trees and precision-skewed threshold calibration produces an exceptionally robust entity resolution system. By respecting multi-candidate cardinality, enforcing open-set country invariance, and utilizing a leakage-free 65-feature taxonomy, our pipeline achieved Macro-$F_{0.5} = 0.8122$ on untouched holdout data and processed 154.58M test candidate pairs with 100% determinism.

---

## Appendix

### A. Code Artefacts

All runnable code is organized under `code/business_entity_resolution/`:

- `src/preprocessing.py`: Multi-representation text and address cleaning.
- `src/blocking.py`: DuckDB SQL-based candidate blocking engine (Cap 150).
- `src/features.py` & `src/feature_schema.py`: Canonical 65-feature extraction.
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

### B. Additional Results

#### Holdout Threshold Sensitivity ($\tau$)

| Decision Threshold ($\tau$) | Holdout Macro-$F_{0.5}$ | Macro Precision | Macro Recall |
| :-------------------------: | :---------------------: | :-------------: | :----------: |
|            0.70             |         0.7712          |     0.8014      |    0.7289    |
|            0.80             |         0.7934          |     0.8451      |    0.7145    |
|            0.85             |         0.8080          |     0.8710      |    0.7012    |
|      **0.88 (Frozen)**      |       **0.8122**        |   **0.8848**    |  **0.6927**  |
|            0.90             |         0.8095          |     0.8962      |    0.6720    |
|            0.92             |         0.7981          |     0.9104      |    0.6385    |

#### Test Set Country Breakdown

| Country    | Test S1 Count | % of Total S1 | Predicted Matches | Avg Matches / S1 | Empty S1 Count | % S1 Empty |
| :--------- | :-----------: | :-----------: | :---------------: | :--------------: | :------------: | :--------: |
| **India**  |    809,986    |    46.75%     |     1,903,932     |       2.35       |    129,075     |   15.94%   |
| **US**     |    663,106    |    38.27%     |     1,728,978     |       2.61       |     71,746     |   10.82%   |
| **France** |    259,452    |    14.98%     |     1,276,220     |       4.92       |     13,639     |   5.26%    |
