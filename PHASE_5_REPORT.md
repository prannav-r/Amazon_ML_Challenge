# PHASE 5 REPORT: SUPERVISED MODEL TRAINING, SCORE CALIBRATION & S1-LEVEL MACRO-F0.5 VALIDATION

**Amazon ML Challenge 2026: Business Entity Resolution**  
**Evaluation Objective:** Entity-level matching evaluated using Macro-$F_{0.5}$ across held-out Source 1 entities:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
**Status:** Phase 5 is **COMPLETE**.  
**Execution Guardrail:** Strict stop enforced — **No test inference performed**, **no submission files generated**, **no final ZIP created**.

---

## 1. Models Evaluated

All supervised models were trained and benchmarked strictly on the approved Phase 4 pipeline using the canonical 65-feature schema from `src/feature_schema.py` and blocked candidate pairs from `src/blocking.py`.

| Model Identifier | Architecture / Library | Core Specifications & Regularization | Class-Weight Configuration |
| :--- | :--- | :--- | :--- |
| **Model A** | **Logistic Regression Baseline** (`scikit-learn` 1.7.1) | L2-regularized Logistic Regression (`C=1.0`, `solver='lbfgs'`, `max_iter=500`), preceded by `SimpleImputer(strategy='median')` and `StandardScaler()`. | `class_weight='balanced'` |
| **Model B** | **Gradient-Boosted Trees (XGBoost Default)** (`xgboost` 3.2.0) | Conservative tabular tree hyperparameters: `n_estimators=300`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `gamma=0.1`, `reg_alpha=0.1`, `reg_lambda=1.0`, `tree_method='hist'`. | Unweighted (`scale_pos_weight=1.0`) |
| **Model C1** | **XGBoost (Mild Class Weighting)** (`xgboost` 3.2.0) | Identical conservative tree structure as Model B, but scaling positive gradients by the square root of the measured training class imbalance ratio. | `scale_pos_weight=5.18` ($\approx \sqrt{26.88}$) |
| **Model C2** | **XGBoost (Full Balanced Weighting)** (`xgboost` 3.2.0) | Identical conservative tree structure as Model B, but scaling positive gradients by the full inverse class imbalance ratio. | `scale_pos_weight=26.88` (full balanced) |
| **Model C3** | **Random Forest Baseline** (`scikit-learn` 1.7.1) | `RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=10, min_samples_leaf=5, n_jobs=-1)`. | `class_weight='balanced'` |

---

## 2. Exact Training and Validation Dataset Sizes

Candidate pairs were generated over a representative 5,000 Source 1 sample (3,000 US, 2,000 India) matching the exact country distribution observed in the full training corpus. Grouped splitting was strictly performed at the **Source 1 entity level** (`seed=42`, 80% train / 20% val).

| Dataset Split | Unique S1 Entities | Total Candidate Pairs | Positive True Matches ($y=1$) | Hard Negatives ($y=0$) | Positive Class % | Exact Negative : Positive Imbalance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | **3,978** ($+22$ empty) | **278,884** | 10,004 | 268,880 | 3.59% | **26.88 : 1** |
| **Validation Set** | **994** ($+28$ empty) | **71,106** | 2,422 | 68,684 | 3.41% | **28.36 : 1** |
| **Full Evaluation Union** | **5,000** | **349,990** | 12,426 | 337,564 | 3.55% | **27.17 : 1** |

> **Zero-Leakage Guarantee:** The intersection of Source 1 entity IDs between the training set and validation set is **strictly empty** ($\lvert \text{Train}_{S1} \cap \text{Val}_{S1} \rvert = 0$). All candidates, positive matches, and hard negatives for any given S1 entity belong exclusively to either train or validation.

---

## 3. Class-Weight Strategy Analysis

On the actual training split, there are exactly **26.88 hard negatives per positive candidate pair** (3.59% positive prevalence). We empirically compared three class-weighting configurations for gradient boosting:

1. **Unweighted (`scale_pos_weight=1.0`, Model B):**
   - Directly optimizes log-loss on the empirical sample distribution.
   - Requires setting a higher decision threshold ($\tau^* \approx 0.60$) to achieve high precision.
   - Validation S1 Macro-$F_{0.5}$: **0.7708**.
2. **Mild Weighting (`scale_pos_weight=5.18`, Model C1):**
   - Applies $\sqrt{\text{imbalance}}$ penalty to false negatives during boosting.
   - Shifts predicted probabilities towards the center, producing a cleaner and more stable threshold plateau around $\tau^* \approx 0.80 - 0.81$.
   - Validation S1 Macro-$F_{0.5}$: **0.7712** (peak **0.7715** at $\tau^*=0.81$).
3. **Full Balanced Weighting (`scale_pos_weight=26.88`, Model C2):**
   - Heavily penalizes false negatives during gradient updates, aggressively pushing candidate scores upward.
   - Because precision is weighted four times more heavily than recall in $F_{0.5}$ ($1.25 \cdot P \cdot R / (0.25 P + R)$), pushing candidate scores upward causes excess false positives on borderline pairs, lowering S1 Macro Precision from 0.8453 down to 0.8286 and reducing Macro-$F_{0.5}$ to **0.7607**.

**Empirical Finding:** Mild class weighting (`scale_pos_weight=5.18`) achieves the optimal balance between high precision and boundary stability. Full inverse-frequency class weighting hurts the $F_{0.5}$ competition objective.

---

## 4. Model Hyperparameters

The final model configuration for the winning estimator (**Model C1: XGBoost Mild Weight**) in `src/model.py`:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=5,
    gamma=0.10,
    reg_alpha=0.10,
    reg_lambda=1.00,
    scale_pos_weight=5.18,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
```

---

## 5. Pair-Level Diagnostic Metrics

These diagnostic classification metrics assess the raw discriminative capacity of the model scores on the 71,106 held-out validation candidate pairs:

| Supervised Model | Pair ROC-AUC | Pair PR-AUC | Pair Log-Loss | Pair Brier Score | Training Time | Validation Scoring Time | Validation Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Logistic Regression)** | 0.9984 | 0.9671 | 0.0541 | 0.0140 | 4.57 s | 0.051 s | 1,401,841 pairs/s |
| **Model B (XGBoost Default)** | **0.9994** | **0.9870** | **0.0104** | **0.0029** | 3.11 s | 0.037 s | **1,935,029 pairs/s** |
| **Model C1 (XGBoost Mild Weight)** | **0.9994** | 0.9869 | 0.0132 | 0.0038 | 3.58 s | 0.065 s | 1,085,291 pairs/s |
| **Model C2 (XGBoost Full Weight)** | 0.9993 | 0.9863 | 0.0215 | 0.0062 | 3.43 s | 0.041 s | 1,712,784 pairs/s |
| **Model C3 (Random Forest)** | 0.9990 | 0.9794 | 0.0289 | 0.0075 | 6.36 s | 0.083 s | 859,507 pairs/s |

---

## 6. S1-Level Macro-$F_{0.5}$ Evaluation

Candidate pairs were aggregated by Source 1 entity, and scored against ground truth using the official challenge semantics in `src/evaluate.py`:

$$\text{Precision}_{S1} = \frac{\lvert \text{True} \cap \text{Pred} \rvert}{\lvert \text{Pred} \rvert}, \quad \text{Recall}_{S1} = \frac{\lvert \text{True} \cap \text{Pred} \rvert}{\lvert \text{True} \rvert}, \quad F_{0.5, S1} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

- True empty + predicted empty $\to P=1.0, R=1.0, F_{0.5}=1.0$
- True empty + false predicted match $\to P=0.0, R=0.0, F_{0.5}=0.0$
- True non-empty + predicted empty $\to P=0.0, R=0.0, F_{0.5}=0.0$

| Supervised Model | Decision Threshold | S1 Macro-$F_{0.5}$ | S1 Macro Precision | S1 Macro Recall | Total Predicted Matches | Avg Matches / S1 | % S1 Predicted Empty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Logistic Regression)** | 0.90 | 0.6794 | 0.7290 | 0.6325 | 2,830 | 2.77 | 12.2% |
| **Model B (XGBoost Default)** | 0.60 | 0.7708 | 0.8468 | 0.6474 | 2,312 | 2.26 | 18.3% |
| **Model C1 (XGBoost Mild Weight)** | **0.80** | **0.7712** | **0.8453** | **0.6528** | 2,357 | 2.31 | 17.4% |
| **Model C2 (XGBoost Full Weight)** | 0.90 | 0.7607 | 0.8286 | 0.6581 | 2,465 | 2.41 | 16.0% |
| **Model C3 (Random Forest)** | 0.80 | 0.7546 | 0.8278 | 0.6423 | 2,388 | 2.34 | 17.3% |

**Key Finding:** XGBoost demonstrates clear superiority over the linear baseline (+0.0918 Macro-$F_{0.5}$, a **+13.5% relative gain**), driven by non-linear interaction features between addresses, names, and cross-script indicators.

---

## 7. Comprehensive Threshold Search

Using `eda/tune_threshold.py`, we conducted a two-stage threshold search across the validation set:

### Stage 1: Coarse Grid (0.05 to 0.95 in steps of 0.05)

| Threshold ($\tau$) | Validation Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Total Predicted Matches | % S1 Predicted Empty | Avg Matches / S1 | True Zero-Match False Positive % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.05 | 0.6268 | 0.6459 | 0.6563 | 3,604 | 6.9% | 3.53 | 61.8% |
| 0.10 | 0.6658 | 0.6962 | 0.6626 | 3,280 | 9.3% | 3.21 | 51.5% |
| 0.20 | 0.6966 | 0.7365 | 0.6641 | 3,000 | 11.4% | 2.94 | 44.1% |
| 0.30 | 0.7208 | 0.7695 | 0.6672 | 2,828 | 12.9% | 2.77 | 32.4% |
| 0.40 | 0.7320 | 0.7848 | 0.6668 | 2,728 | 14.2% | 2.67 | 26.5% |
| 0.50 | 0.7406 | 0.7978 | 0.6631 | 2,634 | 14.9% | 2.58 | 26.5% |
| 0.60 | 0.7576 | 0.8205 | 0.6655 | 2,541 | 15.8% | 2.49 | 16.2% |
| 0.70 | 0.7628 | 0.8313 | 0.6584 | 2,454 | 16.3% | 2.40 | 16.2% |
| 0.75 | 0.7692 | 0.8410 | 0.6564 | 2,397 | 17.0% | 2.35 | 11.8% |
| **0.80** | **0.7712** | **0.8453** | **0.6528** | **2,357** | **17.4%** | **2.31** | **11.8%** |
| 0.85 | 0.7698 | 0.8463 | 0.6455 | 2,308 | 18.3% | 2.26 | 10.3% |
| 0.90 | 0.7665 | 0.8476 | 0.6355 | 2,242 | 19.3% | 2.19 | 7.4% |
| 0.95 | 0.7668 | 0.8561 | 0.6223 | 2,140 | 19.9% | 2.09 | 1.5% |

### Stage 2: Fine Grid (0.70 to 0.90 in steps of 0.01)

| Threshold ($\tau$) | Validation Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Total Predicted Matches | % S1 Predicted Empty | Avg Matches / S1 | True Zero-Match False Positive % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.75 | 0.7692 | 0.8410 | 0.6564 | 2,397 | 17.0% | 2.35 | 11.8% |
| 0.76 | 0.7697 | 0.8423 | 0.6553 | 2,388 | 17.0% | 2.34 | 11.8% |
| 0.77 | 0.7703 | 0.8428 | 0.6548 | 2,379 | 17.2% | 2.33 | 11.8% |
| 0.78 | 0.7703 | 0.8434 | 0.6537 | 2,371 | 17.2% | 2.32 | 11.8% |
| 0.79 | 0.7705 | 0.8440 | 0.6533 | 2,367 | 17.2% | 2.32 | 11.8% |
| 0.80 | 0.7712 | 0.8453 | 0.6528 | 2,357 | 17.4% | 2.31 | 11.8% |
| **0.81** | **0.7715** | **0.8457** | **0.6525** | **2,348** | **17.7%** | **2.30** | **10.3%** |
| 0.82 | 0.7707 | 0.8447 | 0.6511 | 2,335 | 18.1% | 2.28 | 10.3% |
| 0.83 | 0.7698 | 0.8446 | 0.6484 | 2,324 | 18.3% | 2.27 | 10.3% |
| 0.84 | 0.7699 | 0.8447 | 0.6484 | 2,323 | 18.3% | 2.27 | 10.3% |
| 0.85 | 0.7698 | 0.8463 | 0.6455 | 2,308 | 18.3% | 2.26 | 10.3% |

---

## 8. Selected Threshold and Multi-Match Decision Rule

### Selected Threshold
The threshold selected strictly based on validation Macro-$F_{0.5}$ is:
$$\mathbf{\tau^* = 0.81}$$
At $\tau^* = 0.81$:
- **Macro-$F_{0.5}$:** **0.7715**
- **Macro Precision:** **0.8457**
- **Macro Recall:** **0.6525**
- **Avg Predicted Matches:** **2.30 matches / S1 entity**
- **S1 Predicted Empty:** **17.7%**

### Decision Rule Investigation (Multi-Match vs Single-Match / Argmax)

| Decision Rule Strategy | Decision Mechanism | Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Avg Matches / S1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Rule 1 (Standard Multi-Candidate Threshold)** | Accept all candidate IDs where $P(\text{match}) \ge \tau^*$ | **0.7715** | 0.8457 | **0.6525** | **2.30** |
| **Rule 2 (Single-Match Forcing: Argmax Only)** | Predict only $\operatorname{argmax} P(\text{match})$ if $\max(P) \ge \tau^*$ | 0.5960 | **0.8689** | 0.3196 | 0.82 |
| **Rule 3 (Threshold + Margin Rule)** | $P \ge \tau^*$ and $P \ge (\max(P) - 0.15)$ | 0.7719 | 0.8488 | 0.6478 | 2.27 |

> **Argmax Collapse:** Forcing a single match per Source 1 entity via $\operatorname{argmax}$ causes Macro-$F_{0.5}$ to collapse from **0.7715 down to 0.5960** (-22.7% relative drop). Because true Source 1 entities have on average 3.47 matches across S2 and S3, predicting multiple matches when scores exceed $\tau^*$ is mathematically critical.
> 
> Furthermore, Rule 3 (Threshold + Margin) yields practically identical performance (0.7719 vs 0.7715) to Rule 1. Following the rule against premature ad-hoc complexity, **Rule 1 ($P \ge 0.81$) is adopted**.

---

## 9. Diagnostic Performance by Entity Group

Performance broken down by ground-truth match count at $\tau^* = 0.81$:

| Diagnostic Entity Group | Definition | Entity Count | Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Avg Predicted Matches | False Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group 1: True Singletons** | Exactly 1 true match in GT | 60 | 0.5944 | 0.5917 | 0.6167 | 0.72 | 13.95% |
| **Group 2: True Zero-Match Entities** | 0 matches in GT (empty) | 68 | **0.8971** | **0.8971** | **0.8971** | **0.12** | **10.29%** |
| **Group 3: True Multi-Match Entities** | $\ge 2$ true matches in GT | 894 | **0.7738** | **0.8589** | **0.6363** | **2.57** | **3.83%** |

### Group-Specific Insights:
1. **True Zero-Match Entities (Group 2):**
   - 89.71% (61 of 68) of true-empty validation entities had **zero** candidates exceeding $\tau^* = 0.81$.
   - Under official challenge semantics ($\text{empty} + \text{empty} \to 1.0$), these 61 entities received a perfect score of $F_{0.5} = 1.0000$.
   - Only 7 entities had a false alarm (FP rate 10.29%), confirming that $\tau^* = 0.81$ protects against false positives on empty entities.
2. **True Singletons (Group 1):**
   - Singletons are the most challenging entity type ($F_{0.5} = 0.5944$).
   - For singletons, predicting a single correct match gives $F_{0.5}=1.0$, but if a second candidate also exceeds threshold, precision drops from 1.0 to 0.5, dragging $F_{0.5}$ to 0.625.
3. **True Multi-Match Entities (Group 3):**
   - Accounts for 87.5% of validation entities.
   - Reaches strong precision (0.8589) and healthy recall (0.6363), generating 2.57 matches on average per entity.

---

## 10. Probability Calibration Analysis

Using `eda/tune_threshold.py`, we inspected raw model probability distributions and compared uncalibrated trees against Platt (Sigmoid) scaling:

### Score Distributions for Positive vs Negative Pairs

| Pair Ground Truth | Pair Count | Mean Score | Median Score | 10th Percentile | 90th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Positive Pairs ($y=1$)** | 2,422 | **0.9522** | **0.9991** | 0.9000 | 0.9999 |
| **Negative Pairs ($y=0$)** | 68,684 | **0.0061** | **0.0000** | 0.0000 | 0.0007 |

### Calibration Impact on Macro-$F_{0.5}$

| Calibration Method | Best Decision Threshold | S1 Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Pair Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw XGBoost Scores (Model C1)** | **0.81** | **0.7715** | 0.8457 | 0.6525 | 0.0038 |
| **Platt Calibrated (Sigmoid)** | 0.75 | 0.7706 | 0.8432 | 0.6510 | 0.0033 |

**Conclusion:** The raw tree model produces sharp bimodal separation (positive median 0.9991 vs negative median 0.0000). Platt scaling slightly reduces the Brier score (0.0038 to 0.0033) but does not improve Macro-$F_{0.5}$ (0.7715 vs 0.7706). In accordance with the prompt guidance, we **retain the clean uncalibrated tree scores** to avoid extra calibration overhead and ensure stability.

---

## 11. Comparison of Blocking Priority Caps (Cap 60 vs Cap 100 vs Cap 150)

Using the exact same trained model and feature pipeline, we evaluated the effect of Priority Caps 60, 100, and 150 on the held-out validation set:

| Priority Cap | Total Candidates | Avg Candidates / S1 | Blocker Recall % | S1 Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Avg Predicted Matches / S1 | Scoring Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cap 60** | 47,283 | 46.3 / S1 | 64.34% | 0.7483 | 0.8288 | 0.6204 | 2.17 | 0.04 s |
| **Cap 100** | 71,106 | 69.6 / S1 | 68.32% | 0.7715 | 0.8457 | 0.6525 | 2.30 | 0.06 s |
| **Cap 150** | **93,860** | **91.8 / S1** | **72.75%** | **0.7961** | **0.8653** | **0.6850** | **2.43** | **0.09 s** |

### Critical Takeaways on Blocking Caps:
1. **Cap 150 is the Clear Winner:** Expanding from Cap 100 to Cap 150 yields a major jump in validation Macro-$F_{0.5}$ from **0.7715 to 0.7961** (+0.0246, a **+3.2% relative gain**).
2. **Precision Actually Increases:** Counterintuitively, expanding the cap to 150 increases Macro Precision from 0.8457 to **0.8653** while boosting Macro Recall from 0.6525 to **0.6850**. Because our XGBoost classifier is precise (ROC-AUC 0.9994), additional true matches brought into ranks 101–150 are correctly accepted, while hard negatives are cleanly rejected.
3. **Computational Feasibility:** Scoring 93,860 pairs at Cap 150 took only **0.09 seconds** (over 1,000,000 pairs/sec inference throughput). The extra memory footprint is negligible (~28 MB).
4. **Recommendation for Test Inference:** Cap 150 should be adopted as the primary candidate pool for inference.

---

## 12. Model Feature Importance

Top 20 features ranked by Gain Importance from `eda/analyze_errors.py`:

| Rank | Feature Name | Feature Group | Gain Importance | Description |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `address_token_overlap_coef` | Group C: Address | **0.3023** | Szymkiewicz-Simpson overlap coefficient between address word tokens. |
| **2** | `high_name_sim_and_address_overlap` | Group D: Interactions | **0.1458** | Joint indicator: Levenshtein $\ge 0.85$ AND Address Jaccard $\ge 0.30$. |
| **3** | `exact_name_and_address_number_match` | Group D: Interactions | **0.0820** | Joint indicator: Exact core name AND matching building/street number. |
| **4** | `address_token_containment` | Group C: Address | **0.0695** | Max containment ratio of address tokens (handles subset/expanded addresses). |
| **5** | `address_token_jaccard` | Group C: Address | **0.0615** | Token Jaccard similarity across non-stopword address tokens. |
| **6** | `address_missing_cand` | Group C: Address | **0.0579** | Indicator for null/empty address in candidate record (penalty signal). |
| **7** | `address_numeric_token_jaccard` | Group C: Address | **0.0576** | Jaccard similarity across numeric building/flat/plot numbers. |
| **8** | `name_char_3gram_jaccard` | Group B: Name | **0.0224** | Character 3-gram Jaccard similarity (typo-resilient name match). |
| **9** | `shared_name_and_shared_number` | Group D: Interactions | **0.0211** | Shared business word AND shared address numeric token. |
| **10** | `name_token_containment_cand` | Group B: Name | **0.0169** | Fraction of candidate name tokens contained in S1 name. |
| **11** | `blocking_rank_order` | Group F: Provenance | **0.0095** | Deterministic candidate rank from multi-signal blocking engine. |
| **12** | `name_jaro_winkler_sim` | Group B: Name | **0.0079** | Prefix-weighted Jaro-Winkler string similarity. |
| **13** | `cross_script_and_strong_address` | Group D: Interactions | **0.0078** | Cross-script flag AND strong address token/number match. |
| **14** | `address_first_token_match` | Group C: Address | **0.0074** | Binary match of primary street/building address token. |
| **15** | `name_both_latin` | Group B: Name | **0.0073** | Both names in standard Latin script. |
| **16** | `name_token_jaccard` | Group B: Name | **0.0070** | Token Jaccard similarity across business name words. |
| **17** | `candidate_is_source3` | Group E: Source Noise | **0.0069** | Indicator that candidate is from Source 3. |
| **18** | `address_building_num_match` | Group C: Address | **0.0061** | Exact match on primary extracted street/building number. |
| **19** | `candidate_is_source2` | Group E: Source Noise | **0.0057** | Indicator that candidate is from Source 2. |
| **20** | `name_cross_script` | Group B: Name | **0.0048** | Flag indicating Latin vs Devanagari script difference. |

### Importance by Feature Group

| Feature Group | Active Features | Total Gain Sum | % Total Model Gain | Mean Gain / Feature |
| :--- | :---: | :---: | :---: | :---: |
| **Group C: Address Similarities** | 17 | **0.5796** | **58.0%** | **0.0341** |
| **Group D: Cross-Field Interactions** | 7 | **0.2581** | **25.8%** | **0.0369** |
| **Group B: Name Similarities** | 24 | **0.1082** | **10.8%** | 0.0045 |
| **Group F: Blocking Provenance** | 11 | **0.0345** | **3.5%** | 0.0031 |
| **Group E: Source-Specific Signals** | 4 | **0.0195** | **2.0%** | 0.0049 |
| **Group A: Country Invariants** | 2 | 0.0000 | 0.0% | 0.0000 |

> **Dominance of Address and Interaction Features:** Address features (Group C) and cross-field interaction features (Group D) account for **83.8% of total model gain**. Name similarity alone is insufficient because corporate names frequently share generic terms ("Enterprises", "Holdings"), while address overlap provides location disambiguation. Group A (Country) contributes 0.0% gain because country equality was already enforced as a 100% hard blocking invariant.

---

## 13. Deep Error Analysis

We conducted an audit of 20 high-confidence False Positives and 20 low-confidence False Negatives using `eda/analyze_errors.py`:

### Audit Summary: 20 False Positives (Model Score $\ge 0.81$, Ground Truth = 0)
1. **Franchises & Brand Name Clones at Different Addresses (45% of FPs):**
   - *Example:* S1 `Medyne Adr LLC` (1732 Sanford St, Arlington, TX) vs Candidate `Medyne Adr Llc` (No address). Score: 0.9682.
   - *Signal:* Exact brand name match, but candidate has missing address in S2/S3. The model assigns high probability based on near-perfect name similarity.
2. **Co-Located / Multi-Tenant Commercial Buildings (30% of FPs):**
   - *Example:* S1 `Giga Global Inc` (104 Main St, Suite 200, Austin, TX) vs Candidate `Apex Solutions LLC` (104 Main St, Austin, TX). Score: 0.8841.
   - *Signal:* Different business names sharing the exact same commercial building address and postal code.
3. **Shared Generic Legal / Industry Suffixes (15% of FPs):**
   - *Example:* S1 `High Agro Limited` (Pune, Maharashtra) vs Candidate `High Agro Ltd Enterprises` (No address). Score: 0.8412.
   - *Signal:* Name differences caused by combinations of generic words ("Enterprises", "Limited", "Services").
4. **True Missing Ground Truth in S2/S3 (10% of FPs):**
   - Several false positives appear to be genuine real-world business matches that were unannotated in the competition ground truth.

### Audit Summary: 20 False Negatives (Model Score $< 0.81$, Ground Truth = 1)
1. **Missing Candidate Address (40% of FNs):**
   - *Example:* S1 `Anand Business Limited` (Saha Bhawan, Circus Market Place, Kolkata) vs Candidate `Anand Business Enterprises` (Address: `None`). Score: 0.0033.
   - *Signal:* When candidate address is missing, the model cannot utilize its strongest features (Group C: 58.0% of gain), causing the probability to fall below threshold.
2. **Extreme Aliases / Trade Names (30% of FNs):**
   - *Example:* S1 `Elevate Professional Center` (74B, Lowther Road, Allahabad) vs Candidate `Miraumbra` (74B, Lowther Road, Allahabad, उत्तर प्रदेश). Score: 0.0192.
   - *Signal:* Trade name and legal registration name are entirely dissimilar words. Even with matching addresses, the tree downweights pairs with near-zero name token overlap.
3. **Heavy Typographical / Concatenation Errors (20% of FNs):**
   - *Example:* S1 `Maharathi Televisions Pvt Ltd` vs Candidate `MAHARATHI TECVGENISIONS PVT PVT LTD`. Score: 0.0476.
   - *Signal:* Severe OCR/spelling corruption in the core name token reduces Jaccard and Levenshtein metrics.
4. **Cross-Script Native Address Abbreviations (10% of FNs):**
   - *Example:* S1 `Da (India) Agro Limited` (Flat 1703, Brighton Tower, Lokhandwala, Mumbai) vs Candidate `Da (India) Agro Limited` (Address: `17-03, MH`). Score: 0.0051.
   - *Signal:* Address is abbreviated to state code ("MH") and flat number, causing address token overlap to be lower than typical matches.

---

## 14. Hard Subset & Source-Specific Analysis

### Cross-Script Performance (Devanagari vs Latin)
- Total Cross-Script Validation Pairs: **1,302 pairs**
- Ground-Truth True Matches: 59
- **True Positives Recovered by Model:** **51 out of 59 (86.4% Recall)**
- False Negatives: 8
- False Positives: 11
- **Finding:** Despite 0.0 Latin-character name similarity, our interaction feature `cross_script_and_strong_address` (Feature 47) and numeric address alignment successfully recovered **86.4%** of cross-script true matches.

### Source 2 vs Source 3 Candidates

| Candidate Source | Evaluated Pairs | True Matches | Predicted Matches | True Positives | False Positives | Pair Precision | Pair Recall | Pair $F_{0.5}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Source 2** | 45,086 | 1,245 | 1,212 | 1,155 | 57 | **0.9530** | 0.9277 | **0.9478** |
| **Source 3** | 26,020 | 1,177 | 1,145 | 1,096 | 49 | **0.9572** | 0.9312 | **0.9519** |

**Finding:** Performance is balanced across Source 2 and Source 3 (Pair $F_{0.5}$ of 0.9478 vs 0.9519). Source 3 candidates have slightly higher precision due to fewer missing address fields.

### Country-Specific Performance (US vs India)

| Country Subset | Validation S1 Entities | Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Avg Predicted Matches | % S1 Predicted Empty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **United States (US)** | 621 | **0.7834** | 0.8647 | 0.6502 | 2.29 / S1 | 15.6% |
| **India** | 401 | **0.7523** | 0.8151 | 0.6568 | 2.33 / S1 | 20.2% |

**Finding:** US entities achieve slightly higher Macro-$F_{0.5}$ (0.7834 vs 0.7523) because US addresses follow standardized postal and street naming conventions. India entities have higher address variability, but maintain over 0.81 Macro Precision. A single unified model successfully handles both countries without requiring separate models.

---

## 15. Runtime, Throughput, and Memory Profiling

| Pipeline Stage | Implementation | Processed Volume | Execution Time | Processing Speed | Peak Memory |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate Blocking** | `src/blocking.py` (DuckDB) | 5,000 S1 records | 26.10 s | 17,488 pairs/s | ~180 MB |
| **Feature Extraction** | `src/features.py` (NumPy/RapidFuzz) | 456,432 pairs | 32.37 s | **14,099 pairs/s** | ~230 MB |
| **Model Training** | `src/model.py` (XGBoost 3.2.0) | 278,884 pairs | **3.58 s** | 77,898 pairs/s | ~160 MB |
| **Validation Inference** | `src/model.py` (XGBoost 3.2.0) | 93,860 pairs | **0.09 s** | **1,042,888 pairs/s** | ~28 MB |

---

## 16. Reproducibility Configuration

All artifacts, configurations, and scripts are fully deterministic and reproducible:

```yaml
phase: "Phase 5: Supervised Model Training & Calibration"
model_architecture: "XGBClassifier (Hist Tree Method)"
library_versions:
  xgboost: "3.2.0"
  scikit-learn: "1.7.1"
  rapidfuzz: "3.14.3"
  duckdb: "1.4.3"
  pyarrow: "23.0.0"
random_seeds:
  dataset_split_seed: 42
  model_seed: 42
hyperparameters:
  n_estimators: 300
  max_depth: 5
  learning_rate: 0.08
  subsample: 0.80
  colsample_bytree: 0.80
  min_child_weight: 5
  gamma: 0.10
  reg_alpha: 0.10
  reg_lambda: 1.00
  scale_pos_weight: 5.18
  eval_metric: "logloss"
feature_schema_version: "canonical_65_v1.0"
feature_count: 65
feature_schema_file: "src/feature_schema.py"
blocker_channels: "A, A2, B, C, D, E, E2, G"
blocking_cap_default: 100
blocking_cap_recommended: 150
train_s1_count: 4000
val_s1_count: 1000
selected_decision_threshold: 0.81
calibration_applied: false  # Raw tree probabilities empirically superior
```

---

## 17. Exact Files Created and Modified

1. `src/model.py` *(Created)*:
   - Implements `EntityMatcherModel` with unified support for Logistic Regression, XGBoost, Random Forest, probability calibration, and feature importances.
2. `src/evaluate.py` *(Created)*:
   - Implements official S1-level Macro-$F_{0.5}$ evaluation logic with exact zero-match handling, group partitioning, and pair diagnostic metrics.
3. `src/feature_schema.py` *(Existing, Authoritative)*:
   - Canonical 65-feature schema specification with strict column ordering and documentation.
4. `src/features.py` *(Existing, Synchronized)*:
   - High-throughput pairwise feature extractor operating at ~14,100 pairs/sec.
5. `eda/prepare_training_data.py` *(Created)*:
   - Generates the held-out candidate dataset up to Cap 150, attaches ground truth labels, performs S1-level grouped splitting, and extracts features.
6. `eda/benchmark_models.py` *(Created)*:
   - Benchmarks Models A, B, C1, C2, and C3 on training runtimes, ROC-AUC, PR-AUC, and preliminary S1 Macro-$F_{0.5}$.
7. `eda/tune_threshold.py` *(Created)*:
   - Runs coarse and fine threshold sweeps, group performance breakdown (singletons, zero-matches, multi-matches), calibration audits, and Cap 60 vs 100 vs 150 comparisons.
8. `eda/analyze_errors.py` *(Created)*:
   - Computes feature importances (Gain), audits 20 false positives and 20 false negatives, and breaks down cross-script and source-specific behaviors.

---

## 18. Executive Summary & Recommended Strategy

1. **Winning Model:** **Model C1 (XGBoost Mild Weight)** achieves **0.7715 Macro-$F_{0.5}$** at Cap 100 and **0.7961 Macro-$F_{0.5}$** at Cap 150.
2. **Optimal Decision Threshold:** $\mathbf{\tau^* = 0.81}$. Simple thresholding ($P \ge 0.81$) strictly outperforms single-match forcing ($\operatorname{argmax}$ causes a 22.7% performance collapse).
3. **Blocking Cap Selection:** Expanding from Priority Cap 100 to **Priority Cap 150** increases Macro-$F_{0.5}$ by **+3.2% relative** (from 0.7715 to **0.7961**), improving both precision (0.8653) and recall (0.6850).
4. **Primary Drivers:** Address token overlap (30.2% gain) and cross-field interactions (25.8% gain) are the strongest predictors.
5. **Phase Gate Adherence:** In accordance with user instructions, execution has **STOPPED**. No test predictions have been generated, no submission TSV files have been exported, and no final ZIP archive has been created. We await your review and authorization before proceeding to Phase 6.
