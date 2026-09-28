# Amazon ML Challenge 2026: Business Entity Resolution
## Production Pipeline & Reproducibility Guide (Config D1: 76 Features)

**Team Name:** BrawlDevs  
**Architecture:** Multi-Channel Blocking (Enhanced Blocker V2, Cap 200) + Gradient-Boosted Decision Trees (XGBoost C1) + Group K Address Disambiguation + Multi-Match Thresholding ($\tau^* = 0.88$)  
**Validated Fresh Holdout Benchmark:** Macro-$F_{0.5} = \mathbf{0.8351}$ (Precision: **0.9024**, Recall: **0.7206**, End-to-End Recall: **71.33%**, Singleton Macro-$F_{0.5}$: **0.7422**)  
**License Compliance:** 100% Offline, Permitted Under MIT & Apache 2.0. No External APIs, No Internet Enrichment, No Geocoding.

---

## 1. Problem Formulation & Objective

The Amazon ML Challenge 2026 addresses high-scale, cross-source business entity resolution across three unlinked corpuses:
* **Source 1 ($S_1$):** Deduplicated target business records.
* **Source 2 ($S_2$) & Source 3 ($S_3$):** Noisy candidate business records with duplicate listings, typographical variations, and missing fields.

### Evaluation Metric: Macro-$F_{0.5}$
The competition evaluates performance using Macro-$F_{0.5}$ calculated strictly across deduplicated Source 1 entities:
$$\text{Precision} = \frac{|P \cap T|}{|P|}, \quad \text{Recall} = \frac{|P \cap T|}{|T|}$$
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

With $\beta = 0.5$, precision is weighted twice as heavily as recall. False-positive predictions are severely penalized.

---

## 2. Preprocessing & Multi-Representation Normalization

Raw text strings are preprocessed via a multi-representation hierarchy:
1. **Unicode NFKC Normalization & Mojibake Repair:** Fixes corrupted UTF-8 sequences (e.g. `Ã©` $\to$ `é`, smart quotes, non-breaking spaces) while strictly preserving non-Latin native scripts (Devanagari, Tamil, Cyrillic).
2. **Legal Suffix Standardization:** Identifies and strips corporate entity designators across English (`Pvt Ltd`, `LLC`, `Corp`, `Inc`), French (`SARL`, `SAS`, `EURL`, `SA`), and Indic legal forms without mutating core trade names.
3. **Prefix & Noise Token Normalization:** Normalizes common business prefixes (`The `, `M/s `, `Dr `, `DBA: `, `formerly `).
4. **Alphanumeric & Sub-Token Keys:** Generates lowercase alphanumeric canonical strings, character 2-grams and 3-grams, and word tokens.
5. **Address Numeric & Postal Parsing:** Extracts primary building numbers, all numeric tokens, and 5-6 digit postal codes.

---

## 3. Candidate Generation (Enhanced Blocker V2)

To reduce the $1.73\text{M} \times 10.3\text{M} \approx 17.8\text{ trillion}$ pairwise comparison space into a high-purity candidate pool, our DuckDB-powered SQL engine applies 11 deterministic blocking channels:

| Channel | Priority | Blocking Key / Anchor | Frequency Filter | Purpose |
| :--- | :---: | :--- | :---: | :--- |
| **Channel A** | 100 | Exact Normalized Core Name | — | Strips legal suffixes & web domains |
| **Channel A2** | 95 | Prefix-Stripped Core Name | — | Normalizes trade prefixes (`The `, `M/s `, `DBA: `) |
| **Channel E2** | 85 | Address Number + Locality Anchor | DF $\le 100$ | Recovers Indic trade aliases & non-Latin native scripts |
| **Channel B** | 80 | Rare Name Tokens | DF $\in [2, 200]$ | Inverted index of informative, distinctive words |
| **Channel E** | 75 | Address Number + 3-char Name Prefix | — | Physical building match with phonetic prefix |
| **Channel E3** | 70 | Address Number + Street Token | DF $\le 50$ | Exact building and street alignment |
| **Channel H** | 65 | Postal Code + 3-char Name Prefix | — | Postal neighborhood alignment |
| **Channel G** | 60 | 1-Edit Initial Character Typo Key | DF $\le 150$ | Recovers OCR/scan errors (`0` vs `O`, `6` vs `G`) |
| **Channel I** | 55 | Distinctive Name Token Pairs | DF $\le 50$ | Multi-word business descriptor matching |
| **Channel D** | 50 | Distinctive Locality Tokens | DF $\le 150$ | Geographic neighborhood alignment |
| **Channel C** | 40 | 4-gram Prefix + Suffix Index | — | Robust fallback for truncated entities |

### Strict Blocking Invariants:
* **Country Filter:** Hard partition on `s1.country == candidate.country`. France, US, and India are processed generically as open-set fields.
* **Priority Candidate Cap:** Strictly capped at **200 candidates per S1 entity**. Candidates are ordered by $\sum \text{Priority Score}$ descending, then channels fired descending.

---

## 4. Final 76-Feature Schema (Config D1)

Candidate pairs are converted into a rich 76-dimensional pairwise feature vector:

1. **Group A: Country Invariants (2 features):** `country_exact_match`, `country_missing_either`.
2. **Group B: Name Similarities (24 features):** Exact raw, legal-stripped, domain-stripped, alphanumeric matches; Levenshtein normalized similarity; Jaro-Winkler; character 3-gram Jaccard; token Jaccard; overlap coefficient; S1/candidate containment ratios; token counts; first/last token matches; script flags (`name_both_latin`, `name_cross_script`, `name_has_non_ascii`).
3. **Group C: Address Similarities (14 features):** Exact address match; address token Jaccard, overlap, containment; shared token count; first token match; building number exact match; numeric token Jaccard; shared numeric count; length difference and ratio.
4. **Group D: Cross-Field Interactions (7 features):** `exact_name_and_exact_address`, `high_name_sim_and_address_overlap`, `shared_name_and_shared_number`, `cross_script_and_strong_address`, `address_overlap_but_low_name_sim`.
5. **Group E: Source-Specific Signals (4 features):** S2 vs S3 source indicators, URL presence in name, candidate address missingness.
6. **Group F: Blocking Provenance (14 features):** Individual channel firing indicators (`fired_chan_a` to `fired_chan_g`), total channels fired, highest priority channel, total priority score, rank order.
7. **Group G: Missing-Address Recovery (4 features):** Name token Jaccard, 3-gram, containment, and Levenshtein scaled by candidate address missingness.
8. **Group H: Alias & Trade Names (2 features):** Pairwise best token Levenshtein similarity, distinctive token Jaccard.
9. **Group I: OCR & Typo Robustness (1 feature):** Character 2-gram overlap coefficient.
10. **Group K: Address Disambiguation & False Positive Suppression (4 features):**
    * `feat_same_postal_weak_name`: Postal match with name Jaccard $< 0.20$.
    * `feat_distinctive_name_zero_overlap`: Zero overlap on non-generic business tokens.
    * `feat_same_building_weak_name`: Building number match with name Jaccard $< 0.20$.
    * `feat_high_addr_low_name_penalty`: Triggered when address Jaccard $\ge 0.60$ but name Jaccard $\le 0.25$ (penalizes co-located distinct tenants in shopping centers/complexes).

---

## 5. Supervised Model (XGBoost C1)

A single gradient-boosted decision tree ensemble (`XGBClassifier`) is trained using the exact hyperparameters:
```python
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
    "tree_method": "hist",
    "scale_pos_weight": 5.83,  # sqrt(neg / pos) on training pairs
    "random_state": 42,
}
```

### Multi-Match Decision Rule:
* **Threshold:** $\tau^* = \mathbf{0.88}$ (frozen, globally optimal for Macro-$F_{0.5}$).
* **Prediction Logic:** All candidate pairs for an S1 entity with predicted probability $P(y=1) \ge 0.88$ are accepted as true matches. If no candidate exceeds $0.88$, an empty string is emitted (zero predicted matches).

---

## 6. Reproducibility & Execution Instructions

### Prerequisites
* Python 3.8+ (64-bit)
* 16GB+ RAM recommended
* Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Full Inference
```bash
python run_pipeline.py --data-dir dataset/test --output-dir output
```

### Running Train and Inference
```bash
python run_pipeline.py --train --train-dir dataset/train --data-dir dataset/test --output-dir output
```

### Fast Smoke Test (Mock Execution)
```bash
python run_pipeline.py --smoke-test --limit 50
```

---

## 7. Submission Output Format

The pipeline generates two submission TSV files complying strictly with the official validator:
1. **`matching_results.tsv`**:
   * Header: `source1_entity_id\tmatched_entity_ids`
   * Format: `S1-0000001\tS2-0000005,S3-0000012` (empty string for zero matches).
   * Every test S1 entity appears exactly once.
2. **`candidate_pairs.tsv`**:
   * Header: `source1_entity_id\tcandidate_entity_ids`
   * Format: `S1-0000001\tS2-0000005,S2-0000088,S3-0000012`
   * Final matched entities are guaranteed to be a strict subset of candidate pairs.

---

## 8. Compliance Confirmation
* **No External APIs:** Zero calls to Google Maps, geocoding services, web search, or commercial databases.
* **No Unpermitted Data:** All models are trained solely on official competition data.
* **Model Size:** XGBoost C1 has ~300 trees of depth 5 (< 10,000 parameters, well within the 8B parameter limit).
* **Licensing:** All libraries (DuckDB, XGBoost, Scikit-Learn, RapidFuzz, Pandas) are licensed under MIT or Apache 2.0.
