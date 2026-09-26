# Exploratory Data Analysis (EDA) & Validation Scripts

This directory contains standalone, reproducible Python scripts developed during Phase 1 to analyze the dataset, uncover structural noise patterns, benchmark candidate generation rules, and verify the evaluation metric.

All scripts use `duckdb` for high-throughput streaming analysis over the multi-gigabyte dataset files in `student_resource/dataset/`.

---

## Script Index & Summary of Investigations

| Script | Purpose & Key Finding |
| :--- | :--- |
| **`eda_part1.py`** | Counts records across train and test files; analyzes country distributions in S1, S2, S3 (including France in test). |
| **`eda_ground_truth.py`** | Analyzes the ground truth structure, singleton rate (5.58%), 1-to-1 rate (5.40%), and 1-to-many match distribution (89.02%, up to 11 matches). |
| **`eda_matches_country_and_sources.py`** | Tests country consistency across all 7,638,365 ground-truth pairs. Confirms **100.0000%** of matches share the identical country (zero cross-border matches). |
| **`eda_deep_dive.py`** | In-depth linguistic analysis on 100k pairs: raw exact name match (4.64%), case-insensitive (10.71%), punctuation-stripped (21.85%), legal-suffix stripped (38.02%), address token overlap (Jaccard 0.6337), and PIN code utility (93.61% accuracy when both have PINs). |
| **`eda_sources_and_duplicates.py`** | Analyzes missing fields (3.36% of S2/S3 records have empty addresses), duplicate records within S2/S3, and inspects French test set records. |
| **`eda_unicode.py`** | Detects non-ASCII and mojibake characters (`Â€“`, `â€™`) in addresses and names across datasets. |
| **`eda_s2_non_ascii.py`** | Examines non-ASCII names in Source 2 and Source 3. Discovers native Indic scripts (Devanagari, Tamil, Gujarati) and accented Latin characters. |
| **`eda_indic_pairs.py`** | Inspects true match pairs where S1 has an English name while S2/S3 has a native Indic script name. Demonstrates that Latin addresses provide the critical link. |
| **`eda_source_comparison.py`** | Compares Source 1 (clean reference) against Source 2 and Source 3 (casing noise, web URLs, missing addresses, uppercase rates). |
| **`eda_abbreviations_and_suffixes.py`** | Quantifies top legal suffixes (`ltd`, `inc`, `corp`) and address abbreviations (`st`, `rd`, `dr`, `ave`, `ln`, `ct`). |
| **`eda_name_norm.py`** | Evaluates normalized alphanumeric matching (stripping legal suffixes, web domain extensions, spaces, and punctuation). Matches 45.28% of true pairs. |
| **`eda_phonetic_and_blocking_test.py`** | Evaluates individual blocking key recall on 50k true pairs (Soundex, 3-grams, token overlap, address overlap). |
| **`eda_blocking_benchmark.py`** | Benchmarks blocking rules against 10.3M records in DuckDB for recall vs candidate volume. Demonstrates why naive first-word blocking explodes candidates. |
| **`eda_blocking_benchmark2.py`** | Inspects random ground truth pairs to identify invariant blocking anchors across name and address components. |
| **`verify_metric.py`** | Verifies the macro-averaged $F_{0.5}$ metric calculation against the exact numerical example provided in the competition problem statement (`0.714`). |

---

## How to Run

From the root workspace directory (`d:\Github\Amazon_ML_Challenge`):

```bash
# Example: Run ground truth analysis
python eda/eda_ground_truth.py

# Example: Run country consistency check
python eda/eda_matches_country_and_sources.py

# Example: Verify evaluation metric
python eda/verify_metric.py
```
