# Business Entity Resolution — Reproduction Guide

## Overview

This pipeline resolves business entities across three independent data sources (S1, S2, S3) using multi-strategy blocking followed by an XGBoost classifier.

## Environment Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+. All packages are available via pip.

## Directory Structure

```
student_resource/
├── dataset/
│   ├── train/   (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/    (test_source1/2/3.tsv)
├── output/      (generated: matching_results.tsv, candidate_pairs.tsv)
├── utils/
│   └── validate_submission.py
└── code/business_entity_resolution/
    ├── src/
    │   └── entity_resolution.py   ← main pipeline
    ├── requirements.txt
    └── README.md
```

## Running the Pipeline

### Option 1: Evaluate on validation split (train mode)
Trains on 80% of training data, evaluates on held-out 20%, and reports validation F_0.5:

```bash
cd student_resource/
python code/business_entity_resolution/src/entity_resolution.py --mode train
```

### Option 2: Generate final predictions (test mode — default)
Trains on ALL training data, generates predictions for the test set, and writes:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

```bash
cd student_resource/
python code/business_entity_resolution/src/entity_resolution.py --mode test
```

### Validate output before submitting

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Architecture

### Stage 1: Text Normalization
- Unicode → ASCII (handles Devanagari/Kannada transliterations)
- Legal suffix normalization: `Private Limited` → `pvt ltd`, `LLC`, `Inc`, etc.
- Address abbreviation normalization: `Street` → `st`, `Road` → `rd`, etc.
- Removal of noise: URLs, `dba` keywords, punctuation

### Stage 2: Blocking (Candidate Generation)
Three complementary strategies are merged to maximize recall:

1. **Inverted Token Index** — Build a token→[entity_ids] index on S2/S3 names. For each S1 entity, look up all S2/S3 entities sharing ≥1 rare name token (tokens in <500 docs). Handles typos via shared subword tokens.

2. **Country + Name Prefix Blocking** — Group by `(country, first_4_chars_of_name)`. Entities in the same group become candidates. Reduces cross-country false positives.

3. **TF-IDF Cosine Similarity** — Fit a character 2-3gram TF-IDF on all texts. For each S1 entity, find top-30 most similar S2/S3 entities by cosine similarity (batched sparse matrix multiply).

All three candidate sets are merged, capped at 50 candidates per S1 entity.

### Stage 3: Feature Engineering
16 pairwise similarity features per candidate pair:
- **Name (8 features)**: token_set_ratio, token_sort_ratio, partial_ratio, ratio, WRatio, bigram Jaccard, token Jaccard, length ratio
- **Address (5 features)**: token_set_ratio, token_sort_ratio, partial_ratio, ratio, token Jaccard, numeric token overlap
- **Country match (1 feature)**
- **Combined name+address (1 feature)**: token_set_ratio on concatenated text

### Stage 4: XGBoost Classifier
- 300 trees, max_depth=6, learning_rate=0.1
- Class imbalance handled via `scale_pos_weight = neg/pos`
- Threshold tuned to maximize macro F_0.5 on held-out validation

## Key Design Decisions

- **Precision-heavy**: F_0.5 metric penalizes false positives 2× more than false negatives. The threshold is tuned on F_0.5, and caps on blocking (50 candidates/entity) prevent precision degradation.
- **Scalable blocking**: The inverted index and TF-IDF block pass use sparse matrix operations and run in O(n) memory. No all-pairs comparison.
- **Open country support**: Country is treated as a raw string — no hard-coded country list. France (unseen in training) is handled transparently.

## Expected Runtime

| Step | Approximate Time |
|------|----------------|
| Data loading + normalization | ~10 min |
| TF-IDF blocking (2.2M × 10M) | ~30 min |
| Token blocking | ~15 min |
| Feature extraction | ~20 min |
| Model training | ~5 min |
| Inference + output | ~5 min |
| **Total** | **~85 min** |

(Measured on a machine with 32 GB RAM and 8 CPU cores)
