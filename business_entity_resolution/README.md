# Business Entity Resolution — Reproduction Guide

## Overview

This pipeline resolves business entities across three independent data sources (S1, S2, S3) using multi-strategy blocking followed by a CPU gradient-boosted classifier. It uses only the supplied files and makes no external identity lookups.

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
└── business_entity_resolution/
    ├── src/
    │   └── entity_resolution.py   ← main pipeline
    ├── requirements.txt
    └── README.md
```

## Running the Pipeline

### Train, validate, and generate final predictions

The main pipeline trains a gradient-boosted matcher, selects its threshold on an
entity-disjoint held-out validation split, then writes:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

```bash
cd /path/to/AMAZON
python3 -m pip install -r business_entity_resolution/requirements.txt
python3 business_entity_resolution/src/entity_resolution.py --mode test
```

For a validation-only run (no test output), use
`python3 business_entity_resolution/src/entity_resolution.py --mode train`.
Test labels are intentionally not supplied, so a local result cannot guarantee a
particular test accuracy.

### Validate output before submitting

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Git LFS data

The large TSVs are stored through Git LFS. Before running, a data file must begin
with `entity_id`, not `version https://git-lfs...`. If the latter appears, install
Git LFS and hydrate the data:

```bash
git lfs install
git lfs pull
```

The program detects an unhydrated pointer and exits with this actionable error.

### Docker

From the repository root, after hydrating Git LFS:

```bash
docker build -f business_entity_resolution/Dockerfile -t business-er .
docker run --rm -v "$PWD/output:/app/output" business-er
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Pipeline Architecture

### Stage 1: Text Normalization
- Unicode → ASCII (handles Devanagari/Kannada transliterations)
- Legal suffix normalization: `Private Limited` → `pvt ltd`, `LLC`, `Inc`, etc.
- Address abbreviation normalization: `Street` → `st`, `Road` → `rd`, etc.
- Removal of noise: URLs, `dba` keywords, punctuation

### Stage 2: Blocking (Candidate Generation)
Three complementary strategies are merged to maximize recall:

1. **Exact normalized field blocking** — Retrieve same-country records with an exact normalized name or address.

2. **SQLite FTS lexical blocking** — Independently query normalized name and address tokens against a disk-backed full-text index, so one missing field does not prevent a candidate.

3. **Fuzzy reranking** — Merge the blocks and retain the top 80 candidates by field-level token similarity before final feature scoring.

The candidate cap is configurable with `--candidates-per-entity`.

### Stage 3: Feature Engineering
16 pairwise similarity features per candidate pair:
- **Name (8 features)**: token_set_ratio, token_sort_ratio, partial_ratio, ratio, WRatio, bigram Jaccard, token Jaccard, length ratio
- **Address (5 features)**: token_set_ratio, token_sort_ratio, partial_ratio, ratio, token Jaccard, numeric token overlap
- **Country match (1 feature)**
- **Combined name+address (1 feature)**: token_set_ratio on concatenated text

### Stage 4: Gradient-Boosted Classifier
- 300 iterations, fixed seed, CPU-only inference
- Class imbalance handled with positive sample weights
- Threshold tuned to maximize macro F_0.5 on held-out validation

## Key Design Decisions

- **Precision-heavy**: F_0.5 metric penalizes false positives 2× more than false negatives. The threshold is tuned on F_0.5, and caps on blocking (50 candidates/entity) prevent precision degradation.
- **Scalable blocking**: The inverted index and TF-IDF block pass use sparse matrix operations and run in O(n) memory. No all-pairs comparison.
- **Open country support**: Country is treated as a raw string — no hard-coded country list. France (unseen in training) is handled transparently.

## Expected Runtime

| Step | Approximate Time |
|------|----------------|
| In-memory data loading + candidate generation | 1–3 hours |
| Pair features, validation, and model fitting | 1–3 hours |
| Test candidate scoring + TSV output | 1–3 hours |
| **Total (SSD, 32 GB RAM, 8 modern cores)** | **3–9 hours** |

Runtime is dominated by memory capacity and candidate feature generation; 32 GB
RAM is the practical minimum for the full supplied data. The blocker uses only
country-aware inverted postings and does not materialise an all-pairs TF-IDF matrix.
