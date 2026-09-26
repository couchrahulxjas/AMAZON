# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

Our solution employs a two-stage approach: (1) a multi-strategy blocking pipeline that efficiently reduces ~22 billion possible pairs to ≤50 candidate pairs per Source 1 entity, and (2) an XGBoost classifier trained on 16 hand-crafted pairwise similarity features. The blocking combines an inverted token index, country-aware name prefix grouping, and TF-IDF character-ngram cosine similarity — ensuring high blocking recall while keeping the candidate set small. The classifier threshold is tuned to maximize macro F_0.5, aligning with the precision-heavy evaluation metric.

---

## 2. Methodology

### 2.1 Problem Analysis

Key observations from EDA:
- **Scale**: Source 1 has 2.2M records; Sources 2 and 3 have ~5M each. Naive all-pairs comparison (22 billion) is impossible.
- **Name noise**: Legal suffix inconsistency (`Pvt Ltd` vs `Private Limited`), DBA names, typos, transliterations (Devanagari/Kannada characters in S2/S3).
- **Address noise**: Street abbreviations (`St` vs `Street`), reordered components, missing components, landmark references (`Near SBI ATM`), PIN codes absent in one source.
- **Country distribution**: Training data contains US + India; test additionally includes France — the pipeline must handle any country string.
- **Singletons**: Many S1 entities have no match; correctly predicting empty is worth F_0.5 = 1.0 and must not be ignored.

### 2.2 Solution Strategy

**Approach Type:** Blocking → Pairwise Classifier  
**Core Innovation:** Three complementary blocking strategies (inverted token index + prefix blocking + TF-IDF cosine) that complement each other's weaknesses, combined with threshold tuning directly on F_0.5.

---

## 3. Candidate Generation (Blocking)

### 3.1 Text Normalization (Pre-blocking)

Before blocking, all text is normalized:
1. **Unicode → ASCII**: Transliterates Devanagari, Kannada, and other non-ASCII scripts.
2. **Legal suffix normalization**: `Private Limited`, `Pvt. Ltd.`, `Incorporated` → canonical forms.
3. **Address abbreviation**: `Street` → `st`, `Road` → `rd`, `North` → `n`, etc.
4. **Noise removal**: URLs, email addresses, `dba` keywords, punctuation.

### 3.2 Blocking Strategies

**Strategy 1: Inverted Token Index**
- Build a dictionary `token → [S2/S3 entity indices]` on normalized business names.
- Prune tokens that appear in >500 records (they are stop-words equivalent).
- For each S1 entity, look up all S2/S3 entities sharing ≥1 rare token.
- *Strength*: Handles word-order transpositions, typos in non-key words.

**Strategy 2: Country + Name Prefix Blocking**
- Group S2/S3 records by `(country, first_4_chars_of_normalized_name)`.
- For each S1 entity, retrieve all records in the matching group.
- *Strength*: Zero false cross-country matches; very fast.

**Strategy 3: TF-IDF Character N-gram Cosine Similarity**
- Fit TF-IDF (character 2-3grams, 100K features, sublinear TF) on all name+address texts.
- For each S1 entity, find the top-30 most similar S2/S3 records via batched sparse matrix multiply.
- *Strength*: Handles transliterations, abbreviation-heavy names, combined name+address matching.

### 3.3 Merging & Capping
All three candidate sets are unioned. To control the candidate set size (which directly impacts precision), candidates are capped at **50 per S1 entity**.

### 3.4 Candidate Pair Statistics
*(to be filled after full run)*
- Total candidate pairs generated: [X]
- S1 entities with ≥1 candidate: [X]
- Average candidates per S1 entity: [X]
- Blocking recall (train validation split): [X]

### 3.5 Ensuring True Matches Are Not Lost
- **Token index** catches entities sharing any meaningful name word, even across noisy variations.
- **TF-IDF** catches character-level near-matches and name+address combined similarity.
- **Prefix blocking** provides a fast, exact-ish safety net for closely named entities in the same country.

---

## 4. Matching Model

### 4.1 Features Used

**Name features (8):**
| Feature | Description |
|---------|-------------|
| `name_token_set_ratio` | RapidFuzz token set ratio — ignores word order |
| `name_token_sort_ratio` | RapidFuzz token sort ratio |
| `name_partial_ratio` | Best substring alignment score |
| `name_ratio` | Plain edit-distance-based ratio |
| `name_wratio` | Weighted ratio (best of multiple methods) |
| `name_bigram_jaccard` | Jaccard similarity on character bigrams |
| `name_token_jaccard` | Jaccard similarity on word tokens |
| `name_length_ratio` | min(len)/max(len) — penalizes very different lengths |

**Address features (6):**
| Feature | Description |
|---------|-------------|
| `addr_token_set_ratio` | Token set ratio on normalized address |
| `addr_token_sort_ratio` | Token sort ratio on normalized address |
| `addr_partial_ratio` | Partial ratio on normalized address |
| `addr_ratio` | Edit-distance ratio on normalized address |
| `addr_token_jaccard` | Jaccard on address word tokens |
| `addr_numeric_overlap` | Jaccard on numeric tokens (street numbers, PIN codes) |

**Meta features (2):**
| Feature | Description |
|---------|-------------|
| `country_match` | 1 if countries match, else 0 |
| `combined_token_set_ratio` | Token set ratio on name+address concatenated |

### 4.2 Model Architecture

**XGBoost Gradient Boosted Trees**
- 300 estimators, max_depth=6, learning_rate=0.1
- Subsampling: 80% rows and columns per tree
- Class imbalance: `scale_pos_weight = neg/pos`

### 4.3 Threshold Selection

A sweep over thresholds [0.10, 0.15, ..., 0.90] is performed on the held-out 20% validation split. The threshold maximizing macro F_0.5 is selected. This directly aligns training with evaluation.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation):** [to be filled after full run]

**Common false positives (wrong merges):**
- Different businesses with identical or very similar names in the same city (e.g., multiple `Ram Market` in Delhi).
- Abbreviation collisions where multiple expansions are plausible.

**Common false negatives (missed matches):**
- Entities where the S2/S3 name is completely different (DBA/trade name), failing to be blocked.
- Records where both name and address are missing/empty.

---

## 6. Conclusion

Our pipeline combines three complementary blocking strategies to achieve high recall at a small average candidate set size per S1 entity. An XGBoost classifier on 16 pairwise features captures both name and address similarity, with threshold tuning directly on F_0.5 to balance precision and recall appropriately. The pipeline is country-agnostic by design, handling France and any future countries without code changes.

---

## Appendix

### A. Code Artifacts

All code is in `code/business_entity_resolution/src/entity_resolution.py`.

Entry points:
```bash
# Validation mode (train on 80%, evaluate on 20%)
python code/business_entity_resolution/src/entity_resolution.py --mode train

# Production mode (train on all data, predict test set)
python code/business_entity_resolution/src/entity_resolution.py --mode test
```

Outputs are written to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results

*(Insert charts, ablation results here if available)*

---

**Note:** The pipeline is designed for scalability — all blocking operations use sparse matrix algebra and dictionary lookups; no O(n²) comparisons are performed.
