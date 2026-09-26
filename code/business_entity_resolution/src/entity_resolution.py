#!/usr/bin/env python3
"""
Business Entity Resolution Pipeline
=====================================
A high-performance, multi-stage pipeline for resolving business entities
across three independent data sources at scale.

Pipeline Stages:
  1. Text Normalization  — clean, standardize names/addresses
  2. Blocking            — generate small candidate sets per S1 entity
     - Country-aware TF-IDF vector blocking (cosine similarity)
     - Token-based inverted index (shared rare tokens)
     - Prefix blocking on normalized business name
  3. Feature Engineering — pairwise similarity features
  4. Gradient-Boosted Classifier — trained on labeled pairs
  5. Threshold Tuning    — optimize F_0.5 on validation split

Author: Entity Resolution Team
"""

import re
import os
import sys
import time
import logging
import unicodedata
import collections
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# PATHS
BASE_DIR = Path(__file__).resolve().parents[3]  # student_resource/
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR  = BASE_DIR / "dataset" / "test"
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 1. TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────────
LEGAL_SUFFIXES = [
    (r"\bprivate limited\b", "pvt ltd"),
    (r"\bpvt\. limited\b", "pvt ltd"),
    (r"\bprivate ltd\b", "pvt ltd"),
    (r"\bpvt\.? ltd\.?\b", "pvt ltd"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\blimited liability company\b", "llc"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcompany\b", "co"),
    (r"\benterprises?\b", "ent"),
    (r"\bindustries\b", "ind"),
    (r"\bservices?\b", "svc"),
    (r"\bsolutions?\b", "soln"),
    (r"\bmanagement\b", "mgmt"),
    (r"\binternational\b", "intl"),
    (r"\bnational\b", "natl"),
    (r"\btrading\b", "trd"),
    (r"\bventures?\b", "vent"),
    (r"\bassociates?\b", "assoc"),
    (r"\bbrothers?\b", "bros"),
]

ADDRESS_ABBREV = [
    (r"\bstreet\b", "st"),
    (r"\broad\b", "rd"),
    (r"\bavenue\b", "ave"),
    (r"\bboulevard\b", "blvd"),
    (r"\bdrive\b", "dr"),
    (r"\bcourt\b", "ct"),
    (r"\blane\b", "ln"),
    (r"\bplace\b", "pl"),
    (r"\bsuite\b", "ste"),
    (r"\bapartment\b", "apt"),
    (r"\bnorth\b", "n"),
    (r"\bsouth\b", "s"),
    (r"\beast\b", "e"),
    (r"\bwest\b", "w"),
]


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        normalized = unicodedata.normalize("NFKD", text)
        return normalized.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return text


def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    text = normalize_unicode(name).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern, replacement in LEGAL_SUFFIXES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"^[-\s]+", "", text)
    text = re.sub(r"\bdba\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(addr: str) -> str:
    if not isinstance(addr, str) or not addr.strip():
        return ""
    text = normalize_unicode(addr).lower().strip()
    text = re.sub(r"[^\w\s,]", " ", text)
    for pattern, replacement in ADDRESS_ABBREV:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tokens(text: str, min_len: int = 3) -> Set[str]:
    if not text:
        return set()
    tokens = set(text.split())
    return {t for t in tokens if len(t) >= min_len and not t.isdigit()}


def name_prefix(normalized_name: str, length: int = 4) -> str:
    tokens = normalized_name.split()
    if not tokens:
        return ""
    return tokens[0][:length]


def extract_name_bigrams(name: str) -> Set[str]:
    if len(name) < 2:
        return set()
    return {name[i:i+2] for i in range(len(name) - 1)}


# ─────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_source(path: Path) -> pd.DataFrame:
    log.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    df["country"]   = df["country"].str.strip().str.lower()
    df["name_text"] = df["norm_name"] + " " + df["norm_addr"]
    log.info(f"  Loaded {len(df):,} records from {path.name}")
    return df


# ─────────────────────────────────────────────────────────────────
# 3. BLOCKING
# ─────────────────────────────────────────────────────────────────

class BlockingPipeline:
    MAX_POSTING_LIST_SIZE = 500
    PREFIX_LEN = 4
    TFIDF_TOP_K = 30
    MAX_CANDIDATES_PER_S1 = 50

    def __init__(self, s1: pd.DataFrame, s23: pd.DataFrame):
        self.s1  = s1.reset_index(drop=True)
        self.s23 = s23.reset_index(drop=True)

    def _build_token_index(self) -> Dict[str, List[int]]:
        log.info("Building inverted token index on S2/S3 ...")
        idx: Dict[str, List[int]] = collections.defaultdict(list)
        for i, row in enumerate(self.s23["norm_name"]):
            for token in extract_tokens(row):
                idx[token].append(i)
        pruned = {
            tok: ids
            for tok, ids in idx.items()
            if len(ids) <= self.MAX_POSTING_LIST_SIZE
        }
        log.info(f"  Token index: {len(pruned):,} tokens after pruning")
        return pruned

    def _token_blocking(self, token_idx: Dict[str, List[int]]) -> Dict[str, Set[str]]:
        log.info("Running token blocking ...")
        s23_ids = self.s23["entity_id"].values
        candidates: Dict[str, Set[str]] = {eid: set() for eid in self.s1["entity_id"]}
        s1_ids   = self.s1["entity_id"].values
        s1_names = self.s1["norm_name"].values

        for i, (s1_id, name) in enumerate(
            tqdm(zip(s1_ids, s1_names), total=len(s1_ids), desc="Token blocking")
        ):
            for token in extract_tokens(name):
                if token in token_idx:
                    for j in token_idx[token]:
                        candidates[s1_id].add(s23_ids[j])
                        if len(candidates[s1_id]) >= self.MAX_CANDIDATES_PER_S1 * 3:
                            break
        return candidates

    def _prefix_blocking(self) -> Dict[str, Set[str]]:
        log.info("Running prefix blocking ...")
        s23_groups: Dict[Tuple, List[str]] = collections.defaultdict(list)
        for _, row in self.s23.iterrows():
            key = (row["country"], name_prefix(row["norm_name"], self.PREFIX_LEN))
            if key[1]:
                s23_groups[key].append(row["entity_id"])

        candidates: Dict[str, Set[str]] = {eid: set() for eid in self.s1["entity_id"]}
        for _, row in tqdm(self.s1.iterrows(), total=len(self.s1), desc="Prefix blocking"):
            key = (row["country"], name_prefix(row["norm_name"], self.PREFIX_LEN))
            if key[1] and key in s23_groups:
                for eid in s23_groups[key][:self.MAX_CANDIDATES_PER_S1]:
                    candidates[row["entity_id"]].add(eid)
        return candidates

    def _tfidf_blocking(self) -> Dict[str, Set[str]]:
        log.info("Running TF-IDF blocking ...")
        all_texts = list(self.s1["name_text"].values) + list(self.s23["name_text"].values)
        n_s1 = len(self.s1)

        log.info("  Fitting TF-IDF vectorizer ...")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 3),
            max_features=100_000,
            sublinear_tf=True,
            min_df=2,
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        s1_matrix  = tfidf_matrix[:n_s1]
        s23_matrix = tfidf_matrix[n_s1:]
        log.info(f"  TF-IDF matrix shape: S1={s1_matrix.shape}, S23={s23_matrix.shape}")

        s1_ids  = self.s1["entity_id"].values
        s23_ids = self.s23["entity_id"].values
        candidates: Dict[str, Set[str]] = {eid: set() for eid in s1_ids}
        s23_T = s23_matrix.T
        BATCH = 1000

        for start in tqdm(range(0, n_s1, BATCH), desc="TF-IDF blocking"):
            end   = min(start + BATCH, n_s1)
            batch = s1_matrix[start:end]
            sims  = (batch @ s23_T).toarray()
            top_k = min(self.TFIDF_TOP_K, sims.shape[1])
            top_indices = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
            for bi, gi in enumerate(range(start, end)):
                s1_id = s1_ids[gi]
                for j in top_indices[bi]:
                    if sims[bi, j] > 0.05:
                        candidates[s1_id].add(s23_ids[j])
        return candidates

    def generate_candidates(self) -> Dict[str, Set[str]]:
        log.info("=== BLOCKING PHASE ===")
        token_idx    = self._build_token_index()
        cands_token  = self._token_blocking(token_idx)
        cands_prefix = self._prefix_blocking()
        cands_tfidf  = self._tfidf_blocking()

        log.info("Merging candidates from all blocking strategies ...")
        all_candidates: Dict[str, Set[str]] = {}
        for s1_id in self.s1["entity_id"]:
            merged = (
                cands_token.get(s1_id, set())
                | cands_prefix.get(s1_id, set())
                | cands_tfidf.get(s1_id, set())
            )
            if len(merged) > self.MAX_CANDIDATES_PER_S1:
                merged = set(list(merged)[:self.MAX_CANDIDATES_PER_S1])
            all_candidates[s1_id] = merged

        total_cands = sum(len(v) for v in all_candidates.values())
        non_empty   = sum(1 for v in all_candidates.values() if v)
        log.info(
            f"Blocking complete: {total_cands:,} total candidates, "
            f"{non_empty:,} S1 entities with candidates, "
            f"avg {total_cands/max(len(all_candidates),1):.2f} per entity"
        )
        return all_candidates


# ─────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

def pairwise_features(s1_row: pd.Series, s23_row: pd.Series) -> np.ndarray:
    n1, n2 = s1_row["norm_name"], s23_row["norm_name"]
    a1, a2 = s1_row["norm_addr"], s23_row["norm_addr"]
    feats = []

    # Name features
    feats.append(fuzz.token_set_ratio(n1, n2) / 100.0)
    feats.append(fuzz.token_sort_ratio(n1, n2) / 100.0)
    feats.append(fuzz.partial_ratio(n1, n2) / 100.0)
    feats.append(fuzz.ratio(n1, n2) / 100.0)
    feats.append(fuzz.WRatio(n1, n2) / 100.0)

    b1, b2 = extract_name_bigrams(n1), extract_name_bigrams(n2)
    inter, union = len(b1 & b2), len(b1 | b2)
    feats.append(inter / union if union > 0 else 0.0)

    t1, t2   = set(n1.split()), set(n2.split())
    inter_t  = len(t1 & t2)
    union_t  = len(t1 | t2)
    feats.append(inter_t / union_t if union_t > 0 else 0.0)

    l1, l2 = len(n1), len(n2)
    feats.append(min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0)

    # Address features
    feats.append(fuzz.token_set_ratio(a1, a2) / 100.0)
    feats.append(fuzz.token_sort_ratio(a1, a2) / 100.0)
    feats.append(fuzz.partial_ratio(a1, a2) / 100.0)
    feats.append(fuzz.ratio(a1, a2) / 100.0)

    at1, at2 = set(a1.split()), set(a2.split())
    inter_a  = len(at1 & at2)
    union_a  = len(at1 | at2)
    feats.append(inter_a / union_a if union_a > 0 else 0.0)

    nums1 = {t for t in at1 if any(c.isdigit() for c in t)}
    nums2 = {t for t in at2 if any(c.isdigit() for c in t)}
    if nums1 or nums2:
        feats.append(len(nums1 & nums2) / len(nums1 | nums2))
    else:
        feats.append(1.0)

    feats.append(1.0 if s1_row["country"] == s23_row["country"] else 0.0)

    comb1 = n1 + " " + a1
    comb2 = n2 + " " + a2
    feats.append(fuzz.token_set_ratio(comb1, comb2) / 100.0)

    return np.array(feats, dtype=np.float32)


def build_feature_matrix(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    candidate_pairs: Dict[str, Set[str]],
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    log.info("Building feature matrix ...")
    s1_idx  = s1.set_index("entity_id")
    s23_idx = s23.set_index("entity_id")
    X_rows, y_rows, pair_ids = [], [], []

    s1_ids_with_cands = [s1_id for s1_id, cands in candidate_pairs.items() if cands]
    for s1_id in tqdm(s1_ids_with_cands, desc="Feature extraction"):
        if s1_id not in s1_idx.index:
            continue
        s1_row = s1_idx.loc[s1_id]
        for s23_id in candidate_pairs[s1_id]:
            if s23_id not in s23_idx.index:
                continue
            s23_row = s23_idx.loc[s23_id]
            feats = pairwise_features(s1_row, s23_row)
            X_rows.append(feats)
            pair_ids.append((s1_id, s23_id))
            if ground_truth is not None:
                y_rows.append(1 if s23_id in ground_truth.get(s1_id, set()) else 0)

    X = np.array(X_rows, dtype=np.float32) if X_rows else np.empty((0, 16), dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)   if y_rows else np.array([], dtype=np.int32)
    pos_msg = f"Positives: {y.sum()}/{len(y)}" if len(y) > 0 else "Inference mode"
    log.info(f"Feature matrix: {X.shape[0]:,} pairs. {pos_msg}")
    return X, y, pair_ids


# ─────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING & INFERENCE
# ─────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    log.info("Training XGBoost classifier ...")
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / max(pos, 1)
    log.info(f"  Class balance: {pos} positives, {neg} negatives")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    log.info("Training complete.")
    return model


def compute_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Optional[Set[str]] = None,
) -> float:
    if all_s1_ids is None:
        all_s1_ids = set(ground_truth.keys())
    beta_sq = 0.25
    scores = []
    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = set(predictions.get(s1_id, []))
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set and pred_set:
            scores.append(0.0)
        elif true_set and not pred_set:
            scores.append(0.0)
        else:
            tp   = len(true_set & pred_set)
            prec = tp / len(pred_set)
            rec  = tp / len(true_set)
            if prec + rec == 0:
                scores.append(0.0)
            else:
                scores.append((1 + beta_sq) * prec * rec / (beta_sq * prec + rec))
    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(
    model: xgb.XGBClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pair_ids_val: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    all_val_ids: Set[str],
) -> float:
    log.info("Tuning classification threshold ...")
    scores = model.predict_proba(X_val)[:, 1]
    best_f05, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds: Dict[str, List[str]] = collections.defaultdict(list)
        for (s1_id, s23_id), score in zip(pair_ids_val, scores):
            if score >= thresh:
                preds[s1_id].append(s23_id)
        f05 = compute_macro_f05(preds, ground_truth, all_val_ids)
        if f05 > best_f05:
            best_f05, best_thresh = f05, thresh
    log.info(f"Best threshold: {best_thresh:.2f} -> F_0.5 = {best_f05:.4f}")
    return best_thresh


def predict_matches(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    threshold: float,
    all_s1_ids: Set[str],
) -> Dict[str, List[str]]:
    log.info(f"Generating predictions (threshold={threshold:.2f}) ...")
    preds: Dict[str, List[str]] = collections.defaultdict(list)
    if len(X) > 0:
        scores = model.predict_proba(X)[:, 1]
        for (s1_id, s23_id), score in zip(pair_ids, scores):
            if score >= threshold:
                preds[s1_id].append(s23_id)
    result = {}
    for s1_id in all_s1_ids:
        result[s1_id] = list(set(preds.get(s1_id, [])))
    return result


# ─────────────────────────────────────────────────────────────────
# 6. OUTPUT WRITING
# ─────────────────────────────────────────────────────────────────

def write_matching_results(predictions: Dict[str, List[str]], output_path: Path) -> None:
    rows = []
    for s1_id, matched_ids in sorted(predictions.items()):
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(sorted(set(matched_ids))),
        })
    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    log.info(f"Wrote {len(df):,} rows to {output_path}")


def write_candidate_pairs(
    candidates: Dict[str, Set[str]],
    all_s1_ids: Set[str],
    output_path: Path,
) -> None:
    rows = []
    for s1_id in sorted(all_s1_ids):
        cand_ids = candidates.get(s1_id, set())
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cand_ids)),
        })
    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    log.info(f"Wrote {len(df):,} rows to {output_path}")


# ─────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_pipeline(mode: str = "test") -> None:
    t_start = time.time()
    log.info(f"=== Business Entity Resolution Pipeline (mode={mode}) ===")

    if mode == "train":
        s1_path = TRAIN_DIR / "train_source1.tsv"
        s2_path = TRAIN_DIR / "train_source2.tsv"
        s3_path = TRAIN_DIR / "train_source3.tsv"
        gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    else:
        s1_path = TEST_DIR / "test_source1.tsv"
        s2_path = TEST_DIR / "test_source2.tsv"
        s3_path = TEST_DIR / "test_source3.tsv"
        gt_path = None

    # Load test data
    s1  = load_source(s1_path)
    s2  = load_source(s2_path)
    s3  = load_source(s3_path)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    all_s1_ids = set(s1["entity_id"].values)

    if mode == "train":
        # Load ground truth
        log.info("Loading ground truth ...")
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        ground_truth: Dict[str, Set[str]] = {}
        for _, row in gt_df.iterrows():
            s1_id   = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            ground_truth[s1_id] = set(matched.split(",")) if matched.strip() else set()

        # Train/val split
        all_keys = list(ground_truth.keys())
        np.random.seed(42)
        np.random.shuffle(all_keys)
        split   = int(0.8 * len(all_keys))
        val_ids = set(all_keys[split:])
        s1_train = s1[~s1["entity_id"].isin(val_ids)].reset_index(drop=True)
        s1_val   = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
        gt_train = {k: v for k, v in ground_truth.items() if k not in val_ids}
        gt_val   = {k: v for k, v in ground_truth.items() if k in val_ids}
        log.info(f"Train: {len(gt_train):,} entities, Val: {len(gt_val):,} entities")

        # Train blocking
        blocker_tr = BlockingPipeline(s1_train, s23)
        cands_tr   = blocker_tr.generate_candidates()
        X_tr, y_tr, _ = build_feature_matrix(s1_train, s23, cands_tr, gt_train)
        model = train_model(X_tr, y_tr)

        # Val blocking
        blocker_val = BlockingPipeline(s1_val, s23)
        cands_val   = blocker_val.generate_candidates()
        X_val, y_val, pairs_val = build_feature_matrix(s1_val, s23, cands_val, gt_val)

        threshold = tune_threshold(
            model, X_val, y_val, pairs_val, gt_val, set(s1_val["entity_id"])
        )

        val_preds = predict_matches(
            model, X_val, pairs_val, threshold, set(s1_val["entity_id"])
        )
        val_f05 = compute_macro_f05(val_preds, gt_val, set(s1_val["entity_id"]))
        log.info(f"Validation F_0.5 (macro): {val_f05:.4f}")

    else:
        # Test mode: train on all train data first, then predict test
        log.info("Loading training data to fit model ...")
        s1_tr = load_source(TRAIN_DIR / "train_source1.tsv")
        s2_tr = load_source(TRAIN_DIR / "train_source2.tsv")
        s3_tr = load_source(TRAIN_DIR / "train_source3.tsv")
        s23_tr = pd.concat([s2_tr, s3_tr], ignore_index=True)
        del s2_tr, s3_tr

        gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
        ground_truth_tr: Dict[str, Set[str]] = {}
        for _, row in gt_df.iterrows():
            s1_id   = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            ground_truth_tr[s1_id] = set(matched.split(",")) if matched.strip() else set()

        # Train blocking + features
        blocker_tr = BlockingPipeline(s1_tr, s23_tr)
        cands_tr   = blocker_tr.generate_candidates()
        X_tr, y_tr, pairs_tr = build_feature_matrix(s1_tr, s23_tr, cands_tr, ground_truth_tr)

        # Quick val split for threshold tuning (20% of train)
        all_keys = list(ground_truth_tr.keys())
        np.random.seed(42)
        np.random.shuffle(all_keys)
        split   = int(0.8 * len(all_keys))
        val_ids = set(all_keys[split:])
        pairs_val_filter = [(i, p) for i, p in enumerate(pairs_tr) if p[0] in val_ids]
        if pairs_val_filter:
            val_indices, pairs_val = zip(*pairs_val_filter)
            X_val_small = X_tr[list(val_indices)]
            gt_val_small = {k: v for k, v in ground_truth_tr.items() if k in val_ids}
            model = train_model(X_tr, y_tr)
            all_val_ids_small = set(p[0] for p in pairs_val) | val_ids
            # Compute scores only for threshold tuning (no y_val needed for F05)
            threshold = tune_threshold(
                model,
                X_val_small,
                np.array([1 if p[1] in ground_truth_tr.get(p[0], set()) else 0 for p in pairs_val]),
                list(pairs_val),
                gt_val_small,
                all_val_ids_small,
            )
        else:
            model = train_model(X_tr, y_tr)
            threshold = 0.5
        del s1_tr, s23_tr, X_tr, y_tr

        # Generate candidates for test data
        log.info("Generating candidates for test set ...")
        blocker_test = BlockingPipeline(s1, s23)
        test_candidates = blocker_test.generate_candidates()

        # Features (inference mode)
        X_test, _, pairs_test = build_feature_matrix(s1, s23, test_candidates)

        # Predict
        predictions = predict_matches(model, X_test, pairs_test, threshold, all_s1_ids)

        # Write outputs
        write_matching_results(predictions, OUT_DIR / "matching_results.tsv")
        write_candidate_pairs(test_candidates, all_s1_ids, OUT_DIR / "candidate_pairs.tsv")

    elapsed = time.time() - t_start
    log.info(f"Pipeline completed in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "test"], default="test",
        help="'train' to evaluate on held-out validation, 'test' to produce final predictions"
    )
    args = parser.parse_args()
    run_pipeline(mode=args.mode)
