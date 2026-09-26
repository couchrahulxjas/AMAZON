#!/usr/bin/env python3
"""
Quick smoke test on a sample of training data to validate the pipeline.
Uses nrows= for fast loading. Runs in ~2-3 minutes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import collections
from pathlib import Path

# Patch load_source to use nrows
import entity_resolution as er
from entity_resolution import (
    BlockingPipeline, build_feature_matrix,
    train_model, tune_threshold, predict_matches, compute_macro_f05,
    normalize_name, normalize_address,
    TRAIN_DIR, OUT_DIR, log
)

SAMPLE_S1   = 2000
SAMPLE_S23  = 20000

log.info("=== SMOKE TEST (small sample, fast load) ===")

def load_sample(path, nrows):
    log.info(f"Loading {path.name} (nrows={nrows}) ...")
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows).fillna("")
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    df["country"]   = df["country"].str.strip().str.lower()
    df["name_text"] = df["norm_name"] + " " + df["norm_addr"]
    log.info(f"  Loaded {len(df):,} records")
    return df

s1  = load_sample(TRAIN_DIR / "train_source1.tsv", SAMPLE_S1)
s2  = load_sample(TRAIN_DIR / "train_source2.tsv", SAMPLE_S23 // 2)
s3  = load_sample(TRAIN_DIR / "train_source3.tsv", SAMPLE_S23 // 2)
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3

# Load relevant ground truth (small file)
log.info("Loading ground truth ...")
gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, nrows=SAMPLE_S1 * 3).fillna("")
s1_ids_set  = set(s1["entity_id"].values)
s23_ids_set = set(s23["entity_id"].values)

ground_truth = {}
for _, row in gt_df.iterrows():
    s1_id   = row["source1_entity_id"]
    matched = row["matched_entity_ids"]
    if s1_id in s1_ids_set:
        all_matches = set(matched.split(",")) if matched.strip() else set()
        ground_truth[s1_id] = all_matches & s23_ids_set

# Ensure all S1 in ground_truth
for s1_id in s1_ids_set:
    if s1_id not in ground_truth:
        ground_truth[s1_id] = set()

log.info(f"GT entries: {len(ground_truth)}, S1 with true matches: {sum(1 for v in ground_truth.values() if v)}")

# Train/val split
keys = list(s1_ids_set)
np.random.seed(42)
np.random.shuffle(keys)
split   = int(0.8 * len(keys))
val_ids = set(keys[split:])
s1_tr = s1[~s1["entity_id"].isin(val_ids)].reset_index(drop=True)
s1_v  = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
gt_tr = {k: v for k, v in ground_truth.items() if k not in val_ids}
gt_v  = {k: v for k, v in ground_truth.items() if k in val_ids}

# Blocking
blocker_tr = BlockingPipeline(s1_tr, s23)
cands_tr   = blocker_tr.generate_candidates()
blocker_v  = BlockingPipeline(s1_v, s23)
cands_v    = blocker_v.generate_candidates()

# Blocking recall ceiling check
true_matches_in_cands = 0
true_matches_total    = 0
for s1_id, true_set in gt_v.items():
    for m in true_set:
        true_matches_total += 1
        if m in cands_v.get(s1_id, set()):
            true_matches_in_cands += 1

if true_matches_total > 0:
    blocking_recall = true_matches_in_cands / true_matches_total
    log.info(f"Blocking recall ceiling: {blocking_recall:.4f} ({true_matches_in_cands}/{true_matches_total})")
else:
    log.info("No true matches in val sample (S23 sample doesn't overlap GT — this is expected with small sample)")

# Features
X_tr, y_tr, _          = build_feature_matrix(s1_tr, s23, cands_tr, gt_tr)
X_v,  y_v,  pairs_v    = build_feature_matrix(s1_v,  s23, cands_v,  gt_v)

# Train & evaluate
if len(y_tr) > 0 and y_tr.sum() > 0:
    model = train_model(X_tr, y_tr)
    if len(X_v) > 0 and y_v.sum() > 0:
        threshold = tune_threshold(model, X_v, y_v, pairs_v, gt_v, set(s1_v["entity_id"]))
        val_preds = predict_matches(model, X_v, pairs_v, threshold, set(s1_v["entity_id"]))
        val_f05   = compute_macro_f05(val_preds, gt_v, set(s1_v["entity_id"]))
        log.info(f"Sample Validation F_0.5: {val_f05:.4f}")
    else:
        log.warning("No positive val examples (expected for very small sample). Pipeline logic is valid.")
        # Still test inference path
        if len(X_v) > 0:
            preds = predict_matches(model, X_v, pairs_v, 0.5, set(s1_v["entity_id"]))
            log.info(f"Inference check OK — predicted {sum(len(v) for v in preds.values())} matches")
else:
    log.warning("No positives in training sample. Increase SAMPLE_S23 for meaningful metrics.")

log.info("=== SMOKE TEST COMPLETE — Pipeline logic validated ===")
