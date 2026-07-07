# scripts/preprocess.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Step 1 — Load candidates, compute all features, save to disk
#
# Run this ONCE before ranking. It handles the slow work:
#   - Reading all candidates from the JSON/JSONL file
#   - Embedding profiles with sentence-transformers
#   - Computing all 7 feature scores per candidate against the JD
#   - Saving results as a Parquet file
# ─────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time

import numpy as np

# Add the project root to sys.path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_candidates
from src.normalization import normalize_job_description, read_jd_file
from src.feature_engineering import (
    load_embedding_model,
    embed_job_description,
    compute_candidate_embeddings,
    compute_all_features,
    JD_TEXT,
)
from src.ranker import compute_final_score


def main():
    # ── Parse arguments ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Preprocess candidates for ranking.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample_candidates.json (50 candidates) instead of the full file.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to candidates.json/jsonl. Ignored if --sample is set.",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to a custom Job Description file (.txt, .pdf, .docx).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Defaults to outputs/ in the project root.",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Output directory
    out_dir = args.out_dir or os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load Job Description ──────────────────────────────────
    jd_raw_text = JD_TEXT
    if args.jd:
        custom_text = read_jd_file(args.jd)
        if custom_text and custom_text.strip():
            jd_raw_text = custom_text
            print(f"[OK] Loaded custom Job Description from: {args.jd}")
        else:
            print(f"WARNING: Could not parse custom JD from {args.jd}. Using default JD.")

    jd_schema = normalize_job_description(jd_raw_text)

    # ── Step 2: Load candidates ───────────────────────────────────────
    t0 = time.time()

    if args.sample:
        sample_path = os.path.abspath(os.path.join(project_root, "..", "Testing Data", "sample_candidates.json"))
        if not os.path.exists(sample_path):
            # Try within workspace
            sample_path = os.path.abspath(os.path.join(project_root, "Testing Data", "sample_candidates.json"))
        if not os.path.exists(sample_path):
            print(f"ERROR: Sample file not found at: {sample_path}")
            sys.exit(1)
        candidates = load_candidates(sample_path)
    else:
        candidates_path = args.candidates or os.path.join(project_root, "data", "candidates.jsonl")
        if not os.path.exists(candidates_path):
            print(f"ERROR: Candidates file not found at: {candidates_path}")
            sys.exit(1)
        candidates = load_candidates(candidates_path)

    print(f"\n[OK] Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    if len(candidates) == 0:
        print("ERROR: No candidates loaded.")
        sys.exit(1)

    # ── Step 3: Load embedding model ──────────────────────────────────
    print("\n-- Loading embedding model --")
    t1 = time.time()
    model = load_embedding_model()
    print(f"[OK] Model loaded in {time.time() - t1:.1f}s")

    # ── Step 4: Embed the Job Description ─────────────────────────────
    print("\n-- Embedding Job Description --")
    jd_embedding = embed_job_description(model, jd_schema.raw_text)

    # ── Step 5: Embed all candidate profiles ──────────────────────────
    print("\n-- Embedding candidate profiles --")
    t2 = time.time()

    mmap_path = None
    if not args.sample:
        mmap_path = os.path.join(out_dir, "embeddings.mmap")
        print(f"  Using disk-based embedding (memmap): {mmap_path}")

    candidate_embeddings = compute_candidate_embeddings(
        candidates,
        model,
        batch_size=32,
        chunk_size=5000,
        mmap_path=mmap_path,
    )
    print(f"[OK] Embeddings computed in {time.time() - t2:.1f}s")

    embeddings_path = os.path.join(out_dir, "embeddings.npy")
    np.save(embeddings_path, candidate_embeddings)
    print(f"[OK] Embeddings saved to: {embeddings_path}")

    # ── Step 6: Compute all feature scores ────────────────────────────
    print("\n-- Computing feature scores --")
    t3 = time.time()
    features_df = compute_all_features(candidates, candidate_embeddings, jd_embedding, jd_schema)
    print(f"[OK] Features computed in {time.time() - t3:.1f}s")

    # ── Step 7: Compute final ranking score ───────────────────────────
    print("\n-- Computing final ranking scores --")
    ranked_df = compute_final_score(features_df, jd_schema)

    # ── Step 8: Save to Parquet ───────────────────────────────────────
    features_path = os.path.join(out_dir, "features.parquet")
    ranked_df.to_parquet(features_path, index=False)
    print(f"[OK] Features + scores saved to: {features_path}")

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n" + "="*60)
    print(f"PREPROCESSING COMPLETE")
    print("="*60)
    print(f"  Candidates processed : {len(candidates)}")
    print(f"  Honeypots detected   : {ranked_df['is_honeypot'].sum()}")
    print(f"  Total time           : {total_time:.1f}s")
    print(f"  Output               : {features_path}")
    print(f"\nTop 10 candidates by score:")
    print("-"*60)

    top10 = ranked_df.head(10)
    for _, row in top10.iterrows():
        print(
            f"  Rank {int(row['rank']):>3} | "
            f"Score: {row['final_score']:.4f} | "
            f"{row['current_title'][:30]:<30} | "
            f"{str(row.get('city', ''))[:20]}"
        )


if __name__ == "__main__":
    main()
