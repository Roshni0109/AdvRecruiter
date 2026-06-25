# scripts/preprocess.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Step 1 — Load candidates, compute all features, save to disk
#
# Run this ONCE before ranking. It handles the slow work:
#   - Reading all candidates from the JSONL file
#   - Embedding 100K profiles with sentence-transformers (slow but done once)
#   - Computing all 7 feature scores per candidate
#   - Saving results as a Parquet file (fast to reload later)
#
# Usage (sample/test mode — 50 candidates from sample_candidates.json):
#   python scripts/preprocess.py --sample
#
# Usage (full run — all 100K from candidates.jsonl):
#   python scripts/preprocess.py
#
# Output:
#   outputs/features.parquet   — all candidate features + scores
#   outputs/embeddings.npy     — raw candidate embeddings (for debugging)
# ─────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time

import numpy as np

# Add the project root to sys.path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_candidates, extract_candidate_features
from src.feature_engineering import (
    load_embedding_model,
    embed_job_description,
    compute_candidate_embeddings,
    compute_all_features,
    compute_final_score,
)


def load_sample_candidates(filepath: str) -> list[dict]:
    """
    Load candidates from sample_candidates.json (JSON array format).

    The sample file is a regular JSON array [ {...}, {...}, ... ]
    unlike the full candidates.jsonl which is one JSON object per line.
    """
    print(f"Loading sample candidates from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_list = json.load(f)  # loads the whole array at once

    print(f"  Read {len(raw_list)} raw records.")

    # Flatten each raw candidate dict using the same function as the full pipeline
    candidates = [extract_candidate_features(raw) for raw in raw_list]
    print(f"  Extracted features for {len(candidates)} candidates.")
    return candidates


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
        help="Path to candidates.jsonl (full file). Ignored if --sample is set.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Defaults to outputs/ in the project root.",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────
    project_root = os.path.join(os.path.dirname(__file__), "..")
    project_root = os.path.abspath(project_root)

    # Output directory
    out_dir = args.out_dir or os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load candidates ───────────────────────────────────────
    t0 = time.time()

    if args.sample:
        # Look for the sample file relative to project root
        sample_path = os.path.join(
            project_root, "..", "Testing Data", "sample_candidates.json"
        )
        sample_path = os.path.abspath(sample_path)
        if not os.path.exists(sample_path):
            print(f"ERROR: Sample file not found at: {sample_path}")
            print("Please check the path and try again.")
            sys.exit(1)
        candidates = load_sample_candidates(sample_path)
    else:
        # Full candidates.jsonl path
        candidates_path = args.candidates or os.path.join(
            project_root, "data", "candidates.jsonl"
        )
        if not os.path.exists(candidates_path):
            print(f"ERROR: Candidates file not found at: {candidates_path}")
            print("Run with --sample to test on 50 sample candidates instead.")
            sys.exit(1)
        candidates = load_candidates(candidates_path)

    print(f"\n[OK] Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    if len(candidates) == 0:
        print("ERROR: No candidates loaded. Check file path and format.")
        sys.exit(1)

    # ── Step 2: Load embedding model ──────────────────────────────────
    print("\n-- Loading embedding model --")
    t1 = time.time()
    model = load_embedding_model()
    print(f"[OK] Model loaded in {time.time() - t1:.1f}s")

    # ── Step 3: Embed the Job Description ─────────────────────────────
    print("\n-- Embedding Job Description --")
    jd_embedding = embed_job_description(model)

    # ── Step 4: Embed all candidate profiles ──────────────────────────
    print("\n-- Embedding candidate profiles --")
    t2 = time.time()

    # For the full 100K run, write embeddings directly to disk (memmap).
    # This keeps RAM flat — only one 5K-candidate chunk lives in memory at a time.
    # For the 50-sample test, mmap_path=None so it falls back to in-memory (fast).
    mmap_path = None
    if not args.sample:
        mmap_path = os.path.join(out_dir, "embeddings.mmap")
        print(f"  Using disk-based embedding (memmap): {mmap_path}")

    candidate_embeddings = compute_candidate_embeddings(
        candidates,
        model,
        batch_size=32,       # small batch = low RAM per step
        chunk_size=5000,     # write 5K candidates to disk at a time
        mmap_path=mmap_path,
    )
    print(f"[OK] Embeddings computed in {time.time() - t2:.1f}s")

    # Save embeddings separately (useful for debugging or re-running)
    embeddings_path = os.path.join(out_dir, "embeddings.npy")
    np.save(embeddings_path, candidate_embeddings)
    print(f"[OK] Embeddings saved to: {embeddings_path}")

    # ── Step 5: Compute all feature scores ────────────────────────────
    print("\n-- Computing feature scores --")
    t3 = time.time()
    features_df = compute_all_features(candidates, candidate_embeddings, jd_embedding)
    print(f"[OK] Features computed in {time.time() - t3:.1f}s")

    # ── Step 6: Compute final ranking score ───────────────────────────
    print("\n-- Computing final ranking scores --")
    ranked_df = compute_final_score(features_df)

    # ── Step 7: Save to Parquet ───────────────────────────────────────
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

    print(f"\nNext step: run scripts/rank_candidates.py to generate submission CSV")


if __name__ == "__main__":
    main()
