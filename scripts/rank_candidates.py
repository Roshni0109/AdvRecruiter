# scripts/rank_candidates.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Step 2 — Load precomputed features, generate final CSV submission
#
# This is the FAST step that must run in under 5 minutes on the judging server.
# All the slow embedding work was already done by preprocess.py.
#
# What this script does:
#   1. Loads the precomputed features.parquet (output of preprocess.py)
#   2. Takes the top 100 candidates by final_score
#   3. Generates a human-readable reasoning string for each
#   4. Saves the submission CSV in the required format
#
# Usage:
#   python scripts/rank_candidates.py
#   python scripts/rank_candidates.py --out ./outputs/team_001.csv
# ─────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import time

import pandas as pd

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_reasoning(row: pd.Series) -> str:
    """
    Build a short, human-readable explanation for why this candidate ranked here.

    The reasoning is shown to human reviewers in Stage 4 of the hackathon.
    It should be specific, not generic — real numbers and signals matter.
    """
    parts = []

    # --- Title and experience ---
    title = row.get("current_title", "Unknown title")
    years = row.get("years_exp", 0)
    parts.append(f"{title} with {years:.0f}y exp")

    # --- Semantic fit ---
    sem = row.get("semantic_similarity", 0)
    if sem >= 0.75:
        parts.append("strong semantic match to JD")
    elif sem >= 0.55:
        parts.append("good semantic alignment")
    else:
        parts.append("partial JD alignment")

    # --- JD skill overlap ---
    overlap = row.get("jd_skill_overlap", 0)
    if overlap >= 5:
        parts.append(f"{overlap} JD skills matched")
    elif overlap >= 2:
        parts.append(f"{overlap} JD skills")

    # --- Career quality signals ---
    career_q = row.get("career_quality", 0)
    if career_q >= 0.75:
        parts.append("product-company background")
    elif career_q < 0.4:
        parts.append("mostly services background")

    # --- Behavioral engagement ---
    response_rate = row.get("response_rate", 0)
    last_active = row.get("last_active_days", 9999)
    if last_active <= 14 and response_rate >= 0.6:
        parts.append("active & responsive")
    elif last_active > 90:
        parts.append("inactive 90+ days")
    elif response_rate < 0.3:
        parts.append("low response rate")

    # --- Notice period ---
    notice = row.get("notice_days", 90)
    if notice <= 30:
        parts.append(f"{notice}d notice")
    elif notice > 90:
        parts.append(f"long notice ({notice}d)")

    # --- GitHub activity ---
    github = row.get("github_score", -1)
    if github > 60:
        parts.append(f"strong GitHub (score {github:.0f})")
    elif github == -1:
        parts.append("no GitHub linked")

    # --- Location ---
    location = str(row.get("city", ""))
    if location:
        parts.append(location)

    return "; ".join(parts)


def main():
    # ── Parse arguments ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Generate submission CSV from precomputed features.")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to features.parquet (output of preprocess.py).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Default: outputs/submission.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of candidates to include in submission (default: 100).",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    features_path = args.features or os.path.join(out_dir, "features.parquet")
    out_path = args.out or os.path.join(out_dir, "submission.csv")

    # ── Step 1: Load precomputed features ─────────────────────────────
    if not os.path.exists(features_path):
        print(f"ERROR: features.parquet not found at: {features_path}")
        print("Run scripts/preprocess.py first.")
        sys.exit(1)

    print(f"Loading precomputed features from: {features_path}")
    t0 = time.time()
    df = pd.read_parquet(features_path)
    print(f"[OK] Loaded {len(df)} candidates in {time.time() - t0:.2f}s")

    # ── Step 2: Verify it's already sorted by score ────────────────────
    # preprocess.py already sorts + assigns rank, but let's be safe
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    # ── Step 3: Take top N (default 100) ──────────────────────────────
    top_n = min(args.top_n, len(df))
    if top_n < 100:
        print(f"WARNING: Only {top_n} candidates available (need 100 for submission).")
        print("This is fine for testing with the 50-candidate sample file.")

    top_df = df.head(top_n).copy()
    top_df["rank"] = range(1, top_n + 1)

    # ── Step 4: Generate reasoning strings ────────────────────────────
    print("Generating reasoning strings...")
    top_df["reasoning"] = top_df.apply(generate_reasoning, axis=1)

    # ── Step 5: Validate output ────────────────────────────────────────
    # Check score is non-increasing (required by submission spec)
    scores = top_df["final_score"].tolist()
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1] - 1e-9:
            print(f"WARNING: Score not non-increasing at rank {i+1} → {i+2}")

    # Check no duplicate candidate IDs
    n_unique = top_df["candidate_id"].nunique()
    if n_unique != len(top_df):
        print(f"WARNING: Duplicate candidate_ids detected! ({n_unique} unique / {len(top_df)} rows)")

    # ── Step 6: Build and save submission CSV ─────────────────────────
    submission = top_df[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission = submission.rename(columns={"final_score": "score"})

    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Submission saved to: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUBMISSION GENERATED")
    print("="*70)
    print(f"  Rows          : {len(submission)}")
    print(f"  Top score     : {submission['score'].iloc[0]:.4f}  (rank 1)")
    print(f"  Bottom score  : {submission['score'].iloc[-1]:.4f}  (rank {len(submission)})")
    print(f"  Score range   : {submission['score'].max():.4f} -> {submission['score'].min():.4f}")
    print("\nTop 10 submissions:")
    print("-"*70)
    for _, row in submission.head(10).iterrows():
        print(
            f"  Rank {int(row['rank']):>3} | "
            f"Score: {row['score']:.4f} | "
            f"{row['candidate_id']} | "
            f"{row['reasoning'][:55]}..."
        )

    print("\n" + "-"*70)
    print(f"Output file: {out_path}")
    print("\nValidate with:")
    validate_path = os.path.abspath(
        os.path.join(project_root, "..", "Testing Data", "validate_submission.py")
    )
    print(f'  python "{validate_path}" "{out_path}"')


if __name__ == "__main__":
    main()
