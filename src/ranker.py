# src/ranker.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Score and rank candidates using the engineered features
#
# After feature_engineering.py turns candidates into numbers,
# this file takes those numbers and produces a final score (0-1)
# for each candidate — telling us "how good a fit are they?"
# ─────────────────────────────────────────────────────────────────────

import pandas as pd
from src.feature_engineering import WEIGHTS

def compute_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all 7 feature scores into one final ranking score using WEIGHTS.

    This is a weighted linear combination:
        final_score = w1*s1 + w2*s2 + ... + w7*s7

    Why weighted linear combination?
    → Simple, fast, interpretable. We know exactly why a candidate got
      a high or low score — we can point to each component.

    After computing the score, we:
    1. Apply a honeypot penalty (score → 0 for honeypots)
    2. Sort by final_score descending
    3. Assign ranks 1–N
    """
    print("Computing final scores...")

    # ── STEP 1: Compute base weighted score from all 7 features ──
    base_score = (
        WEIGHTS["semantic_similarity"] * df["semantic_similarity"] +
        WEIGHTS["title_relevance"]     * df["title_relevance"]     +
        WEIGHTS["behavioral"]          * df["behavioral"]           +
        WEIGHTS["experience_fit"]      * df["experience_fit"]       +
        WEIGHTS["career_quality"]      * df["career_quality"]       +
        WEIGHTS["skill_depth"]         * df["skill_depth"]          +
        WEIGHTS["location"]            * df["location_score"]
    )

    # ── STEP 2: Apply title gate as a MULTIPLIER ──
    title_gate = 0.35 + 0.65 * df["title_relevance"]
    df["final_score"] = base_score * title_gate

    # ── HONEYPOT PENALTY ──
    # Honeypots must never appear in the top 100.
    # We set their score to 0.0 so they sink to the bottom.
    honeypot_count = df["is_honeypot"].sum()
    if honeypot_count > 0:
        print(f"  Detected {honeypot_count} honeypot candidates — penalizing scores.")
    df.loc[df["is_honeypot"] == True, "final_score"] = 0.0

    # ── SORT AND RANK ──
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1   # rank starts at 1

    print(f"Scoring complete. Top score: {df['final_score'].iloc[0]:.4f}")

    bottom_rank = min(99, len(df) - 1)
    print(f"Bottom score (rank {bottom_rank + 1}): {df['final_score'].iloc[bottom_rank]:.4f}")
    return df

