# src/ranker.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Score and rank candidates using the engineered features
#
# Takes precomputed features and combines them using weights from the
# Job Description schema, applying rules and fallback filters.
# ─────────────────────────────────────────────────────────────────────

import pandas as pd
from src.feature_engineering import WEIGHTS
from src.schemas import CanonicalJobDescription

def compute_final_score(df: pd.DataFrame, jd_schema: CanonicalJobDescription = None) -> pd.DataFrame:
    """
    Combine all 5 feature scores into one final ranking score using weights.
    Weights are pulled dynamically from jd_schema, or fall back to defaults.
    """
    print("Computing final scores...")

    # Resolve weights dynamically
    active_weights = WEIGHTS
    if jd_schema and jd_schema.weights:
        # Verify weights sum close to 1.0 or normalize them
        w = jd_schema.weights
        total_w = sum(w.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in w.items()}
        else:
            active_weights = w

    # ── STEP 1: Compute base weighted score ──
    base_score = (
        active_weights.get("semantic_similarity", 0.50) * df["semantic_similarity"] +
        active_weights.get("title_relevance", 0.20)     * df["title_relevance"]     +
        active_weights.get("experience_fit", 0.10)      * df["experience_fit"]      +
        active_weights.get("skill_depth", 0.05)         * df["skill_depth"]         +
        active_weights.get("location", 0.15)            * df["location_score"]
    )

    # ── STEP 2: Apply title gate as a MULTIPLIER ──
    title_gate = 0.35 + 0.65 * df["title_relevance"]
    df["final_score"] = base_score * title_gate

    # ── SORT AND RANK ──
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    print(f"Scoring complete. Top score: {df['final_score'].iloc[0]:.4f}")
    return df
