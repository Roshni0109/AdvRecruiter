# src/ranker.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Score and rank candidates using LightGBM
#
# After feature_engineering.py turns candidates into numbers,
# this file takes those numbers and produces a final score (0-1)
# for each candidate — telling us "how good a fit are they?"
#
# We use LightGBM (a type of gradient boosting model) because:
# - It's very fast (can handle 100K candidates easily)
# - It works well even without a GPU
# - It's great at combining many features into one score
# ─────────────────────────────────────────────────────────────────────

# TODO: Will be implemented in Step 4 (Ranking Model)
