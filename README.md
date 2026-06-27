---
title: Ai Recruiter

sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
short_description: 'Recruiter that classifies the candidates according to the J '
---

# AdvRecruiter 🤖

An AI-powered candidate ranking engine built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

## What it does

Given a job description and a pool of 100,000 candidate profiles, this system ranks the **top 100 best-fit candidates** — not by keyword matching, but by understanding the *meaning* of their experience, the arc of their career, and whether their behavioral signals suggest they're actually reachable and available.

## Approach (High Level)

1. **Semantic Similarity** — Use `sentence-transformers` to embed the JD and each candidate's profile into vectors. Candidates whose profiles mean the same thing as the JD (even in different words) score higher.
2. **Structured Feature Engineering** — Extract features from career history, education, location, and skills: years of relevant experience, company type (product vs. services), title relevance, etc.
3. **Behavioral Signal Scoring** — Weight candidates by engagement signals: recency of last activity, recruiter response rate, notice period, open-to-work status.
4. **Composite Ranking** — Combine all signals into a final score using a weighted formula, then output the top 100.

## Project Structure

```
AdvRecruiter/
├── data/                    # Place candidates.jsonl here (see data/README.md)
├── src/                     # Core logic
│   ├── data_loader.py       # Load and parse candidates.jsonl
│   ├── feature_engineering.py  # Convert profiles into ML features
│   ├── ranker.py            # Scoring and ranking logic
│   └── utils.py             # Helper functions
├── scripts/
│   ├── preprocess.py        # One-time: compute and save all features
│   └── rank_candidates.py   # Main: produce submission CSV
├── outputs/                 # Final CSV goes here
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/AdvRecruiter.git
cd AdvRecruiter

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place candidates.jsonl in the data/ folder (see data/README.md)
```

## Running

```bash
# Step 1 — Pre-compute features (run once, can take a few minutes)
python scripts/preprocess.py

# Step 2 — Generate the ranked submission CSV (must complete in < 5 min)
python scripts/rank_candidates.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
```

## Compute Constraints

This system is designed to run within the hackathon's hard limits:
- ✅ CPU only (no GPU required)
- ✅ ≤ 16 GB RAM
- ✅ ≤ 5 minutes for the ranking step
- ✅ No external API calls during ranking

## Team

Built for the Redrob Hackathon, 2026.
