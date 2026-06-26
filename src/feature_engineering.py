# src/feature_engineering.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Turn raw candidate data into numbers our model can learn from
#
# This file has 3 main jobs:
#   1. Embed the Job Description into a vector (a list of numbers)
#   2. Compute 7 individual feature scores for every candidate
#   3. Combine those scores into one final ranking score (0.0 – 1.0)
#
# Pipeline flow:
#   data_loader.py → [flat candidate dicts]
#        ↓
#   feature_engineering.py → [feature DataFrame with scores]
#        ↓
#   ranker.py → [top-100 ranked CSV]
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# ── MODEL ──────────────────────────────────────────────────────────────────
# We use a lightweight but powerful sentence embedding model.
# "all-MiniLM-L6-v2" is 80MB, fast on CPU, and produces 384-dim vectors.
# It's the sweet spot for speed vs. quality on CPU-only systems.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ── SCORING WEIGHTS ─────────────────────────────────────────────────────────
# These weights control how much each feature contributes to the final score.
# They must sum to 1.0.
#
# Why these weights?
#   - semantic_similarity (0.35): Hardest to fake, best signal of true fit
#   - title_relevance    (0.20): Prevents HR Managers ranking above ML Engineers
#   - behavioral         (0.15): Ghost candidates must be penalized
#   - experience_fit     (0.10): 5–9 year sweet spot per JD
#   - career_quality     (0.10): Product companies > services firms
#   - skill_depth        (0.05): Depth beyond raw keyword count
#   - location           (0.05): Pune/Noida preferred but not a dealbreaker
WEIGHTS = {
    "semantic_similarity":  0.35,
    "title_relevance":      0.20,
    "behavioral":           0.15,
    "experience_fit":       0.10,
    "career_quality":       0.10,
    "skill_depth":          0.05,
    "location":             0.05,
}


# ── AI TITLE RELEVANCE MAP ─────────────────────────────────────────────────
# Maps job title keywords → relevance score (0.0 to 1.0)
# We check if any of these keywords appear in the candidate's current_title.
# We go from most specific to least specific.
TITLE_RELEVANCE_MAP = [
    # Perfect match — these are exactly what the JD is hiring for
    (["senior ai engineer", "staff ai engineer", "principal ai engineer",
      "lead ai engineer", "founding engineer ai"], 1.0),

    # Strong match — applied ML / NLP / search / ranking roles
    (["ml engineer", "machine learning engineer", "nlp engineer",
      "search engineer", "ranking engineer", "retrieval engineer",
      "recommendation", "personalization engineer",
      "applied scientist", "applied ml", "applied ai",
      "ai engineer", "llm engineer", "recsys"], 0.95),

    # Good match — research with production, or adjacent AI roles
    (["research engineer", "research scientist", "data scientist",
      "deep learning engineer", "cv engineer",
      "recommendation engineer", "personalization engineer"], 0.80),

    # Partial match — engineers who could do this with some AI background
    (["backend engineer", "software engineer", "full stack engineer",
      "platform engineer", "infrastructure engineer",
      "senior engineer", "senior software"], 0.45),

    # Weak match — data roles that have some overlap
    (["data engineer", "analytics engineer", "bi engineer",
      "data analyst", "product analyst"], 0.30),

    # Near-zero match — clearly wrong domains
    (["hr", "human resource", "recruiter", "graphic", "designer",
      "content writer", "accountant", "civil engineer",
      "mechanical engineer", "sales", "marketing",
      "customer support", "project manager",
      "operations manager", "business analyst"], 0.0),
]


# ── JD TEXT ────────────────────────────────────────────────────────────────
# The core text we're ranking candidates against.
# We extract the most important sections of the JD to build a rich embedding.
JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI.
5-9 years experience. Series A AI-native talent intelligence platform.

Required: Production experience with embeddings-based retrieval systems
using sentence-transformers, BGE, E5, OpenAI embeddings or similar.
Production experience with vector databases: Pinecone, Weaviate, Qdrant,
Milvus, FAISS, Elasticsearch, OpenSearch. Strong Python. Hands-on experience
designing evaluation frameworks for ranking systems: NDCG, MRR, MAP,
offline-to-online correlation, A/B testing.

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT). Learning-to-rank
models (XGBoost, neural LTR). HR-tech or marketplace products.
Open-source AI/ML contributions.

Ideal: 6-8 years total, 4-5 in applied ML at product companies (not services).
Shipped end-to-end ranking, search, or recommendation system to real users.
Strong opinions on hybrid vs dense retrieval, offline vs online evaluation,
when to fine-tune vs prompt LLMs. Located in or willing to relocate to
Noida or Pune India.
"""


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — EMBEDDING THE JOB DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════

def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence embedding model.

    What is a sentence embedding model?
    → It's an AI model trained to convert any text into a list of numbers
      (called a vector or embedding). The key property: texts that MEAN the
      same thing end up with SIMILAR number-lists, even if the words differ.

    Example:
      "vector database" → [0.12, -0.45, 0.87, ...]  (384 numbers)
      "ANN index"       → [0.11, -0.44, 0.85, ...]  (384 numbers, very close!)
      "cooking recipe"  → [0.92, 0.31, -0.22, ...]  (very different!)
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded.")
    return model


def embed_job_description(model: SentenceTransformer) -> np.ndarray:
    """
    Convert the Job Description text into a 384-dimensional embedding vector.

    We call this ONCE and reuse the result for all 100K candidates.
    No need to re-embed the JD for every candidate.

    Returns:
        A numpy array of shape (384,) — the JD's "meaning fingerprint"
    """
    print("Embedding Job Description...")
    jd_embedding = model.encode(JD_TEXT, convert_to_numpy=True)
    print(f"JD embedding shape: {jd_embedding.shape}")
    return jd_embedding


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — INDIVIDUAL FEATURE SCORE FUNCTIONS
# Each function takes one candidate dict and returns a float from 0.0 to 1.0
# ═══════════════════════════════════════════════════════════════════════════

def score_semantic_similarity(
    candidate_embedding: np.ndarray,
    jd_embedding: np.ndarray
) -> float:
    """
    Compute how semantically similar the candidate's profile is to the JD.

    What is cosine similarity?
    → It measures the "angle" between two vectors. If the angle is 0° (same
      direction), similarity = 1.0 (perfect match). If they point in opposite
      directions, similarity = -1.0. In practice for text, values range 0.0–1.0.

    Think of it like: how much does this candidate's "meaning" overlap with
    the JD's "meaning"?
    """
    # cosine_similarity expects 2D arrays, so we reshape (384,) → (1, 384)
    sim = cosine_similarity(
        candidate_embedding.reshape(1, -1),
        jd_embedding.reshape(1, -1)
    )
    # Result is a 2D array [[score]] — we extract just the float
    return float(sim[0][0])


def score_title_relevance(current_title: str) -> float:
    """
    Score how AI-relevant this candidate's job title is.

    This is a rule-based lookup — we check the TITLE_RELEVANCE_MAP above.
    We go top-to-bottom and return the score for the first match we find.

    Why rules instead of ML here?
    → Job title relevance has clear, well-defined boundaries. Rules are
      more interpretable and reliable than letting a model guess.
    """
    title_lower = current_title.lower()

    for title_keywords, score in TITLE_RELEVANCE_MAP:
        # Check if ANY of the keywords in this group appear in the title
        if any(kw in title_lower for kw in title_keywords):
            return score

    # If no rule matched, default to a neutral-low score
    # (unknown title — could be anything)
    return 0.25


def score_experience_fit(years_exp: float) -> float:
    """
    Score how well the candidate's years of experience fits the JD's 5–9 year range.

    We use a "tent" shape: max score in the sweet spot, tapering off outside.

                1.0  ┌────────┐
                     │        │
                0.6  │        ├──────┐
                     │        │      │
                0.2  ┤        │      └──────
                     0  4  6  9  12  15+ years

    This avoids penalizing someone with 10 years as harshly as someone with 1 year.
    """
    if years_exp < 2:
        return 0.1   # Way too junior
    elif years_exp < 4:
        return 0.3   # Too junior for this senior role
    elif years_exp < 5:
        return 0.6   # Borderline — might work
    elif years_exp <= 9:
        return 1.0   # Sweet spot: 5–9 years
    elif years_exp <= 12:
        return 0.75  # Slightly over, but fine
    elif years_exp <= 15:
        return 0.55  # Getting into management territory
    else:
        return 0.35  # Very likely overqualified / management-only


def score_location(
    in_preferred_location: int,
    in_india: int,
    willing_relocate: int,
    work_mode: str
) -> float:
    """
    Score based on how well the candidate's location matches the JD.

    JD says: Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR welcome,
    outside India: case-by-case, no visa sponsorship.

    Logic:
      - In preferred city (Pune/Noida) → best
      - In India, other preferred city → very good
      - In India, willing to relocate → acceptable
      - Outside India, willing to relocate → risky but possible
      - Outside India, not willing → very low
    """
    # in_preferred_location is 1 if they're in Pune/Noida/Hyderabad/etc.
    if in_preferred_location and in_india:
        return 1.0
    elif in_india and willing_relocate:
        return 0.75
    elif in_india and not willing_relocate:
        return 0.5    # In India but in wrong city and won't move
    elif not in_india and willing_relocate:
        return 0.3    # Outside India, willing to relocate — risky
    else:
        return 0.05   # Outside India, not willing to relocate


def score_behavioral_availability(
    last_active_days: int,
    open_to_work: int,
    response_rate: float,
    notice_days: int,
    interview_rate: float,
    response_time_hrs: float
) -> float:
    """
    Score whether the candidate is actually reachable and ready to hire.

    A perfect-on-paper candidate who is "dark" (inactive, low response rate,
    long notice) is, for practical hiring purposes, not actually available.

    We combine 6 signals into one availability score.
    """
    # --- Signal 1: Recency of last login ---
    # Active in last 30 days = great. 6+ months inactive = bad.
    if last_active_days <= 14:
        recency = 1.0
    elif last_active_days <= 30:
        recency = 0.85
    elif last_active_days <= 60:
        recency = 0.65
    elif last_active_days <= 90:
        recency = 0.45
    elif last_active_days <= 180:
        recency = 0.25
    else:
        recency = 0.05   # Inactive for 6+ months

    # --- Signal 2: Open to work flag ---
    # Simple boolean: are they even looking?
    open_score = 1.0 if open_to_work else 0.4

    # --- Signal 3: Recruiter response rate ---
    # response_rate is already 0.0–1.0, so use directly
    response_score = response_rate

    # --- Signal 4: Notice period ---
    # JD says "we'd love sub-30 day notice, can buy out up to 30 days"
    if notice_days <= 15:
        notice_score = 1.0
    elif notice_days <= 30:
        notice_score = 0.9
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.35
    else:
        notice_score = 0.15   # 90+ day notice is a big obstacle

    # --- Signal 5: Interview completion rate ---
    # If they schedule interviews and don't show up, that's a bad sign
    interview_score = interview_rate  # Already 0.0–1.0

    # --- Signal 6: Response time ---
    # Faster is better. Over 72 hours is slow.
    if response_time_hrs <= 4:
        resp_time_score = 1.0
    elif response_time_hrs <= 24:
        resp_time_score = 0.8
    elif response_time_hrs <= 72:
        resp_time_score = 0.55
    elif response_time_hrs <= 168:
        resp_time_score = 0.3
    else:
        resp_time_score = 0.1

    # --- Weighted average of all 6 signals ---
    behavioral = (
        0.30 * recency          +  # Most important: are they even active?
        0.25 * response_score   +  # Do they reply to recruiters?
        0.20 * notice_score     +  # Can we start soon?
        0.10 * open_score       +  # Are they officially looking?
        0.10 * interview_score  +  # Do they show up?
        0.05 * resp_time_score     # How fast do they reply?
    )
    return behavioral


def score_career_quality(
    total_companies: int,
    services_company_count: int,
    max_tenure_months: int,
    avg_tenure_months: float,
    career_text: str
) -> float:
    """
    Score the quality and relevance of the candidate's career history.

    We look at 3 things:
    1. What fraction of jobs were at product companies (not consulting)?
    2. How stable is their job history? (long stints = deep expertise)
    3. Does their career description mention production-system keywords?
    """
    # --- Sub-score 1: Product company ratio ---
    # (1 - services_fraction) gives us the product company fraction
    if total_companies > 0:
        services_fraction = services_company_count / total_companies
        product_score = 1.0 - services_fraction
    else:
        product_score = 0.5   # No data — neutral

    # --- Sub-score 2: Job stability (average tenure) ---
    # Short average tenure = job hopping = shallow expertise
    # 24+ months avg = healthy depth
    if avg_tenure_months >= 36:
        stability_score = 1.0
    elif avg_tenure_months >= 24:
        stability_score = 0.85
    elif avg_tenure_months >= 18:
        stability_score = 0.65
    elif avg_tenure_months >= 12:
        stability_score = 0.45
    else:
        stability_score = 0.2    # Very short stints — unstable

    # --- Sub-score 3: Production keywords in career description ---
    # We look for words that suggest real, shipped systems (not just research/demos)
    production_keywords = [
        "production", "deployed", "shipped", "real users", "at scale",
        "million", "billion", "latency", "throughput", "serving",
        "a/b test", "online", "pipeline", "inference", "monitoring"
    ]
    career_lower = career_text.lower()
    keyword_hits = sum(1 for kw in production_keywords if kw in career_lower)
    # Cap at 5 hits → 1.0
    production_score = min(keyword_hits / 5.0, 1.0)

    # --- Weighted combination ---
    quality = (
        0.45 * product_score     +  # Product vs services is most important
        0.30 * stability_score   +  # Tenure stability
        0.25 * production_score     # Production system evidence
    )
    return quality


def score_skill_depth(
    jd_skill_overlap: int,
    advanced_skills: int,
    avg_skill_months: float,
    github_score: float,
    endorsements: int
) -> float:
    """
    Score the depth of the candidate's relevant skills.

    This is different from just counting JD skill overlaps.
    We want to know: how DEEP is their AI/ML expertise?

    We look at:
    - JD skill overlap count (normalized)
    - How many skills are at advanced/expert level
    - Average months of skill usage (experience depth)
    - GitHub activity (signals real open-source work)
    - Endorsements from peers (social validation)
    """
    # --- Sub-score 1: JD skill overlap ---
    # Cap at 8 matching skills → 1.0 (more than 8 is diminishing returns)
    overlap_score = min(jd_skill_overlap / 8.0, 1.0)

    # --- Sub-score 2: Advanced/expert skill count ---
    # Cap at 6 → 1.0
    advanced_score = min(advanced_skills / 6.0, 1.0)

    # --- Sub-score 3: Average skill usage depth (months) ---
    # 36+ months average = seasoned in their skills
    if avg_skill_months >= 36:
        depth_score = 1.0
    elif avg_skill_months >= 24:
        depth_score = 0.8
    elif avg_skill_months >= 12:
        depth_score = 0.55
    else:
        depth_score = 0.3

    # --- Sub-score 4: GitHub activity ---
    # github_score is -1 (no GitHub) or 0–100
    # -1 means no GitHub linked — we treat this as neutral (0.4), not bad
    if github_score == -1:
        github_norm = 0.4    # No GitHub linked — we don't know
    else:
        github_norm = github_score / 100.0

    # --- Sub-score 5: Endorsements (peer validation) ---
    # Cap at 50 endorsements → 1.0
    endorsement_score = min(endorsements / 50.0, 1.0)

    # --- Weighted combination ---
    skill = (
        0.35 * overlap_score      +  # JD relevance of skills
        0.25 * advanced_score     +  # Seniority of skills
        0.20 * depth_score        +  # How long they've used them
        0.15 * github_norm        +  # Open source activity
        0.05 * endorsement_score     # Peer validation
    )
    return skill


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN PIPELINE FUNCTIONS
# These are the functions called from scripts/preprocess.py
# ═══════════════════════════════════════════════════════════════════════════

def compute_candidate_embeddings(
    candidates: list[dict],
    model: SentenceTransformer,
    batch_size: int = 32,
    chunk_size: int = 5000,
    mmap_path: str = None,
) -> np.ndarray:
    """
    Embed all candidates' full_text into 384-dimensional vectors.

    Uses disk-based memory mapping (np.memmap) so embeddings are written
    directly to disk chunk by chunk. Only one chunk lives in RAM at a time,
    keeping memory usage flat even for 100K candidates.

    Supports RESUME: if mmap_path already exists with the correct shape,
    we detect which chunks are done (non-zero rows) and skip them.

    Parameters:
        candidates  : list of candidate dicts (from data_loader.py)
        model       : the loaded SentenceTransformer model
        batch_size  : candidates per encode() call (keep small to save RAM)
        chunk_size  : candidates per disk-write chunk (5000 = ~7.5MB per flush)
        mmap_path   : path to save the .npy memmap file. If None, falls back
                      to a regular in-memory array (used for small sample runs).

    Returns:
        A numpy array of shape (N, 384) where N = number of candidates.
        For large runs this is a read-only memmap backed by disk.
    """
    n = len(candidates)
    dim = 384  # all-MiniLM-L6-v2 output dimension
    print(f"Embedding {n} candidate profiles (batch={batch_size}, chunk={chunk_size})...")

    texts = [c["full_text"] for c in candidates]

    # ── Small run (no mmap_path given): just encode everything into RAM ──
    if mmap_path is None or n <= chunk_size:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    # ── Large run: write directly to disk via memmap ──────────────────
    #
    # np.memmap creates a file on disk and maps it into Python's address space.
    # Writing to `mmap[i]` writes directly to disk — RAM stays small.
    #
    # Shape on disk: (n_candidates, 384) of float32 = ~154MB for 100K candidates
    import os

    resume = os.path.exists(mmap_path)
    mode = "r+" if resume else "w+"  # r+ = open existing, w+ = create new

    mmap = np.memmap(mmap_path, dtype="float32", mode=mode, shape=(n, dim))

    # Detect resume point: find first chunk where all rows are still zero
    start_chunk = 0
    if resume:
        for ci in range(0, n, chunk_size):
            end = min(ci + chunk_size, n)
            # If ANY row in this chunk is non-zero, the chunk was already written
            if np.any(mmap[ci:end]):
                start_chunk = ci + chunk_size
            else:
                break
        if start_chunk > 0:
            print(f"  Resuming from candidate {start_chunk} (chunks already on disk).")
        else:
            print("  No previous progress found — starting from scratch.")

    # Process one chunk at a time
    chunks = list(range(start_chunk, n, chunk_size))
    for ci in tqdm(chunks, desc="Embedding chunks"):
        end = min(ci + chunk_size, n)
        chunk_texts = texts[ci:end]

        # Embed this chunk entirely into RAM (small: ~7.5MB for 5000 candidates)
        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Write to disk immediately and flush (free RAM before next chunk)
        mmap[ci:end] = chunk_emb
        mmap.flush()          # force write to disk now
        del chunk_emb         # free the chunk from RAM

    print(f"Embeddings written to disk: {mmap_path}  shape={mmap.shape}")
    return np.array(mmap)     # return a regular array for downstream use


def compute_all_features(
    candidates: list[dict],
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray
) -> pd.DataFrame:
    """
    For every candidate, compute all 7 feature scores and return a DataFrame.

    A DataFrame is like an Excel sheet — rows = candidates, columns = features.
    Each score is a float from 0.0 to 1.0.

    Parameters:
        candidates            : list of flat candidate dicts
        candidate_embeddings  : (N, 384) array of candidate embeddings
        jd_embedding          : (384,) array of JD embedding

    Returns:
        pd.DataFrame with columns: candidate_id + all score columns
    """
    print("Computing feature scores for all candidates...")
    rows = []

    for i, cand in enumerate(tqdm(candidates, desc="Scoring candidates")):

        # --- Semantic similarity ---
        sem_sim = score_semantic_similarity(candidate_embeddings[i], jd_embedding)

        # --- Title relevance ---
        title_rel = score_title_relevance(cand["current_title"])

        # --- Experience fit ---
        exp_fit = score_experience_fit(cand["years_exp"])

        # --- Location ---
        loc = score_location(
            cand["in_preferred_location"],
            cand["in_india"],
            cand["willing_relocate"],
            cand["work_mode"]
        )

        # --- Behavioral availability ---
        behav = score_behavioral_availability(
            cand["last_active_days"],
            cand["open_to_work"],
            cand["response_rate"],
            cand["notice_days"],
            cand["interview_rate"],
            cand["response_time_hrs"]
        )

        # --- Career quality ---
        career_q = score_career_quality(
            cand["total_companies"],
            cand["services_company_count"],
            cand["max_tenure_months"],
            cand["avg_tenure_months"],
            cand["career_text"]
        )

        # --- Skill depth ---
        skill_d = score_skill_depth(
            cand["jd_skill_overlap"],
            cand["advanced_skills"],
            cand["avg_skill_months"],
            cand["github_score"],
            cand["endorsements"]
        )

        rows.append({
            # Identifiers (kept for output generation later)
            "candidate_id":        cand["candidate_id"],
            "current_title":       cand["current_title"],
            "city":                cand["location"],   # renamed: avoid collision with location_score
            "years_exp":           cand["years_exp"],
            "headline":            cand["headline"],
            "summary":             cand["summary"],
            "career_text":         cand["career_text"],
            "response_rate":       cand["response_rate"],
            "notice_days":         cand["notice_days"],
            "last_active_days":    cand["last_active_days"],
            "github_score":        cand["github_score"],
            "jd_skill_overlap":    cand["jd_skill_overlap"],
            "is_honeypot":         cand["is_honeypot"],
            "is_non_ai_title":     cand["is_non_ai_title"],

            # The 7 feature scores (note: location_score is the float, city is the string)
            "semantic_similarity": sem_sim,
            "title_relevance":     title_rel,
            "experience_fit":      exp_fit,
            "location_score":      loc,
            "behavioral":          behav,
            "career_quality":      career_q,
            "skill_depth":         skill_d,
        })

    df = pd.DataFrame(rows)
    print(f"Feature matrix shape: {df.shape}")
    return df



