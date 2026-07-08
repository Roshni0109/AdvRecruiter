# src/feature_engineering.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Turn raw/normalized candidate data into numbers
#
# This file is refactored to compute scores against any CanonicalJobDescription.
# ─────────────────────────────────────────────────────────────────────

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.schemas import CanonicalJobDescription

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── BACKWARD COMPATIBLE FALLBACK CONSTANTS ──────────────────────────────────
WEIGHTS = {
    "semantic_similarity":  0.50,
    "title_relevance":      0.20,
    "experience_fit":       0.10,
    "skill_depth":          0.05,
    "location":             0.15,
}

TITLE_RELEVANCE_MAP = [
    (["senior ai engineer", "staff ai engineer", "principal ai engineer",
      "lead ai engineer", "founding engineer ai"], 1.0),
    (["ml engineer", "machine learning engineer", "nlp engineer",
      "search engineer", "ranking engineer", "retrieval engineer",
      "recommendation", "personalization engineer",
      "applied scientist", "applied ml", "applied ai",
      "ai engineer", "llm engineer", "recsys"], 0.95),
    (["research engineer", "research scientist", "data scientist",
      "deep learning engineer", "cv engineer",
      "recommendation engineer", "personalization engineer"], 0.80),
    (["backend engineer", "software engineer", "full stack engineer",
      "platform engineer", "infrastructure engineer",
      "senior engineer", "senior software"], 0.45),
    (["data engineer", "analytics engineer", "bi engineer",
      "data analyst", "product analyst"], 0.30),
    (["hr", "human resource", "recruiter", "graphic", "designer",
      "content writer", "accountant", "civil engineer",
      "mechanical engineer", "sales", "marketing",
      "customer support", "project manager",
      "operations manager", "business analyst"], 0.0),
]

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


# ── EMBEDDING MODELS ───────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded.")
    return model


def embed_job_description(model: SentenceTransformer, jd_text: str = None) -> np.ndarray:
    text = jd_text if jd_text is not None else JD_TEXT
    print("Embedding Job Description...")
    jd_embedding = model.encode(text, convert_to_numpy=True)
    print(f"JD embedding shape: {jd_embedding.shape}")
    return jd_embedding


# ── FEATURE SCORING FUNCTIONS ──────────────────────────────────────────────

def score_semantic_similarity(
    candidate_embedding: np.ndarray,
    jd_embedding: np.ndarray
) -> float:
    sim = cosine_similarity(
        candidate_embedding.reshape(1, -1),
        jd_embedding.reshape(1, -1)
    )
    return float(sim[0][0])


def score_title_relevance(current_title: str, jd_schema: CanonicalJobDescription = None) -> float:
    title_lower = current_title.lower()

    # Dynamic matching: If jd_schema contains a target title keyword, rank matching titles higher
    if jd_schema and jd_schema.title:
        target_words = [w.strip().lower() for w in re.split(r'[-\s/]', jd_schema.title) if w.strip()]
        # Check if the title is clearly in the non-matching list
        if any(non_ai in title_lower for non_ai in jd_schema.non_ai_titles):
            return 0.0
        
        # Direct exact match or contains complete phrase
        if jd_schema.title.lower() in title_lower:
            return 1.0
            
        # Count overlapping keywords
        matched_words = sum(1 for w in target_words if w in title_lower)
        if len(target_words) > 0 and matched_words == len(target_words):
            return 0.95
        elif matched_words > 0:
            return 0.80

    # Fallback to standard relevance map
    for title_keywords, score in TITLE_RELEVANCE_MAP:
        if any(kw in title_lower for kw in title_keywords):
            return score
    return 0.25


def score_experience_fit(years_exp: float, jd_schema: CanonicalJobDescription = None) -> float:
    min_exp = jd_schema.min_years_exp if jd_schema else 5.0
    max_exp = jd_schema.max_years_exp if jd_schema else 9.0

    # Handle cases where experience bounds are unset
    if min_exp == 0.0 and max_exp == 99.0:
        return 1.0  # experience-agnostic

    if years_exp < min_exp - 3.0:
        return 0.1
    elif years_exp < min_exp - 1.0:
        return 0.3
    elif years_exp < min_exp:
        return 0.6
    elif years_exp <= max_exp:
        return 1.0
    elif years_exp <= max_exp + 3.0:
        return 0.75
    elif years_exp <= max_exp + 6.0:
        return 0.55
    else:
        return 0.35


def score_location(
    in_preferred_location: int,
    in_india: int,
    willing_relocate: int,
    work_mode: str,
    jd_schema: CanonicalJobDescription = None
) -> float:
    # Location mapping
    if in_preferred_location and in_india:
        return 1.0
    elif in_india and willing_relocate:
        return 0.75
    elif in_india and not willing_relocate:
        return 0.5
    elif not in_india and willing_relocate:
        return 0.3
    else:
        return 0.05


def score_behavioral_availability(
    last_active_days: int,
    open_to_work: int,
    response_rate: float,
    notice_days: int,
    interview_rate: float,
    response_time_hrs: float
) -> float:
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
        recency = 0.05

    open_score = 1.0 if open_to_work else 0.4
    response_score = response_rate

    if notice_days <= 15:
        notice_score = 1.0
    elif notice_days <= 30:
        notice_score = 0.9
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.35
    else:
        notice_score = 0.15

    interview_score = interview_rate

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

    behavioral = (
        0.30 * recency          +
        0.25 * response_score   +
        0.20 * notice_score     +
        0.10 * open_score       +
        0.10 * interview_score  +
        0.05 * resp_time_score
    )
    return behavioral


def score_career_quality(
    total_companies: int,
    services_company_count: int,
    max_tenure_months: int,
    avg_tenure_months: float,
    career_text: str
) -> float:
    if total_companies > 0:
        services_fraction = services_company_count / total_companies
        product_score = 1.0 - services_fraction
    else:
        product_score = 0.5

    if avg_tenure_months >= 36:
        stability_score = 1.0
    elif avg_tenure_months >= 24:
        stability_score = 0.85
    elif avg_tenure_months >= 18:
        stability_score = 0.65
    elif avg_tenure_months >= 12:
        stability_score = 0.45
    else:
        stability_score = 0.2

    production_keywords = [
        "production", "deployed", "shipped", "real users", "at scale",
        "million", "billion", "latency", "throughput", "serving",
        "a/b test", "online", "pipeline", "inference", "monitoring"
    ]
    career_lower = career_text.lower()
    keyword_hits = sum(1 for kw in production_keywords if kw in career_lower)
    production_score = min(keyword_hits / 5.0, 1.0)

    quality = (
        0.45 * product_score     +
        0.30 * stability_score   +
        0.25 * production_score
    )
    return quality


def score_skill_depth(
    jd_skill_overlap: int,
    advanced_skills: int,
    avg_skill_months: float,
    github_score: float,
    endorsements: int
) -> float:
    overlap_score = min(jd_skill_overlap / 8.0, 1.0)
    advanced_score = min(advanced_skills / 6.0, 1.0)

    if avg_skill_months >= 36:
        depth_score = 1.0
    elif avg_skill_months >= 24:
        depth_score = 0.8
    elif avg_skill_months >= 12:
        depth_score = 0.55
    else:
        depth_score = 0.3

    if github_score == -1:
        github_norm = 0.4
    else:
        github_norm = github_score / 100.0

    endorsement_score = min(endorsements / 50.0, 1.0)

    skill = (
        0.35 * overlap_score      +
        0.25 * advanced_score     +
        0.20 * depth_score        +
        0.15 * github_norm        +
        0.05 * endorsement_score
    )
    return skill


# ── MAIN PIPELINE FUNCTIONS ──────────────────────────────────────────────────

def compute_candidate_embeddings(
    candidates: list[dict],
    model: SentenceTransformer,
    batch_size: int = 32,
    chunk_size: int = 5000,
    mmap_path: str = None,
) -> np.ndarray:
    n = len(candidates)
    dim = 384
    print(f"Embedding {n} candidate profiles...")

    texts = [c.get("full_text", "") for c in candidates]

    if mmap_path is None or n <= chunk_size:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    import os
    resume = os.path.exists(mmap_path)
    mode = "r+" if resume else "w+"
    mmap = np.memmap(mmap_path, dtype="float32", mode=mode, shape=(n, dim))

    start_chunk = 0
    if resume:
        for ci in range(0, n, chunk_size):
            end = min(ci + chunk_size, n)
            if np.any(mmap[ci:end]):
                start_chunk = ci + chunk_size
            else:
                break

    chunks = list(range(start_chunk, n, chunk_size))
    for ci in tqdm(chunks, desc="Embedding chunks"):
        end = min(ci + chunk_size, n)
        chunk_texts = texts[ci:end]

        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        mmap[ci:end] = chunk_emb
        mmap.flush()
        del chunk_emb

    return np.array(mmap)


def compute_all_features(
    candidates: list[dict],
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    jd_schema: CanonicalJobDescription = None
) -> pd.DataFrame:
    """
    Compute features for all candidates relative to the given CanonicalJobDescription.
    """
    # 1. Normalize JD if not provided
    if jd_schema is None:
        from src.normalization import normalize_job_description
        jd_schema = normalize_job_description(JD_TEXT)

    print(f"Computing features against JD: '{jd_schema.title}' (Exp: {jd_schema.min_years_exp}-{jd_schema.max_years_exp} yrs)")
    rows = []

    for i, cand in enumerate(tqdm(candidates, desc="Scoring candidates")):
        # Get values
        current_title = cand.get("current_title", "")
        location = cand.get("location", "").lower()
        country = cand.get("country", "").lower()
        skills = cand.get("skills", [])
        career_history = cand.get("career_history", [])

        # Validate required fields
        if not cand.get("candidate_id"):
            continue

        # Dynamic feature recalculations
        # A. Skill overlap
        skill_names = {s.get("name", "").lower() for s in skills if isinstance(s, dict)} if isinstance(skills, list) else set()
        if not skill_names and isinstance(skills, list):
            skill_names = {s.lower() for s in skills if isinstance(s, str)}
        jd_skill_overlap = len(skill_names & set(jd_schema.core_skills))

        # B. Location preferred check
        in_preferred_location = any(city in location for city in jd_schema.preferred_cities)
        in_india = (country == "india")

        # C. Title relevance match
        title_rel = score_title_relevance(current_title, jd_schema)

        # D. Services company count
        services_company_count = sum(
            1 for job in career_history
            if isinstance(job, dict) and any(firm in job.get("company", "").lower() for firm in jd_schema.services_firms)
        )

        # E. Non AI title check
        is_non_ai_title = int(any(title in current_title.lower() for title in jd_schema.non_ai_titles))

        # Compute Scores
        sem_sim = score_semantic_similarity(candidate_embeddings[i], jd_embedding)
        exp_fit = score_experience_fit(cand.get("years_exp", 0.0), jd_schema)
        
        loc_score = score_location(
            int(in_preferred_location),
            int(in_india),
            cand.get("willing_relocate", 0),
            cand.get("work_mode", ""),
            jd_schema
        )

        behav = 0.0
        career_q = 0.0

        expert_skills = sum(1 for s in skills if isinstance(s, dict) and s.get("proficiency") == "expert")
        advanced_skills = sum(1 for s in skills if isinstance(s, dict) and s.get("proficiency") in ("expert", "advanced"))
        durations = [s.get("duration_months", 0) for s in skills if isinstance(s, dict)]
        avg_skill_months = sum(durations) / len(durations) if durations else 0

        skill_d = score_skill_depth(
            jd_skill_overlap,
            advanced_skills,
            avg_skill_months,
            cand.get("github_score", -1),
            cand.get("endorsements", 0)
        )

        rows.append({
            "candidate_id":        cand["candidate_id"],
            "current_title":       current_title,
            "city":                cand.get("city", cand.get("location", "")),
            "years_exp":           cand.get("years_exp", 0.0),
            "headline":            cand.get("headline", ""),
            "summary":             cand.get("summary", ""),
            "career_text":         cand.get("career_text", ""),
            "response_rate":       cand.get("response_rate", 0.0),
            "notice_days":         cand.get("notice_days", 90),
            "last_active_days":    cand.get("last_active_days", 9999),
            "github_score":        cand.get("github_score", -1.0),
            "jd_skill_overlap":    jd_skill_overlap,
            "is_honeypot":         cand.get("is_honeypot", False),
            "is_non_ai_title":     is_non_ai_title,

            "semantic_similarity": sem_sim,
            "title_relevance":     title_rel,
            "experience_fit":      exp_fit,
            "location_score":      loc_score,
            "behavioral":          behav,
            "career_quality":      career_q,
            "skill_depth":         skill_d,
        })

    df = pd.DataFrame(rows)
    print(f"Feature matrix shape: {df.shape}")
    return df
