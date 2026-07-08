# src/data_loader.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Load and clean candidate data from candidates.jsonl
#
# This file is the "front door" of our pipeline. It delegates to
# src/normalization.py to map and normalize candidate fields dynamically.
# ─────────────────────────────────────────────────────────────────────

import json
from datetime import datetime
from tqdm import tqdm
from src.normalization import normalize_candidate, TODAY, CANDIDATE_ID_KEYS

# ── CONSTANTS (Backward Compatibility Fallbacks) ─────────────────────────
JD_CORE_SKILLS = {
    "embeddings", "sentence-transformers", "vector database", "pinecone",
    "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch",
    "retrieval", "ranking", "nlp", "information retrieval", "bge", "e5",
    "ndcg", "mrr", "map", "a/b testing", "fine-tuning", "lora", "qlora",
    "peft", "llm", "rag", "hybrid search", "reranking", "xgboost",
    "lightgbm", "python", "pytorch", "transformers"
}

PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurugram", "gurgaon", "delhi ncr"
}

SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra"
}

NON_AI_TITLES = {
    "hr manager", "graphic designer", "content writer", "accountant",
    "civil engineer", "mechanical engineer", "sales executive",
    "marketing manager", "customer support", "business analyst",
    "project manager", "operations manager"
}


# ── MAIN LOADING FUNCTION ──────────────────────────────────────────────────

def load_candidates(filepath: str, limit: int = None) -> list[dict]:
    """
    Read candidates from a JSON/JSONL file/array and return a list of cleaned dicts.
    Supports single JSON objects, JSON lists, index-oriented dictionaries, and stream JSONL.
    """
    candidates = []
    print(f"Loading candidates from: {filepath}")

    try:
        # First, try to read the entire file and parse as a single JSON structure
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if content:
            try:
                raw_data = json.loads(content)
                if isinstance(raw_data, list):
                    # 1. JSON Array of candidates
                    print(f"  Read {len(raw_data)} candidates as JSON array.")
                    for i, raw in enumerate(raw_data):
                        if limit and i >= limit:
                            break
                        candidates.append(extract_candidate_features(raw))
                elif isinstance(raw_data, dict):
                    # 2. Dictionary structure
                    if "candidates" in raw_data and isinstance(raw_data["candidates"], list):
                        # Wrapper dict containing candidates list
                        cand_list = raw_data["candidates"]
                        print(f"  Read {len(cand_list)} candidates from nested list.")
                        for i, raw in enumerate(cand_list):
                            if limit and i >= limit:
                                break
                            candidates.append(extract_candidate_features(raw))
                    elif any(isinstance(v, dict) and any(k in v for k in ["candidate_id", "id", "RegistrationID"]) for v in raw_data.values()):
                        # Index-oriented dictionary (e.g. {"0": {...}, "1": {...}})
                        cand_dict_values = list(raw_data.values())
                        print(f"  Read {len(cand_dict_values)} candidates from index dictionary.")
                        for i, raw in enumerate(cand_dict_values):
                            if limit and i >= limit:
                                break
                            candidates.append(extract_candidate_features(raw))
                    else:
                        # Single candidate JSON object
                        print("  Read single candidate JSON object.")
                        candidates.append(extract_candidate_features(raw_data))
                return candidates
            except json.JSONDecodeError:
                # If parsing the whole file failed, fall back to line-by-line JSONL streaming
                pass

        # 3. Stream JSONL line-by-line fallback
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Reading candidates")):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    candidates.append(extract_candidate_features(raw))
                except json.JSONDecodeError:
                    print(f"  Warning: Skipping malformed JSON at line {i+1}")
                    continue
    except Exception as e:
        print(f"Error loading candidates from {filepath}: {e}")

    print(f"Loaded {len(candidates)} candidates successfully.")
    return candidates


# ── FEATURE EXTRACTION ─────────────────────────────────────────────────────

def extract_candidate_features(raw: dict) -> dict:
    """
    Normalizes raw candidate dictionary into a flat dictionary using normalize_candidate.
    Keeps backward-compatible pre-computed fields using default JDs.
    """
    # 1. Graceful error handling for missing/empty records
    if not raw or not isinstance(raw, dict):
        print("Warning: Empty or non-dictionary candidate record encountered.")
        raw = {}

    # 2. Normalize raw data to CanonicalCandidate schema
    canonical = normalize_candidate(raw)
    
    # Convert back to flat dict representation
    cand_dict = canonical.to_dict()

    # 3. Add backward-compatible precomputed fields using fallback constants
    # (These will be overwritten dynamically during feature engineering if a new JD is present)
    skills = raw.get("skills", raw.get("Skills", []))
    skill_names = {s.get("name", "").lower() for s in skills if isinstance(s, dict)} if isinstance(skills, list) else set()
    if not skill_names and isinstance(skills, list):
        skill_names = {s.lower() for s in skills if isinstance(s, str)}

    cand_dict["jd_skill_overlap"] = len(skill_names & JD_CORE_SKILLS)
    cand_dict["expert_skills"] = sum(1 for s in skills if isinstance(s, dict) and s.get("proficiency") == "expert")
    cand_dict["advanced_skills"] = sum(1 for s in skills if isinstance(s, dict) and s.get("proficiency") in ("expert", "advanced"))
    cand_dict["total_skills"] = len(skills)
    
    durations = [s.get("duration_months", 0) for s in skills if isinstance(s, dict)]
    cand_dict["avg_skill_months"] = sum(durations) / len(durations) if durations else 0

    career = raw.get("career_history", raw.get("career", []))
    career_list = career if isinstance(career, list) else []
    cand_dict["total_companies"] = len(career_list)
    cand_dict["services_company_count"] = sum(
        1 for job in career_list
        if isinstance(job, dict) and any(firm in job.get("company", "").lower() for firm in SERVICES_FIRMS)
    )
    
    tenures = [job.get("duration_months", 0) for job in career_list if isinstance(job, dict)]
    cand_dict["max_tenure_months"] = max(tenures) if tenures else 0
    cand_dict["avg_tenure_months"] = sum(tenures) / len(tenures) if tenures else 0

    education = raw.get("education", [])
    edu_tiers = [e.get("tier", "unknown") for e in education if isinstance(e, dict)]
    tier_map = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    scores = [tier_map.get(t.lower(), 0) for t in edu_tiers]
    cand_dict["top_edu_tier"] = max(scores) if scores else 0

    cand_dict["in_preferred_location"] = int(any(city in canonical.location.lower() for city in PREFERRED_CITIES))
    cand_dict["in_india"] = int(canonical.country.lower() == "india")
    cand_dict["is_non_ai_title"] = int(any(title in canonical.current_title.lower() for title in NON_AI_TITLES))

    # Detect honeypot (Disabled)
    cand_dict["is_honeypot"] = False

    # Make city/location mapping consistent
    cand_dict["city"] = canonical.location

    return cand_dict


if __name__ == "__main__":
    import os
    import sys
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.jsonl")
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    print("Testing loader...")
    candidates = load_candidates(filepath, limit=5)
    print(f"Loaded {len(candidates)} candidates.")
