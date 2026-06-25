# src/data_loader.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Load and clean candidate data from candidates.jsonl
#
# This file is the "front door" of our pipeline. Before any AI magic
# happens, we need to:
#   1. Open the file and read candidates one by one (streaming)
#   2. Parse the JSON on each line into a Python dictionary
#   3. Detect and flag honeypots (candidates with impossible profiles)
#   4. Flatten the nested JSON into a simple, clean structure
#
# Think of this as the "data cleaning" step — garbage in, garbage out.
# If we feed bad data to our model, it will produce bad rankings.
# ─────────────────────────────────────────────────────────────────────

import json                  # built-in: reads JSON text into Python dicts
from datetime import datetime  # built-in: for working with dates
from tqdm import tqdm        # shows a progress bar (e.g. "50000/100000")


# ── CONSTANTS ──────────────────────────────────────────────────────────────
# These are the AI/ML skills the JD specifically mentions.
# We'll check how many of these each candidate has.
JD_CORE_SKILLS = {
    "embeddings", "sentence-transformers", "vector database", "pinecone",
    "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch",
    "retrieval", "ranking", "nlp", "information retrieval", "bge", "e5",
    "ndcg", "mrr", "map", "a/b testing", "fine-tuning", "lora", "qlora",
    "peft", "llm", "rag", "hybrid search", "reranking", "xgboost",
    "lightgbm", "python", "pytorch", "transformers"
}

# Cities that the JD considers "preferred" locations for this role
PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurugram", "gurgaon", "delhi ncr"
}

# Consulting/services firms the JD explicitly says are a negative signal
SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra"
}

# Job titles that are clearly NOT AI/ML roles (used for title relevance scoring)
NON_AI_TITLES = {
    "hr manager", "graphic designer", "content writer", "accountant",
    "civil engineer", "mechanical engineer", "sales executive",
    "marketing manager", "customer support", "business analyst",
    "project manager", "operations manager"
}

# Today's date — used to calculate how recently a candidate was active
TODAY = datetime.today()


# ── MAIN LOADING FUNCTION ──────────────────────────────────────────────────

def load_candidates(filepath: str, limit: int = None) -> list[dict]:
    """
    Read candidates from a JSONL file and return a list of cleaned dicts.

    What is JSONL?
    → Each line in the file is one complete JSON object (one candidate).
    → We read line by line — this is called "streaming" — so we never
      load the whole 465MB file into memory at once.

    Parameters:
        filepath : path to candidates.jsonl
        limit    : if set, only load the first N candidates (useful for testing)

    Returns:
        A list of dictionaries, one per candidate, with extracted features.
    """
    candidates = []

    print(f"Loading candidates from: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        # tqdm wraps the file to show a live progress bar
        for i, line in enumerate(tqdm(f, desc="Reading candidates")):

            # Stop early if we only want a sample (for quick testing)
            if limit and i >= limit:
                break

            # Skip blank lines (sometimes files have trailing newlines)
            line = line.strip()
            if not line:
                continue

            # Parse the JSON text on this line into a Python dict
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                # If a line is corrupted/malformed, skip it and continue
                print(f"  Warning: Skipping malformed JSON at line {i+1}")
                continue

            # Extract and flatten the candidate into a clean dict
            candidate = extract_candidate_features(raw)
            candidates.append(candidate)

    print(f"Loaded {len(candidates)} candidates successfully.")
    return candidates


# ── FEATURE EXTRACTION ─────────────────────────────────────────────────────

def extract_candidate_features(raw: dict) -> dict:
    """
    Take a raw candidate JSON dict and pull out all the fields we care about
    into a flat (non-nested) dictionary.

    Why flatten?
    → The raw data is deeply nested: raw["redrob_signals"]["last_active_date"]
    → A flat dict is much easier to work with: candidate["last_active_days_ago"]
    → We also compute some derived values here (e.g. days since last active)

    Parameters:
        raw : the raw parsed JSON for one candidate

    Returns:
        A flat dict with all the features we'll use for ranking.
    """
    # Safely get nested sections (return empty dict if missing)
    profile   = raw.get("profile", {})
    signals   = raw.get("redrob_signals", {})
    skills    = raw.get("skills", [])
    career    = raw.get("career_history", [])
    education = raw.get("education", [])

    # ── BASIC PROFILE FIELDS ──
    candidate_id      = raw.get("candidate_id", "UNKNOWN")
    years_exp         = profile.get("years_of_experience", 0)
    current_title     = profile.get("current_title", "").lower()
    current_company   = profile.get("current_company", "").lower()
    headline          = profile.get("headline", "")
    summary           = profile.get("summary", "")
    location          = profile.get("location", "").lower()
    country           = profile.get("country", "India")
    company_size      = profile.get("current_company_size", "")
    industry          = profile.get("current_industry", "")

    # ── BEHAVIORAL SIGNALS ──
    profile_complete  = signals.get("profile_completeness_score", 0)
    open_to_work      = signals.get("open_to_work_flag", False)
    response_rate     = signals.get("recruiter_response_rate", 0)
    response_time_hrs = signals.get("avg_response_time_hours", 999)
    notice_days       = signals.get("notice_period_days", 90)
    github_score      = signals.get("github_activity_score", -1)
    interview_rate    = signals.get("interview_completion_rate", 0)
    offer_accept_rate = signals.get("offer_acceptance_rate", -1)
    views_30d         = signals.get("profile_views_received_30d", 0)
    apps_30d          = signals.get("applications_submitted_30d", 0)
    saved_30d         = signals.get("saved_by_recruiters_30d", 0)
    connection_count  = signals.get("connection_count", 0)
    endorsements      = signals.get("endorsements_received", 0)
    verified_email    = signals.get("verified_email", False)
    verified_phone    = signals.get("verified_phone", False)
    linkedin          = signals.get("linkedin_connected", False)
    willing_relocate  = signals.get("willing_to_relocate", False)
    work_mode         = signals.get("preferred_work_mode", "")

    salary_range      = signals.get("expected_salary_range_inr_lpa", {})
    salary_min        = salary_range.get("min", 0)
    salary_max        = salary_range.get("max", 0)

    # ── DATE-BASED SIGNALS ──
    # How many days ago was the candidate last active?
    # A candidate active yesterday is much better than one active 6 months ago.
    last_active_days = _days_since(signals.get("last_active_date"))
    account_age_days = _days_since(signals.get("signup_date"))

    # ── SKILL ANALYSIS ──
    # Count how many JD-relevant skills this candidate has
    skill_names = {s.get("name", "").lower() for s in skills}
    jd_skill_overlap = len(skill_names & JD_CORE_SKILLS)

    # Count skills by proficiency level
    expert_skills   = sum(1 for s in skills if s.get("proficiency") == "expert")
    advanced_skills = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    total_skills    = len(skills)

    # Average duration (months) across all skills — longer = more seasoned
    skill_durations = [s.get("duration_months", 0) for s in skills]
    avg_skill_months = sum(skill_durations) / len(skill_durations) if skill_durations else 0

    # ── CAREER ANALYSIS ──
    # Has this person worked at product companies (good) vs. services firms (bad)?
    services_company_count = sum(
        1 for job in career
        if any(firm in job.get("company", "").lower() for firm in SERVICES_FIRMS)
    )
    total_companies = len(career)

    # What's the longest single stint? Job-hoppers have short stints.
    durations = [job.get("duration_months", 0) for job in career]
    max_tenure_months = max(durations) if durations else 0
    avg_tenure_months = sum(durations) / len(durations) if durations else 0

    # Build a combined text blob of all career descriptions (for later embedding)
    career_text = " ".join(
        job.get("description", "") for job in career
    )

    # ── EDUCATION ──
    edu_tiers = [e.get("tier", "unknown") for e in education]
    top_edu_tier = _best_edu_tier(edu_tiers)

    # ── HONEYPOT DETECTION ──
    # The hackathon planted ~80 "impossible" profiles in the dataset.
    # Example: "8 years experience at company founded 3 years ago"
    # If we detect one, we flag it so it never enters the top 100.
    is_honeypot = _detect_honeypot(raw, career, skills, years_exp)

    # ── LOCATION SIGNAL ──
    # The JD prefers Pune/Noida and welcomes several other Indian cities
    in_preferred_location = any(city in location for city in PREFERRED_CITIES)
    in_india = (country.lower() == "india")

    # ── TITLE RELEVANCE ──
    # Is this person's job title clearly non-AI? (The #1 flaw in the sample submission)
    is_non_ai_title = any(title in current_title for title in NON_AI_TITLES)

    # Build the full text representation for semantic embedding
    # (We combine headline + summary + career descriptions into one string)
    full_text = f"{headline} {summary} {career_text}".strip()

    # ── RETURN FLAT DICT ──
    return {
        # Identifiers
        "candidate_id":         candidate_id,
        "current_title":        profile.get("current_title", ""),
        "current_company":      profile.get("current_company", ""),
        "location":             profile.get("location", ""),
        "country":              country,

        # For generating reasoning later
        "headline":             headline,
        "summary":              summary,
        "full_text":            full_text,
        "career_text":          career_text,

        # Profile features
        "years_exp":            years_exp,
        "company_size":         company_size,
        "industry":             industry,
        "in_preferred_location": int(in_preferred_location),
        "in_india":             int(in_india),
        "willing_relocate":     int(willing_relocate),
        "work_mode":            work_mode,

        # Skill features
        "jd_skill_overlap":     jd_skill_overlap,
        "expert_skills":        expert_skills,
        "advanced_skills":      advanced_skills,
        "total_skills":         total_skills,
        "avg_skill_months":     avg_skill_months,

        # Career features
        "total_companies":      total_companies,
        "services_company_count": services_company_count,
        "max_tenure_months":    max_tenure_months,
        "avg_tenure_months":    avg_tenure_months,

        # Education
        "top_edu_tier":         top_edu_tier,

        # Behavioral signals
        "profile_complete":     profile_complete,
        "open_to_work":         int(open_to_work),
        "response_rate":        response_rate,
        "response_time_hrs":    response_time_hrs,
        "notice_days":          notice_days,
        "github_score":         github_score,
        "interview_rate":       interview_rate,
        "offer_accept_rate":    offer_accept_rate,
        "views_30d":            views_30d,
        "apps_30d":             apps_30d,
        "saved_30d":            saved_30d,
        "connection_count":     connection_count,
        "endorsements":         endorsements,
        "verified_email":       int(verified_email),
        "verified_phone":       int(verified_phone),
        "linkedin":             int(linkedin),
        "salary_min":           salary_min,
        "salary_max":           salary_max,

        # Derived date signals
        "last_active_days":     last_active_days,
        "account_age_days":     account_age_days,

        # Flags (used for filtering/penalizing)
        "is_honeypot":          is_honeypot,
        "is_non_ai_title":      int(is_non_ai_title),
    }


# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    """
    Given a date string like '2025-10-16', return how many days ago that was.
    Returns 9999 if the date is missing or can't be parsed (treated as very old).
    """
    if not date_str:
        return 9999
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return (TODAY - date).days
    except ValueError:
        return 9999


def _best_edu_tier(tiers: list[str]) -> int:
    """
    Convert education tier strings to a number.
    tier_1 = 4 (best, e.g. IIT/IIM), tier_4 = 1 (lowest), unknown = 0.

    Why convert to numbers?
    → Our ML model needs numbers, not text like "tier_1".
    """
    tier_map = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    scores = [tier_map.get(t, 0) for t in tiers]
    return max(scores) if scores else 0


def _detect_honeypot(
    raw: dict,
    career: list,
    skills: list,
    years_exp: float
) -> bool:
    """
    Detect candidates with impossible/suspicious profiles.

    The hackathon planted ~80 "honeypot" candidates to catch naive keyword-
    matching systems. Examples of honeypots:
      - Claims 8 years at a company that was founded 3 years ago
      - Claims "expert" in 10+ skills all with 0 years of use
      - Has far more years of experience than their career timeline allows

    If any check triggers, we flag this candidate as a honeypot.
    They will be excluded from our top 100.
    """
    # CHECK 1: Career timeline impossible
    # Sum up all job durations (in years). If the total is much less than
    # their stated years_of_experience, something is wrong.
    total_career_months = sum(job.get("duration_months", 0) for job in career)
    total_career_years  = total_career_months / 12

    if years_exp > 0 and total_career_years > 0:
        # If stated experience is more than 3 years beyond their career timeline
        if years_exp > total_career_years + 3:
            return True

    # CHECK 2: Too many "expert" skills with 0 months of usage
    # Real experts have used their skills for a long time.
    # A honeypot might list 10 "expert" skills all with duration_months = 0.
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 5:
        return True

    # CHECK 3: Extremely young career but very high experience
    # (More checks can be added here as we discover more patterns)

    return False  # No honeypot signals found


# ── QUICK TEST (run this file directly to verify it works) ─────────────────
if __name__ == "__main__":
    import sys
    import os

    # Default path — look for candidates.jsonl in the data/ folder
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.jsonl")

    # Allow overriding with a command-line argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    # Load just the first 5 candidates as a quick test
    print("Running quick test — loading first 5 candidates...")
    candidates = load_candidates(filepath, limit=5)

    print(f"\nLoaded {len(candidates)} candidates.")
    print("\nFirst candidate preview:")
    for key, value in candidates[0].items():
        # Truncate long text fields so the output is readable
        if isinstance(value, str) and len(value) > 80:
            value = value[:80] + "..."
        print(f"  {key:30s}: {value}")
#
# Think of this file as the "data doorman" — before any AI magic happens,
# someone has to open the file, check each record is valid, and hand the
# data to the rest of the system in a clean, consistent format.
# ─────────────────────────────────────────────────────────────────────

# TODO: Will be implemented in Step 2 (Data Loading)
