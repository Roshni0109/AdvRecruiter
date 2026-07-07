# src/normalization.py
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Union
from src.schemas import CanonicalCandidate, CanonicalJobDescription

TODAY = datetime.today()

# Common key variations for candidate mapping
CANDIDATE_ID_KEYS = ["candidate_id", "id", "user_id", "RegistrationID", "BeneficaryProfileId", "candidateId"]
YEARS_EXP_KEYS = ["years_of_experience", "years_exp", "experience", "total_experience", "exp", "yearsExp", "Ex_id"]
TITLE_KEYS = ["current_title", "title", "job_title", "role", "designation"]
COMPANY_KEYS = ["current_company", "company", "currentCompany", "employer"]
LOCATION_KEYS = ["location", "city", "address", "PA_Address", "TA_Address", "PA_District", "TA_District", "PA_Village", "TA_Village"]
COUNTRY_KEYS = ["country", "nationality", "state", "PA_State", "TA_State"]
HEADLINE_KEYS = ["headline", "summary", "objective", "bio", "about"]

def get_mapped_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Helper to look up values in nested or flat structures by checking variations of keys."""
    # First search at top level
    for k in keys:
        if k in data:
            return data[k]
    # Check common sub-structures (e.g. profile, redrob_signals, JobSeeker, JS_RegistrationDetails)
    for sub_key in ["profile", "redrob_signals", "JS_RegistrationDetails", "JS_PersonalDetail", "personal_details"]:
        sub_data = data.get(sub_key)
        if isinstance(sub_data, dict):
            for k in keys:
                if k in sub_data:
                    return sub_data[k]
    return default

def normalize_candidate(raw: Dict[str, Any]) -> CanonicalCandidate:
    """
    Ingest a raw candidate profile dict and normalize it into a CanonicalCandidate instance.
    Handles varied key names, missing fields, and nested structures.
    """
    # 1. Identify Candidate ID
    candidate_id = get_mapped_value(raw, CANDIDATE_ID_KEYS)
    if not candidate_id:
        # Generate a fallback unique ID if not present
        import uuid
        candidate_id = f"CAND_FALLBACK_{uuid.uuid4().hex[:8].upper()}"
    
    candidate_id = str(candidate_id)

    # 2. Extract profile fields
    years_exp_raw = get_mapped_value(raw, YEARS_EXP_KEYS, 0.0)
    try:
        years_exp = float(years_exp_raw)
    except (ValueError, TypeError):
        years_exp = 0.0

    current_title = get_mapped_value(raw, TITLE_KEYS, "")
    current_company = get_mapped_value(raw, COMPANY_KEYS, "")
    location = get_mapped_value(raw, LOCATION_KEYS, "")
    country = get_mapped_value(raw, COUNTRY_KEYS, "")
    
    headline = get_mapped_value(raw, ["headline"], "")
    summary = get_mapped_value(raw, ["summary", "about", "bio"], "")
    
    # 3. Extract lists
    raw_skills = raw.get("skills", raw.get("Skills", []))
    skills = []
    if isinstance(raw_skills, list):
        for sk in raw_skills:
            if isinstance(sk, dict):
                skills.append({
                    "name": sk.get("name", sk.get("skill_name", sk.get("skill", ""))),
                    "proficiency": sk.get("proficiency", sk.get("level", "intermediate")).lower(),
                    "duration_months": sk.get("duration_months", sk.get("months", 0)),
                    "endorsements": sk.get("endorsements", sk.get("endorsement_count", 0))
                })
            elif isinstance(sk, str):
                skills.append({
                    "name": sk,
                    "proficiency": "intermediate",
                    "duration_months": 0,
                    "endorsements": 0
                })
    elif isinstance(raw_skills, str):
        # comma-separated string
        for sk in re.split(r'[,;]', raw_skills):
            if sk.strip():
                skills.append({
                    "name": sk.strip(),
                    "proficiency": "intermediate",
                    "duration_months": 0,
                    "endorsements": 0
                })

    raw_career = raw.get("career_history", raw.get("career", raw.get("experience_history", [])))
    career_history = []
    if isinstance(raw_career, list):
        for job in raw_career:
            if isinstance(job, dict):
                career_history.append({
                    "company": job.get("company", job.get("employer", "")),
                    "title": job.get("title", job.get("role", "")),
                    "duration_months": job.get("duration_months", job.get("months", 0)),
                    "description": job.get("description", job.get("responsibilities", ""))
                })

    raw_education = raw.get("education", raw.get("education_history", []))
    education = []
    if isinstance(raw_education, list):
        for edu in raw_education:
            if isinstance(edu, dict):
                education.append({
                    "institution": edu.get("institution", edu.get("school", edu.get("college", ""))),
                    "degree": edu.get("degree", ""),
                    "field_of_study": edu.get("field_of_study", edu.get("major", "")),
                    "tier": edu.get("tier", "unknown").lower()
                })

    # 4. Behavioral signals (check flat or nested in redrob_signals)
    signals = raw.get("redrob_signals", raw)
    profile_complete = signals.get("profile_completeness_score", 0.0)
    
    # Try different key names for flags
    open_to_work_raw = signals.get("open_to_work_flag", signals.get("open_to_work", signals.get("IsEmployed", False)))
    open_to_work = open_to_work_raw if isinstance(open_to_work_raw, bool) else (str(open_to_work_raw).lower() in ("true", "1", "yes"))
    
    response_rate = signals.get("recruiter_response_rate", 0.0)
    response_time_hrs = signals.get("avg_response_time_hours", 999.0)
    notice_days = signals.get("notice_period_days", 90)
    github_score = signals.get("github_activity_score", -1.0)
    interview_rate = signals.get("interview_completion_rate", 0.0)
    offer_accept_rate = signals.get("offer_acceptance_rate", -1.0)
    views_30d = signals.get("profile_views_received_30d", 0)
    apps_30d = signals.get("applications_submitted_30d", 0)
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    connection_count = signals.get("connection_count", 0)
    endorsements = signals.get("endorsements_received", 0)
    
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)
    linkedin = signals.get("linkedin_connected", False)
    
    willing_relocate_raw = signals.get("willing_to_relocate", False)
    willing_relocate = willing_relocate_raw if isinstance(willing_relocate_raw, bool) else (str(willing_relocate_raw).lower() in ("true", "1", "yes"))
    
    work_mode = signals.get("preferred_work_mode", "")
    
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    if isinstance(salary_range, dict):
        salary_min = salary_range.get("min", 0.0)
        salary_max = salary_range.get("max", 0.0)
    else:
        salary_min = 0.0
        salary_max = 0.0

    # Date-based signals
    last_active_date = signals.get("last_active_date")
    signup_date = signals.get("signup_date", signals.get("RegistrationDate"))
    
    last_active_days = _days_since(last_active_date)
    account_age_days = _days_since(signup_date)

    # 5. Build Text representations for semantic search
    career_texts = []
    for job in career_history:
        desc = job.get("description", "")
        t = job.get("title", "")
        c = job.get("company", "")
        career_texts.append(f"{t} at {c}: {desc}")
    career_text = " ".join(career_texts)
    
    full_text = f"{headline} {summary} {career_text}".strip()

    return CanonicalCandidate(
        candidate_id=candidate_id,
        current_title=current_title,
        current_company=current_company,
        location=location,
        country=country,
        years_exp=years_exp,
        skills=skills,
        headline=headline,
        summary=summary,
        career_history=career_history,
        education=education,
        profile_complete=profile_complete,
        open_to_work=open_to_work,
        response_rate=response_rate,
        response_time_hrs=response_time_hrs,
        notice_days=notice_days,
        github_score=github_score,
        interview_rate=interview_rate,
        offer_accept_rate=offer_accept_rate,
        views_30d=views_30d,
        apps_30d=apps_30d,
        saved_30d=saved_30d,
        connection_count=connection_count,
        endorsements=endorsements,
        verified_email=verified_email,
        verified_phone=verified_phone,
        linkedin=linkedin,
        willing_relocate=willing_relocate,
        work_mode=work_mode,
        salary_min=salary_min,
        salary_max=salary_max,
        last_active_days=last_active_days,
        account_age_days=account_age_days,
        full_text=full_text,
        career_text=career_text
    )

def _days_since(date_str: str) -> int:
    """Helper to parse varied date strings and return days since today."""
    if not date_str:
        return 9999
    # support formats: '2025-10-16', '2025-10-16T00:00:00', etc.
    clean_date = str(date_str).split("T")[0]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            date_val = datetime.strptime(clean_date, fmt)
            return max(0, (TODAY - date_val).days)
        except ValueError:
            continue
    return 9999

def normalize_job_description(jd_text: str) -> CanonicalJobDescription:
    """
    Parse a raw job description text and extract key schema fields heuristically.
    If parsing fails/is ambiguous, uses reasonable defaults.
    """
    text_lower = jd_text.lower()
    
    # 1. Experience extraction (e.g. 5-9 years, 4+ years)
    min_exp, max_exp = 0.0, 99.0
    exp_range_match = re.search(r"(\d+)\s*[-to]+\s*(\d+)\s*years?", text_lower)
    if exp_range_match:
        min_exp = float(exp_range_match.group(1))
        max_exp = float(exp_range_match.group(2))
    else:
        exp_plus_match = re.search(r"(\d+)\+\s*years?", text_lower)
        if exp_plus_match:
            min_exp = float(exp_plus_match.group(1))
            max_exp = min_exp + 5.0  # safe assumption for range upper limit
    
    # 2. Job Title
    title = "Senior AI Engineer"
    title_match = re.search(r"(?:role|title|position):\s*([^\n]+)", text_lower)
    if title_match:
        title = title_match.group(1).strip().title()
    else:
        # Default heuristic: check first non-empty line
        first_line = [line.strip() for line in jd_text.split("\n") if line.strip()]
        if first_line:
            title = first_line[0][:60].strip()

    # 3. Core skills list extraction
    common_skills = {
        "embeddings", "sentence-transformers", "vector database", "pinecone", "weaviate", 
        "qdrant", "milvus", "faiss", "elasticsearch", "opensearch", "retrieval", "ranking", 
        "nlp", "information retrieval", "bge", "e5", "ndcg", "mrr", "map", "a/b testing", 
        "fine-tuning", "lora", "qlora", "peft", "llm", "rag", "hybrid search", "reranking", 
        "xgboost", "lightgbm", "python", "pytorch", "transformers", "fastapi", "django", 
        "sql", "postgres", "aws", "docker", "kubernetes", "go"
    }
    extracted_skills = []
    for skill in common_skills:
        # use word boundary regex
        if re.search(rf"\b{re.escape(skill)}\b", text_lower):
            extracted_skills.append(skill)
            
    if not extracted_skills:
        extracted_skills = list(common_skills) # fallback

    # 4. Preferred locations
    common_cities = {"pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "gurugram", "gurgaon"}
    extracted_cities = []
    for city in common_cities:
        if re.search(rf"\b{re.escape(city)}\b", text_lower):
            extracted_cities.append(city)
            
    if not extracted_cities:
        # Default cities from hackathon
        extracted_cities = ["pune", "noida", "bangalore", "bengaluru", "delhi ncr"]

    # 5. Weights allocation based on JD contents
    weights = {
        "semantic_similarity": 0.35,
        "title_relevance": 0.20,
        "behavioral": 0.15,
        "experience_fit": 0.10,
        "career_quality": 0.10,
        "skill_depth": 0.05,
        "location": 0.05
    }

    return CanonicalJobDescription(
        title=title,
        min_years_exp=min_exp,
        max_years_exp=max_exp,
        core_skills=extracted_skills,
        preferred_cities=extracted_cities,
        services_firms=["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra"],
        non_ai_titles=["hr manager", "graphic designer", "content writer", "accountant", "civil engineer", "mechanical engineer"],
        weights=weights,
        raw_text=jd_text
    )

def extract_text_from_pdf(pdf_path):
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def extract_text_from_docx(docx_path):
    import docx
    doc = docx.Document(docx_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_jd_file(file_path):
    import os
    if not file_path or not os.path.exists(file_path):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return extract_text_from_docx(file_path)
        else:
            return extract_text_from_txt(file_path)
    except Exception as e:
        print(f"Error parsing JD file {file_path}: {e}")
        return None

