# src/schemas.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class CanonicalCandidate:
    candidate_id: str
    current_title: str = ""
    current_company: str = ""
    location: str = ""
    country: str = ""
    years_exp: float = 0.0
    skills: List[Dict[str, Any]] = field(default_factory=list)
    headline: str = ""
    summary: str = ""
    career_history: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    
    # Behavioral and engagement signals
    profile_complete: float = 0.0
    open_to_work: bool = False
    response_rate: float = 0.0
    response_time_hrs: float = 999.0
    notice_days: int = 90
    github_score: float = -1.0
    interview_rate: float = 0.0
    offer_accept_rate: float = -1.0
    views_30d: int = 0
    apps_30d: int = 0
    saved_30d: int = 0
    connection_count: int = 0
    endorsements: int = 0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin: bool = False
    willing_relocate: bool = False
    work_mode: str = ""
    salary_min: float = 0.0
    salary_max: float = 0.0
    
    # Derived signals
    last_active_days: int = 9999
    account_age_days: int = 9999
    
    # Rule/Filter flags
    is_honeypot: bool = False
    is_non_ai_title: bool = False
    
    # Text representations for embedding
    full_text: str = ""
    career_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CanonicalCandidate instance into a dictionary."""
        d = self.__dict__.copy()
        # Convert booleans to int for modeling compatibility if needed
        d["open_to_work"] = int(self.open_to_work)
        d["verified_email"] = int(self.verified_email)
        d["verified_phone"] = int(self.verified_phone)
        d["linkedin"] = int(self.linkedin)
        d["willing_relocate"] = int(self.willing_relocate)
        d["is_honeypot"] = int(self.is_honeypot)
        d["is_non_ai_title"] = int(self.is_non_ai_title)
        return d

@dataclass
class CanonicalJobDescription:
    title: str = ""
    min_years_exp: float = 0.0
    max_years_exp: float = 99.0
    core_skills: List[str] = field(default_factory=list)
    preferred_cities: List[str] = field(default_factory=list)
    services_firms: List[str] = field(default_factory=list)
    non_ai_titles: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CanonicalJobDescription instance into a dictionary."""
        return self.__dict__.copy()
