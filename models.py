from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, date
from enum import Enum
import uuid
import re

class ContactInfo(BaseModel):
    email: Optional[str] = None
    email_is_valid: bool = True
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v is None:
            return v

        partial_email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+(?:\.[A-Za-z]{2,})?$'
        if not re.match(partial_email_pattern, v):
            return None

        return v

class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    major: Optional[str] = None
    gpa: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    honors: Optional[List[str]] = Field(default_factory=list)
    awards: Optional[List[str]] = Field(default_factory=list)

    @field_validator('gpa')
    @classmethod
    def validate_gpa(cls, v):
        if v is not None and (v < 0 or v > 4.5):
            return None
        return v

class Experience(BaseModel):
    employer: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    asset_class: Optional[List[str]] = Field(default_factory=list)
    strategy: Optional[List[str]] = Field(default_factory=list)
    sector_coverage: Optional[List[str]] = Field(default_factory=list)
    products: Optional[List[str]] = Field(default_factory=list)
    is_current: bool = False

class Skill(BaseModel):
    skill: str
    family: Optional[str] = None
    level: Optional[str] = None
    evidence_span: Optional[str] = None

class Publication(BaseModel):
    title: str
    venue: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None

class Project(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: Optional[List[str]] = Field(default_factory=list)
    url: Optional[str] = None

class ParseMetadata(BaseModel):
    source_file: str
    parse_version: str = "1.0.0"
    parse_confidence: float
    ingestion_ts: datetime = Field(default_factory=datetime.now)
    parse_errors: List[str] = Field(default_factory=list)
    token_usage: Optional[int] = None
    processing_time_ms: Optional[float] = None

class Candidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: f"cand_{uuid.uuid4().hex[:12]}")
    name: str
    contacts: ContactInfo = Field(default_factory=ContactInfo)
    location: Optional[str] = None
    work_auth: Optional[str] = None
    years_experience_band: Optional[str] = None
    availability_date: Optional[str] = None
    comp_expectations_band: Optional[str] = None

    strategy: List[str] = Field(default_factory=list)
    asset_classes: List[str] = Field(default_factory=list)
    sector_coverage: List[str] = Field(default_factory=list)

    skills: List[Skill] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)

    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    awards: List[Dict[str, Any]] = Field(default_factory=list)

    meta: ParseMetadata

    resume_text: Optional[str] = None
    embedding: Optional[List[float]] = None

    def get_programming_skills(self) -> List[str]:
        prog_skills = []
        for skill in self.skills:
            if skill.family == "Programming":
                prog_skills.append(skill.skill)
        return prog_skills

    def get_school_tier(self) -> Optional[str]:
        from config import SCHOOL_TIERS
        for edu in self.education:
            for tier, schools in SCHOOL_TIERS.items():
                if any(school.lower() in edu.institution.lower() for school in schools):
                    return tier
        return None

    def get_total_experience_years(self) -> Optional[float]:
        if self.years_experience_band:
            band = self.years_experience_band
            if band == "Intern":
                return 0.0
            elif band == "New Grad":
                return 0.5
            elif "-" in band:
                parts = band.split("-")
                try:
                    return (float(parts[0]) + float(parts[1])) / 2
                except:
                    return None
            elif "+" in band:
                try:
                    return float(band.replace("+", ""))
                except:
                    return None
        return None

    def get_all_awards(self) -> List[str]:
        all_awards = []
        for edu in self.education:
            if edu.awards:
                all_awards.extend(edu.awards)
        for award in self.awards:
            if isinstance(award, dict):
                all_awards.append(award.get("title", ""))
            else:
                all_awards.append(str(award))
        return all_awards

    def calculate_career_progression_score(self) -> Dict[str, Any]:
        from dateutil.parser import parse
        from dateutil.relativedelta import relativedelta
        from datetime import datetime

        score_components = {
            "progression_velocity": 0.0,
            "pedigree_score": 0.0,
            "specialization_score": 0.0,
            "high_potential_indicators": [],
            "entry_level": "Unknown",
            "total_score": 0.0
        }

        # 1. Pedigree Score (0-40 points)
        pedigree = 0

        # Top school (20 points)
        school_tier = self.get_school_tier()
        if school_tier == "Tier 1":
            pedigree += 20
            score_components["high_potential_indicators"].append("Tier 1 Education")
        elif school_tier == "Tier 2":
            pedigree += 12

        # High GPA (10 points)
        if self.education:
            max_gpa = max([e.gpa for e in self.education if e.gpa], default=0)
            if max_gpa >= 3.8:
                pedigree += 10
                score_components["high_potential_indicators"].append(f"High GPA ({max_gpa:.2f})")
            elif max_gpa >= 3.5:
                pedigree += 6

        # Awards (10 points)
        awards = self.get_all_awards()
        if len(awards) >= 3:
            pedigree += 10
            score_components["high_potential_indicators"].append(f"{len(awards)} Awards/Honors")
        elif len(awards) >= 1:
            pedigree += 5

        score_components["pedigree_score"] = min(pedigree, 40)

        # 2. Progression Velocity (0-35 points)
        velocity = 0

        if len(self.experience) >= 2:
            # Sort by date
            sorted_exp = sorted(
                self.experience,
                key=lambda x: parse(x.start_date) if x.start_date else datetime.min,
                reverse=False
            )

            # Analyze title progression
            titles = [exp.title.lower() for exp in sorted_exp]
            promotions = 0

            title_seniority = {
                "intern": 1, "analyst": 2, "associate": 3, "senior": 4,
                "vp": 5, "vice president": 5, "director": 6, "managing director": 7,
                "partner": 8, "principal": 8, "manager": 4, "lead": 4
            }

            for i in range(1, len(titles)):
                prev_level = max([v for k, v in title_seniority.items() if k in titles[i-1]], default=0)
                curr_level = max([v for k, v in title_seniority.items() if k in titles[i]], default=0)

                if curr_level > prev_level:
                    promotions += 1

            if promotions >= 2:
                velocity += 15
                score_components["high_potential_indicators"].append(f"{promotions} Promotions")
            elif promotions == 1:
                velocity += 8

            # Company tier progression (startup → BB → HF)
            company_tiers = []
            prestigious_firms = [
                "goldman", "morgan stanley", "jp morgan", "blackstone", "citadel",
                "millennium", "two sigma", "renaissance", "bridgewater", "de shaw",
                "jane street", "hudson river", "point72"
            ]

            for exp in sorted_exp:
                employer_lower = exp.employer.lower()
                if any(firm in employer_lower for firm in prestigious_firms):
                    company_tiers.append("prestigious")
                else:
                    company_tiers.append("other")

            if "prestigious" in company_tiers:
                velocity += 10
                score_components["high_potential_indicators"].append("Prestigious Firm Experience")

            # Time efficiency (years to current level)
            total_years = self.get_total_experience_years() or 0
            if total_years < 3 and any(word in titles[-1] for word in ["senior", "lead", "vp"]):
                velocity += 10
                score_components["high_potential_indicators"].append("Rapid Career Acceleration")

        score_components["progression_velocity"] = min(velocity, 35)

        # 3. Specialization Score (0-25 points)
        specialization = 0

        # Sector depth
        unique_sectors = set()
        for exp in self.experience:
            unique_sectors.update(exp.sector_coverage)

        if len(unique_sectors) == 1:
            specialization += 15
            score_components["high_potential_indicators"].append(f"Deep Sector Specialist: {list(unique_sectors)[0]}")
        elif len(unique_sectors) == 2:
            specialization += 10
        elif len(unique_sectors) >= 3:
            specialization += 5
            score_components["high_potential_indicators"].append(f"Versatile: {len(unique_sectors)} Sectors")

        # Publications (research depth)
        if len(self.publications) >= 2:
            specialization += 10
            score_components["high_potential_indicators"].append(f"{len(self.publications)} Publications")
        elif len(self.publications) == 1:
            specialization += 5

        score_components["specialization_score"] = min(specialization, 25)

        # 4. Determine entry level
        if self.experience:
            first_title = self.experience[-1].title.lower() if self.experience else ""
            if "intern" in first_title:
                score_components["entry_level"] = "Intern"
            elif "analyst" in first_title and "senior" not in first_title:
                score_components["entry_level"] = "Analyst"
            elif any(word in first_title for word in ["associate", "senior"]):
                score_components["entry_level"] = "Associate+"
            elif any(word in first_title for word in ["vp", "director", "manager"]):
                score_components["entry_level"] = "VP/Director"
            else:
                score_components["entry_level"] = "Unknown"

        # Total score
        score_components["total_score"] = (
            score_components["pedigree_score"] +
            score_components["progression_velocity"] +
            score_components["specialization_score"]
        )

        return score_components

    def matches_filters(self, filters: Dict[str, Any]) -> bool:
        if filters.get("strategy") and not any(s in self.strategy for s in filters["strategy"]):
            return False

        if filters.get("asset_class") and not any(a in self.asset_classes for a in filters["asset_class"]):
            return False

        if filters.get("sector") and not any(s in self.sector_coverage for s in filters["sector"]):
            return False

        if filters.get("work_auth") and self.work_auth != filters["work_auth"]:
            return False

        if filters.get("min_gpa"):
            max_gpa = max([e.gpa for e in self.education if e.gpa], default=0)
            if max_gpa < filters["min_gpa"]:
                return False

        if filters.get("programming_langs"):
            prog_skills = self.get_programming_skills()
            if not any(lang in prog_skills for lang in filters["programming_langs"]):
                return False

        if filters.get("school_tier"):
            if self.get_school_tier() != filters["school_tier"]:
                return False

        # Priority filter for top tier schools
        if filters.get("prioritize_top_schools") and filters.get("school_tier"):
            tier = self.get_school_tier()
            if tier and tier != filters["school_tier"]:
                return False

        if filters.get("experience_band") and self.years_experience_band != filters["experience_band"]:
            return False

        if filters.get("location") and filters["location"].lower() not in (self.location or "").lower():
            return False

        # Filter by awards
        if filters.get("has_awards"):
            awards = self.get_all_awards()
            if not awards:
                return False

        if filters.get("award_keywords"):
            awards = self.get_all_awards()
            awards_lower = " ".join(awards).lower()
            if not any(keyword.lower() in awards_lower for keyword in filters["award_keywords"]):
                return False

        return True

    def to_dict_with_pii_control(self, include_pii: bool = False) -> Dict[str, Any]:
        data = self.model_dump()
        if not include_pii:
            data["contacts"] = {
                "email": "***REDACTED***" if self.contacts.email else None,
                "phone": "***REDACTED***" if self.contacts.phone else None,
                "linkedin": "***REDACTED***" if self.contacts.linkedin else None,
                "website": None
            }
        return data

class SearchQuery(BaseModel):
    text_query: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = 50
    include_pii: bool = False

class SearchResult(BaseModel):
    candidate: Candidate
    score: float
    match_highlights: List[str] = Field(default_factory=list)

class CandidateDatabase(BaseModel):
    candidates: List[Candidate] = Field(default_factory=list)
    embeddings: Optional[Any] = None
    last_updated: datetime = Field(default_factory=datetime.now)

    def add_candidate(self, candidate: Candidate):
        existing_ids = [c.candidate_id for c in self.candidates]
        if candidate.candidate_id not in existing_ids:
            self.candidates.append(candidate)
            self.last_updated = datetime.now()

    def get_by_id(self, candidate_id: str) -> Optional[Candidate]:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def get_all(self) -> List[Candidate]:
        return self.candidates

    def filter_candidates(self, filters: Dict[str, Any]) -> List[Candidate]:
        results = []
        for candidate in self.candidates:
            if candidate.matches_filters(filters):
                results.append(candidate)
        return results
