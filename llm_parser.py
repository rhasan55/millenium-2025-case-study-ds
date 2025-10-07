import json
import openai
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import re
import config
from models import Candidate, ContactInfo, Education, Experience, Skill, Publication, Project, ParseMetadata
from extraction import PIIExtractor

logger = logging.getLogger(__name__)

class ResumeParser:

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.OPENAI_MODEL
        self.fast_model = "gpt-4o-mini"  # Fast initial parse
        self.pro_model = "o3-mini"  # Escalate complex resumes
        self.confidence_threshold = 0.6  # Threshold for escalation

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "location": {"type": "string"},
                "work_auth": {
                    "type": "string",
                    "enum": config.WORK_AUTH_OPTIONS + ["unknown"]
                },
                "years_experience_band": {
                    "type": "string",
                    "enum": config.EXPERIENCE_BANDS + ["unknown"]
                },
                "strategy": {
                    "type": "array",
                    "items": {"type": "string", "enum": config.STRATEGIES}
                },
                "asset_classes": {
                    "type": "array",
                    "items": {"type": "string", "enum": config.ASSET_CLASSES}
                },
                "sector_coverage": {
                    "type": "array",
                    "items": {"type": "string", "enum": config.SECTORS}
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "institution": {"type": "string"},
                            "degree": {"type": "string"},
                            "major": {"type": "string"},
                            "gpa": {"type": "number"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "honors": {"type": "array", "items": {"type": "string"}},
                            "awards": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["institution"]
                    }
                },
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employer": {"type": "string"},
                            "title": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "description": {"type": "string"},
                            "location": {"type": "string"},
                            "asset_class": {"type": "array", "items": {"type": "string"}},
                            "strategy": {"type": "array", "items": {"type": "string"}},
                            "sector_coverage": {"type": "array", "items": {"type": "string"}},
                            "products": {"type": "array", "items": {"type": "string"}},
                            "is_current": {"type": "boolean"}
                        },
                        "required": ["employer", "title"]
                    }
                },
                "skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill": {"type": "string"},
                            "family": {"type": "string"},
                            "level": {"type": "string"}
                        },
                        "required": ["skill"]
                    }
                },
                "publications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "venue": {"type": "string"},
                            "date": {"type": "string"}
                        },
                        "required": ["title"]
                    }
                },
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "technologies": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["name"]
                    }
                },
                "certifications": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "awards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "issuer": {"type": "string"},
                            "date": {"type": "string"}
                        },
                        "required": ["title"]
                    }
                },
                "parse_confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["name"]
        }

    def _build_prompt(self, resume_text: str) -> str:
        all_skills = []
        for family, skills in config.SKILLS_TAXONOMY.items():
            all_skills.extend(skills)

        prompt = f"""You are extracting structured data from a finance resume. Output ONLY valid JSON.

CRITICAL REQUIREMENTS:
1. "name" field is REQUIRED - extract the candidate's full name from the resume
2. For experience entries, use "employer" (not "company") and "title" as required fields
3. Extract ALL phone numbers (including international formats, with/without country codes)
4. Handle multiple degrees per institution as separate education entries
5. Extract current job location from the most recent experience entry
6. Calculate years_experience_band by analyzing date ranges in experience section
7. Extract all awards and honors (academic, professional, competition wins)
8. Skills should be objects with "skill" field, or simple strings
9. Use exact field names from the schema

NORMALIZATION RULES:
- Strategies: {', '.join(config.STRATEGIES)}
- Asset Classes: {', '.join(config.ASSET_CLASSES)}
- Sectors: {', '.join(config.SECTORS[:20])}
- Skills: {', '.join(all_skills[:40])}
- Experience bands: {', '.join(config.EXPERIENCE_BANDS)}
- Work auth: {', '.join(config.WORK_AUTH_OPTIONS[:4])} or null
- Dates: YYYY-MM format
- GPA: 0-4.0 scale or null

FIELD MAPPING:
- Experience: {{"employer": "Company Name", "title": "Job Title", "start_date": "YYYY-MM", "end_date": "YYYY-MM" or null, "location": "City, State/Country", "sector_coverage": [...], "strategy": [...], "asset_class": [...], "is_current": true/false}}
- Education: {{"institution": "University", "degree": "BS/MS/MBA/PhD", "major": "Field", "gpa": 3.5, "awards": ["Dean's List", "Summa Cum Laude"], "honors": [...]}}
- Skills: {{"skill": "Python", "family": "Programming", "level": "Advanced"}} OR just "Python"
- Awards: {{"title": "Award Name", "issuer": "Organization", "date": "YYYY-MM"}}
- parse_confidence: 0.0-1.0 based on resume quality and completeness

EDGE CASES TO HANDLE:
- Multiple degrees from same institution: create separate education entries
- Missing skills section: extract skills from experience descriptions
- Phone numbers: extract all formats (US: (xxx) xxx-xxxx, +1-xxx-xxx-xxxx, international: +xx xxx xxx xxxx)
- Current position: set is_current=true and location field for most recent job
- Experience calculation: analyze all date ranges to determine years_experience_band

Resume text:
{resume_text[:8000]}
"""
        return prompt

    def parse_resume(self, resume_text: str, source_file: str) -> Optional[Candidate]:
        start_time = datetime.now()

        try:
            pii = PIIExtractor.extract_all_pii(resume_text)

            prompt = self._build_prompt(resume_text)
            schema = self._build_schema()

            # Step 1: Fast parse with gpt-4o-mini
            logger.info(f"Parsing {source_file} with {self.fast_model}")
            response = self.client.chat.completions.create(
                model=self.fast_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise information extractor. Output only valid JSON conforming to the schema. Never hallucinate fields."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            parsed_data = json.loads(result_text)

            # Step 2: Check confidence and escalate if needed
            confidence = parsed_data.get("parse_confidence", 0.5)
            token_usage = response.usage.total_tokens

            if confidence < self.confidence_threshold:
                logger.warning(f"Low confidence ({confidence:.2f}) for {source_file}, escalating to {self.pro_model}")
                try:
                    response = self.client.chat.completions.create(
                        model=self.pro_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a precise information extractor. Output only valid JSON conforming to the schema. Never hallucinate fields."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=config.TEMPERATURE,
                        max_tokens=config.MAX_TOKENS,
                        response_format={"type": "json_object"}
                    )
                    result_text = response.choices[0].message.content
                    parsed_data = json.loads(result_text)
                    token_usage = response.usage.total_tokens
                    logger.info(f"Escalated parse completed for {source_file}")
                except Exception as e:
                    logger.error(f"Pro model escalation failed: {e}, using initial parse")
                    pass

            logger.info(f"Parsed data keys: {parsed_data.keys()}")
            logger.info(f"Name extracted: {parsed_data.get('name', 'MISSING')}")

            email_value = pii["emails"][0] if pii["emails"] else None
            email_valid = PIIExtractor.validate_email(email_value) if email_value else True

            contacts = ContactInfo(
                email=email_value,
                email_is_valid=email_valid,
                phone=pii["phones"][0] if pii["phones"] else None,
                linkedin=pii["linkedin"]
            )

            if not parsed_data.get("name") or parsed_data.get("name") == "Unknown":
                name_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', resume_text, re.MULTILINE)
                if name_match:
                    parsed_data["name"] = name_match.group(1)
                    logger.info(f"Extracted name from text: {parsed_data['name']}")
                elif pii["emails"]:
                    email_name = pii["emails"][0].split("@")[0].replace(".", " ").title()
                    parsed_data["name"] = email_name
                    logger.info(f"Derived name from email: {parsed_data['name']}")

            education_list = []
            for edu in parsed_data.get("education", []):
                try:
                    # Extract awards from education honors if present
                    if "awards" not in edu and "honors" in edu:
                        edu["awards"] = edu.get("honors", [])
                    education_list.append(Education(**edu))
                except Exception as e:
                    logger.warning(f"Skipping education entry: {e}")
                    continue

            experience_list = []
            for exp in parsed_data.get("experience", []):
                if "company" in exp and "employer" not in exp:
                    exp["employer"] = exp.pop("company")
                if "employer" in exp:
                    experience_list.append(Experience(**exp))

            # Calculate experience from date ranges if not provided
            if not parsed_data.get("years_experience_band") or parsed_data.get("years_experience_band") == "unknown":
                parsed_data["years_experience_band"] = self._calculate_experience_band(experience_list)

            # Extract current job location if not provided at top level
            if not parsed_data.get("location") and experience_list:
                current_job = next((exp for exp in experience_list if exp.is_current), experience_list[0] if experience_list else None)
                if current_job and hasattr(current_job, 'location') and current_job.location:
                    parsed_data["location"] = current_job.location

            skills_list = []
            for skill_data in parsed_data.get("skills", []):
                if isinstance(skill_data, str):
                    skill_obj = Skill(skill=skill_data)
                elif isinstance(skill_data, dict):
                    skill_obj = Skill(**skill_data)
                else:
                    continue
                if not skill_obj.family:
                    skill_obj.family = self._infer_skill_family(skill_obj.skill)
                skills_list.append(skill_obj)

            publications_list = []
            for pub in parsed_data.get("publications", []):
                try:
                    publications_list.append(Publication(**pub))
                except:
                    continue

            projects_list = []
            for proj in parsed_data.get("projects", []):
                if "title" in proj and "name" not in proj:
                    proj["name"] = proj.pop("title")
                if "name" in proj:
                    try:
                        projects_list.append(Project(**proj))
                    except:
                        continue

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            metadata = ParseMetadata(
                source_file=source_file,
                parse_confidence=parsed_data.get("parse_confidence", 0.5),
                token_usage=response.usage.total_tokens,
                processing_time_ms=processing_time
            )

            languages_list = []
            for lang in parsed_data.get("languages", []):
                if isinstance(lang, str):
                    languages_list.append(lang)
                elif isinstance(lang, dict) and "language" in lang:
                    languages_list.append(lang["language"])

            certifications_list = []
            for cert in parsed_data.get("certifications", []):
                if isinstance(cert, str):
                    certifications_list.append(cert)
                elif isinstance(cert, dict) and "name" in cert:
                    certifications_list.append(cert["name"])

            # Extract awards
            awards_list = []
            for award in parsed_data.get("awards", []):
                if isinstance(award, dict):
                    awards_list.append(award)
                elif isinstance(award, str):
                    awards_list.append({"title": award})

            candidate = Candidate(
                name=parsed_data.get("name", "Unknown"),
                contacts=contacts,
                location=parsed_data.get("location"),
                work_auth=parsed_data.get("work_auth") if parsed_data.get("work_auth") != "unknown" else None,
                years_experience_band=parsed_data.get("years_experience_band") if parsed_data.get("years_experience_band") != "unknown" else None,
                strategy=parsed_data.get("strategy", []),
                asset_classes=parsed_data.get("asset_classes", []),
                sector_coverage=parsed_data.get("sector_coverage", []),
                education=education_list,
                experience=experience_list,
                skills=skills_list,
                publications=publications_list,
                projects=projects_list,
                certifications=certifications_list,
                languages=languages_list,
                awards=awards_list,
                meta=metadata,
                resume_text=resume_text
            )

            return candidate

        except Exception as e:
            logger.error(f"Resume parsing failed for {source_file}: {e}")
            return None

    def _infer_skill_family(self, skill: str) -> str:
        skill_lower = skill.lower()
        for family, skills in config.SKILLS_TAXONOMY.items():
            for known_skill in skills:
                if known_skill.lower() in skill_lower or skill_lower in known_skill.lower():
                    return family
        return "Other"

    def _calculate_experience_band(self, experience_list: list) -> str:
        """Calculate experience band from date ranges in experience entries"""
        from dateutil.parser import parse
        from dateutil.relativedelta import relativedelta

        total_months = 0
        for exp in experience_list:
            try:
                if exp.start_date:
                    start = parse(exp.start_date)
                    if exp.end_date and exp.end_date.lower() not in ["present", "current"]:
                        end = parse(exp.end_date)
                    else:
                        end = datetime.now()

                    delta = relativedelta(end, start)
                    months = delta.years * 12 + delta.months
                    total_months += months
            except Exception as e:
                logger.warning(f"Error calculating experience duration: {e}")
                continue

        years = total_months / 12

        if years < 0.5:
            return "Intern"
        elif years < 1:
            return "New Grad"
        elif years < 3:
            return "1-2"
        elif years < 6:
            return "3-5"
        elif years < 9:
            return "6-8"
        elif years < 13:
            return "9-12"
        else:
            return "13+"
