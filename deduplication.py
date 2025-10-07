from rapidfuzz import fuzz
from typing import List, Tuple, Optional
from models import Candidate
import logging

logger = logging.getLogger(__name__)

class EntityResolver:

    @staticmethod
    def normalize_name(name: str) -> str:
        name = name.lower().strip()
        name = name.replace(".", "").replace(",", "")
        parts = name.split()
        return " ".join(sorted(parts))

    @staticmethod
    def calculate_similarity(candidate1: Candidate, candidate2: Candidate) -> float:
        scores = []

        name_score = fuzz.ratio(
            EntityResolver.normalize_name(candidate1.name),
            EntityResolver.normalize_name(candidate2.name)
        ) / 100.0
        scores.append(("name", name_score, 0.4))

        if candidate1.contacts.email and candidate2.contacts.email:
            email_match = 1.0 if candidate1.contacts.email.lower() == candidate2.contacts.email.lower() else 0.0
            scores.append(("email", email_match, 0.5))

        if candidate1.education and candidate2.education:
            edu1 = candidate1.education[0].institution.lower()
            edu2 = candidate2.education[0].institution.lower()
            edu_score = fuzz.ratio(edu1, edu2) / 100.0
            scores.append(("education", edu_score, 0.3))

        if candidate1.experience and candidate2.experience:
            exp1 = candidate1.experience[0].employer.lower()
            exp2 = candidate2.experience[0].employer.lower()
            exp_score = fuzz.ratio(exp1, exp2) / 100.0
            scores.append(("employer", exp_score, 0.2))

        if candidate1.contacts.phone and candidate2.contacts.phone:
            phone1 = ''.join(filter(str.isdigit, candidate1.contacts.phone))
            phone2 = ''.join(filter(str.isdigit, candidate2.contacts.phone))
            phone_match = 1.0 if phone1 == phone2 else 0.0
            scores.append(("phone", phone_match, 0.4))

        total_weight = sum(weight for _, _, weight in scores)
        weighted_score = sum(score * weight for _, score, weight in scores) / total_weight if total_weight > 0 else 0

        return weighted_score

    @staticmethod
    def find_duplicates(candidates: List[Candidate], threshold: float = 0.75) -> List[Tuple[str, str, float]]:
        duplicates = []

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if candidates[i].name == "Unknown" or candidates[j].name == "Unknown":
                    continue

                similarity = EntityResolver.calculate_similarity(candidates[i], candidates[j])
                if similarity >= threshold:
                    duplicates.append((
                        candidates[i].candidate_id,
                        candidates[j].candidate_id,
                        similarity
                    ))
                    logger.info(f"Potential duplicate: {candidates[i].name} <-> {candidates[j].name} (score: {similarity:.2f})")

        return duplicates

    @staticmethod
    def merge_candidates(primary: Candidate, secondary: Candidate) -> Candidate:
        if not primary.contacts.email and secondary.contacts.email:
            primary.contacts.email = secondary.contacts.email
        if not primary.contacts.phone and secondary.contacts.phone:
            primary.contacts.phone = secondary.contacts.phone
        if not primary.contacts.linkedin and secondary.contacts.linkedin:
            primary.contacts.linkedin = secondary.contacts.linkedin

        existing_schools = {e.institution for e in primary.education}
        for edu in secondary.education:
            if edu.institution not in existing_schools:
                primary.education.append(edu)

        existing_employers = {e.employer for e in primary.experience}
        for exp in secondary.experience:
            if exp.employer not in existing_employers:
                primary.experience.append(exp)

        existing_skills = {s.skill for s in primary.skills}
        for skill in secondary.skills:
            if skill.skill not in existing_skills:
                primary.skills.append(skill)

        primary.meta.parse_confidence = max(primary.meta.parse_confidence, secondary.meta.parse_confidence)

        return primary

class OntologyMapper:

    @staticmethod
    def map_strategy(text: str) -> List[str]:
        from config import STRATEGIES
        text_lower = text.lower()
        mapped = []

        strategy_keywords = {
            "Fundamental": ["fundamental", "discretionary", "long short equity", "long/short", "value", "growth"],
            "Systematic": ["systematic", "quant", "quantitative", "algorithmic", "statistical arbitrage", "stat arb"],
            "Credit": ["credit", "distressed", "high yield", "corporate bonds"],
            "Macro": ["macro", "global macro", "currency", "commodities", "rates"]
        }

        for strategy, keywords in strategy_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mapped.append(strategy)

        return list(set(mapped))

    @staticmethod
    def map_sectors(text: str) -> List[str]:
        from config import SECTORS
        text_lower = text.lower()
        mapped = []

        for sector in SECTORS:
            if sector.lower() in text_lower:
                mapped.append(sector)

        sector_aliases = {
            "tech": "Technology",
            "healthcare": "Healthcare",
            "biotech": "Biotechnology",
            "pharma": "Pharmaceuticals",
            "semis": "Semiconductors",
            "software": "Software",
            "banks": "Banks",
            "insurance": "Insurance"
        }

        for alias, sector in sector_aliases.items():
            if alias in text_lower and sector not in mapped:
                mapped.append(sector)

        return list(set(mapped))

    @staticmethod
    def normalize_skill(skill: str) -> str:
        from config import SKILLS_TAXONOMY

        skill_lower = skill.lower().strip()

        skill_mapping = {
            "py": "Python",
            "cpp": "C++",
            "c++": "C++",
            "ml": "Machine Learning",
            "ai": "Machine Learning",
            "nlp": "NLP",
            "cv": "Computer Vision",
            "nn": "Deep Learning",
            "neural networks": "Deep Learning"
        }

        if skill_lower in skill_mapping:
            return skill_mapping[skill_lower]

        for family, skills in SKILLS_TAXONOMY.items():
            for known_skill in skills:
                if fuzz.ratio(skill_lower, known_skill.lower()) > 85:
                    return known_skill

        return skill.title()
