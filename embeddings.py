from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Optional
from models import Candidate
import logging
import openai
import os

logger = logging.getLogger(__name__)

class EmbeddingEngine:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = True):
        self.use_openai = use_openai
        if use_openai:
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "text-embedding-3-large"
            self.dimension = 3072
        else:
            self.model = SentenceTransformer(model_name)
            self.dimension = 384

    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            if self.use_openai:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                embedding = np.array(response.data[0].embedding)
                return embedding
            else:
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(self.dimension)

    def generate_candidate_embedding(self, candidate: Candidate) -> np.ndarray:
        text_parts = []

        text_parts.append(f"Name: {candidate.name}")

        if candidate.location:
            text_parts.append(f"Location: {candidate.location}")

        for edu in candidate.education:
            text_parts.append(f"Education: {edu.degree} {edu.major} from {edu.institution}")

        for exp in candidate.experience:
            exp_text = f"Experience: {exp.title} at {exp.employer}"
            if exp.description:
                exp_text += f" - {exp.description[:200]}"
            if exp.sector_coverage:
                exp_text += f" Sectors: {', '.join(exp.sector_coverage)}"
            text_parts.append(exp_text)

        if candidate.skills:
            skills_text = "Skills: " + ", ".join([s.skill for s in candidate.skills])
            text_parts.append(skills_text)

        if candidate.sector_coverage:
            text_parts.append(f"Sector Coverage: {', '.join(candidate.sector_coverage)}")

        if candidate.strategy:
            text_parts.append(f"Investment Strategy: {', '.join(candidate.strategy)}")

        combined_text = " | ".join(text_parts)
        return self.generate_embedding(combined_text)

    def batch_generate_embeddings(self, candidates: List[Candidate]) -> np.ndarray:
        embeddings = []
        for candidate in candidates:
            embedding = self.generate_candidate_embedding(candidate)
            candidate.embedding = embedding.tolist()
            embeddings.append(embedding)

        return np.array(embeddings)

    def semantic_search(
        self,
        query: str,
        candidate_embeddings: np.ndarray,
        candidates: List[Candidate],
        top_k: int = 50
    ) -> List[Tuple[Candidate, float]]:
        query_embedding = self.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            if idx < len(candidates):
                results.append((candidates[idx], float(similarities[idx])))

        return results

class BM25Scorer:

    @staticmethod
    def tokenize(text: str) -> List[str]:
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    @staticmethod
    def compute_bm25(query: str, candidate: Candidate, k1: float = 1.5, b: float = 0.75) -> float:
        doc_text = ""

        if candidate.resume_text:
            doc_text = candidate.resume_text
        else:
            parts = []
            parts.append(candidate.name)
            for edu in candidate.education:
                parts.append(f"{edu.institution} {edu.degree or ''} {edu.major or ''}")
            for exp in candidate.experience:
                parts.append(f"{exp.employer} {exp.title} {exp.description or ''}")
            for skill in candidate.skills:
                parts.append(skill.skill)
            doc_text = " ".join(parts)

        query_tokens = BM25Scorer.tokenize(query)
        doc_tokens = BM25Scorer.tokenize(doc_text)

        if not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        avg_doc_len = 500

        score = 0.0
        for term in query_tokens:
            tf = doc_tokens.count(term)
            if tf == 0:
                continue

            idf = 1.0
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    @staticmethod
    def rank_candidates(query: str, candidates: List[Candidate], top_k: int = 50) -> List[Tuple[Candidate, float]]:
        scores = []
        for candidate in candidates:
            score = BM25Scorer.compute_bm25(query, candidate)
            scores.append((candidate, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class HybridSearchEngine:

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.rerank_model = "gpt-4o-mini"
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _rerank_with_llm(self, query: str, candidates: List[Candidate], top_n: int = 10) -> List[Tuple[Candidate, float]]:
        rerank_input = []
        for i, candidate in enumerate(candidates[:top_n]):
            candidate_summary = f"""
            Name: {candidate.name}
            Education: {', '.join([f"{e.degree} from {e.institution}" for e in candidate.education])}
            Experience: {', '.join([f"{exp.title} at {exp.employer}" for exp in candidate.experience[:3]])}
            Skills: {', '.join([s.skill for s in candidate.skills[:10]])}
            """
            rerank_input.append({
                "id": i,
                "candidate_id": candidate.candidate_id,
                "summary": candidate_summary.strip()
            })

        prompt = f"""Given the search query: "{query}"

Rate how well each candidate matches the query on a scale of 0-10.
Return a JSON object with candidate IDs as keys and scores as values.

Candidates:
{chr(10).join([f"ID {item['id']}: {item['summary']}" for item in rerank_input])}

Output format: {{"0": 8.5, "1": 7.2, ...}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.rerank_model,
                messages=[
                    {"role": "system", "content": "You are a candidate relevance scorer. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            scores_dict = eval(response.choices[0].message.content)
            reranked = []
            for item in rerank_input:
                idx = item["id"]
                score = float(scores_dict.get(str(idx), 5.0)) / 10.0
                candidate = candidates[idx]
                reranked.append((candidate, score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(c, 0.5) for c in candidates[:top_n]]

    def search(
        self,
        query: Optional[str],
        candidates: List[Candidate],
        candidate_embeddings: Optional[np.ndarray] = None,
        filters: dict = None,
        top_k: int = 50,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_reranking: bool = True
    ) -> List[Tuple[Candidate, float, List[str]]]:

        filtered_candidates = candidates
        if filters:
            filtered_candidates = [c for c in candidates if c.matches_filters(filters)]

        if not query or not query.strip():
            results = [(c, 1.0, []) for c in filtered_candidates[:top_k]]
            return results

        if candidate_embeddings is None:
            candidate_embeddings = self.embedding_engine.batch_generate_embeddings(filtered_candidates)
        else:
            filtered_indices = [candidates.index(c) for c in filtered_candidates if c in candidates]
            if filtered_indices:
                candidate_embeddings = candidate_embeddings[filtered_indices]

        semantic_results = self.embedding_engine.semantic_search(
            query, candidate_embeddings, filtered_candidates, top_k=top_k
        )

        bm25_results = BM25Scorer.rank_candidates(query, filtered_candidates, top_k=top_k)

        combined_scores = {}
        for candidate, score in semantic_results:
            combined_scores[candidate.candidate_id] = semantic_weight * score

        for candidate, score in bm25_results:
            if candidate.candidate_id in combined_scores:
                combined_scores[candidate.candidate_id] += bm25_weight * score
            else:
                combined_scores[candidate.candidate_id] = bm25_weight * score

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        initial_results = []
        for candidate_id, score in ranked:
            candidate = next((c for c in filtered_candidates if c.candidate_id == candidate_id), None)
            if candidate:
                initial_results.append(candidate)

        if use_reranking and query and len(initial_results) >= 10:
            logger.info(f"Reranking top 10 results with {self.rerank_model}")
            reranked = self._rerank_with_llm(query, initial_results, top_n=10)

            final_results = []
            for candidate, score in reranked:
                highlights = self._extract_highlights(query, candidate)
                final_results.append((candidate, score, highlights))

            for candidate in initial_results[10:]:
                highlights = self._extract_highlights(query, candidate)
                original_score = combined_scores.get(candidate.candidate_id, 0.5)
                final_results.append((candidate, original_score, highlights))
        else:
            final_results = []
            for candidate in initial_results:
                highlights = self._extract_highlights(query, candidate)
                score = combined_scores.get(candidate.candidate_id, 0.5)
                final_results.append((candidate, score, highlights))

        return final_results

    def _extract_highlights(self, query: str, candidate: Candidate) -> List[str]:
        highlights = []
        query_lower = query.lower()

        for skill in candidate.skills:
            if query_lower in skill.skill.lower():
                highlights.append(f"Skill: {skill.skill}")

        for sector in candidate.sector_coverage:
            if query_lower in sector.lower():
                highlights.append(f"Sector: {sector}")

        for exp in candidate.experience:
            if query_lower in exp.title.lower() or query_lower in exp.employer.lower():
                highlights.append(f"{exp.title} at {exp.employer}")

        return highlights[:5]
