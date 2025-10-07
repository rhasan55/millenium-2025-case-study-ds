from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime
import numpy as np

from extraction import DocumentExtractor
from llm_parser import ResumeParser
from embeddings import EmbeddingEngine
from deduplication import EntityResolver
from models import Candidate, CandidateDatabase

logger = logging.getLogger(__name__)

class IngestionPipeline:

    def __init__(self, openai_api_key: str, use_openai_embeddings: bool = True):
        self.parser = ResumeParser(openai_api_key)
        self.embedding_engine = EmbeddingEngine(use_openai=use_openai_embeddings)
        self.extractor = DocumentExtractor()

    def process_resume(self, file_path: Path) -> Optional[Candidate]:
        logger.info(f"Processing: {file_path.name}")

        raw_text = self.extractor.extract_text(file_path)
        if not raw_text or len(raw_text) < 50:
            logger.warning(f"Insufficient text extracted from {file_path.name}")
            return None

        normalized_text = self.extractor.normalize_text(raw_text)

        candidate = self.parser.parse_resume(normalized_text, file_path.name)
        if not candidate:
            logger.error(f"Parsing failed for {file_path.name}")
            return None

        embedding = self.embedding_engine.generate_candidate_embedding(candidate)
        candidate.embedding = embedding.tolist()

        logger.info(f"Successfully processed: {candidate.name} (confidence: {candidate.meta.parse_confidence:.2f})")
        return candidate

    def process_directory(self, directory: Path) -> CandidateDatabase:
        database = CandidateDatabase()
        supported_extensions = {".pdf", ".docx", ".doc"}

        resume_files = []
        for ext in supported_extensions:
            resume_files.extend(directory.glob(f"*{ext}"))

        logger.info(f"Found {len(resume_files)} resume files")

        for file_path in resume_files:
            try:
                candidate = self.process_resume(file_path)
                if candidate:
                    database.add_candidate(candidate)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue

        logger.info(f"Processed {len(database.candidates)} candidates")

        duplicates = EntityResolver.find_duplicates(database.candidates)
        if duplicates:
            logger.info(f"Found {len(duplicates)} potential duplicates")

        embeddings_matrix = np.array([c.embedding for c in database.candidates])
        database.embeddings = embeddings_matrix

        return database

    def get_processing_stats(self, database: CandidateDatabase) -> dict:
        total = len(database.candidates)
        avg_confidence = np.mean([c.meta.parse_confidence for c in database.candidates])
        avg_processing_time = np.mean([c.meta.processing_time_ms for c in database.candidates if c.meta.processing_time_ms])

        strategies = {}
        for candidate in database.candidates:
            for strategy in candidate.strategy:
                strategies[strategy] = strategies.get(strategy, 0) + 1

        sectors = {}
        for candidate in database.candidates:
            for sector in candidate.sector_coverage:
                sectors[sector] = sectors.get(sector, 0) + 1

        return {
            "total_candidates": total,
            "avg_parse_confidence": avg_confidence,
            "avg_processing_time_ms": avg_processing_time,
            "strategies": strategies,
            "sectors": sectors
        }
