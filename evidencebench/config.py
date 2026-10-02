"""
Configuration module for EvidenceBench RAG system.
Contains tunable hyperparameters for ingestion, chunking, retrieval, reranking, and abstention.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("evidencebench.config")

# Robust base directory detection
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = _THIS_FILE.parent
_PROJECT_ROOT_CANDIDATES = [
    _PACKAGE_DIR.parent,  # normal: scratch/evidencebench
    Path(os.getcwd()).resolve(),  # current working directory (Streamlit Cloud root)
    Path(".").resolve(),
]

def find_project_root() -> Path:
    """Finds the root directory containing data/corpus or evidencebench package."""
    # 1. Environment variable override
    env_root = os.getenv("EVIDENCEBENCH_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p

    # 2. Check candidate roots
    for candidate in _PROJECT_ROOT_CANDIDATES:
        if (candidate / "data" / "corpus").exists() or (candidate / "evidencebench").exists():
            return candidate

    # 3. Default to package parent
    return _PACKAGE_DIR.parent

PROJECT_ROOT = find_project_root()

def get_corpus_dir() -> Path:
    """Resolves data/corpus directory with robust fallbacks."""
    env_corpus = os.getenv("CORPUS_DIR")
    if env_corpus:
        p = Path(env_corpus).resolve()
        if p.exists():
            return p

    candidates: List[Path] = [
        PROJECT_ROOT / "data" / "corpus",
        Path.cwd() / "data" / "corpus",
        _PACKAGE_DIR.parent / "data" / "corpus",
        Path("data/corpus").resolve(),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    # Create default if not existing
    default_dir = PROJECT_ROOT / "data" / "corpus"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir

def get_storage_dir() -> Path:
    """Resolves data/indices directory with robust fallbacks."""
    env_storage = os.getenv("STORAGE_DIR")
    if env_storage:
        p = Path(env_storage).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    candidates: List[Path] = [
        PROJECT_ROOT / "data" / "indices",
        Path.cwd() / "data" / "indices",
        _PACKAGE_DIR.parent / "data" / "indices",
        Path("data/indices").resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c

    default_dir = PROJECT_ROOT / "data" / "indices"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = get_corpus_dir()
BENCHMARK_DIR = DATA_DIR / "benchmark"
INDICES_DIR = get_storage_dir()


class EvidenceBenchConfig(BaseModel):
    # Ingestion & Document Store
    storage_dir: Path = INDICES_DIR
    corpus_dir: Path = CORPUS_DIR
    benchmark_file: Path = BENCHMARK_DIR / "dataset.json"

    # Chunking Hyperparameters
    fixed_chunk_size: int = 500  # characters
    fixed_chunk_overlap: int = 100  # characters
    structural_max_chunk_size: int = 800  # characters
    structural_min_chunk_size: int = 150  # characters

    # Dense Embedding
    dense_model_name: str = "all-MiniLM-L6-v2"
    dense_batch_size: int = 32

    # Keyword (BM25)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Hybrid Fusion (Reciprocal Rank Fusion + Normalized Score Calibration)
    rrf_k: int = 60
    dense_weight: float = 0.6
    keyword_weight: float = 0.4
    hybrid_top_k: int = 20  # Stage 1 candidate pool size

    # Reranker
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5  # Final candidate set for synthesis
    reranker_batch_size: int = 16

    # Abstention Gatekeeper Thresholds
    weak_evidence_threshold: float = 0.35  # Normalized rerank score below this triggers weak evidence abstention
    missing_evidence_overlap_threshold: float = 0.15  # Minimum lexical/semantic overlap required
    conflict_similarity_threshold: float = 0.70  # Chunks must be on same topic to trigger conflict check
    contradiction_score_threshold: float = 0.60  # Polarity/numerical discrepancy score threshold

    # Generation & Citations
    citation_overlap_min: float = 0.40  # Minimum token/subword overlap to verify citation alignment
    max_context_tokens: int = 2048


# Default singleton instance
config = EvidenceBenchConfig()
