"""
Application settings loaded from environment variables.
"""
import os
from typing import Dict, List, Optional, Union, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HU_LLM_API_URL = os.getenv("HU_LLM_API_URL", "https://llm3-compute.cms.hu-berlin.de/v1/")

# Vector Database Settings - REMOTE ONLY
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "dighist.geschichte.hu-berlin.de")
CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT", "8000"))
CHROMA_DB_SSL = os.getenv("CHROMA_DB_SSL", "true").lower() == "true"

# Local path only needed for cache
CHROMA_DB_CACHE_DIR = os.getenv("CHROMA_DB_CACHE_DIR", "./cache")

# Embedding Settings - REMOTE OLLAMA ONLY 
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://dighist.geschichte.hu-berlin.de:11434")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORD_EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "fasttext_model_spiegel_corpus_neu_50epochs_2.model")

# Chunk Settings
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "3000"))
DEFAULT_CHUNK_OVERLAP_PERCENTAGE = int(os.getenv("DEFAULT_CHUNK_OVERLAP_PERCENTAGE", "10"))
AVAILABLE_CHUNK_SIZES = [2000, 3000]  # Nur diese beiden Größen sind verfügbar

# Default LLM Model
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "hu-llm")
AVAILABLE_LLM_MODELS = ["hu-llm", "gpt4o"]

# Time range settings
MIN_YEAR = 1948
MAX_YEAR = 1979

# Application Settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Collection name format (used in vector_store.py)
def get_collection_name(
    chunk_size: int, 
    chunk_overlap: Optional[int] = None,
    embedding_model: str = "nomic-embed-text"
) -> str:
    """Generate collection name based on chunk size and overlap."""
    # Verwende die spezifischen Überlappungswerte für die verfügbaren Kollektionen
    if chunk_overlap is None:
        if chunk_size == 2000:
            chunk_overlap = 400
        elif chunk_size == 3000:
            chunk_overlap = 300
        else:
            # Fallback zur ursprünglichen Berechnung
            chunk_overlap = chunk_size // 10
    
    return f"recursive_chunks_{chunk_size}_{chunk_overlap}_TH_cosine_{embedding_model}"

# Add feature flags
ENABLE_QUERY_REFINEMENT = os.getenv("ENABLE_QUERY_REFINEMENT", "false").lower() == "true"
ENABLE_CITATIONS = os.getenv("ENABLE_CITATIONS", "false").lower() == "true"

# System prompts for different use cases
SYSTEM_PROMPTS = {
    "default": """Du bist ein hilfreicher Assistent, der die gestellte Frage einzig auf 
    Grundlage der nachfolgenden Textauszüge aus dem SPIEGEL-Archiv beantwortet. 
    Beziehe dich dabei explizit auf die Quellen und deren Datum. 
    Sind die Textauszüge für die gestellte Frage nicht relevant, dann sag das klar.""",
    
    "historical_analysis": """Als Historiker analysierst du die SPIEGEL-Artikel mit 
    besonderem Fokus auf historische Kontextualisierung, Entwicklungslinien und
    zeitgenössische Diskurse. Beziehe dich dabei auf die konkreten Quellen und 
    ihre Datierung. Wenn die Quellen keine ausreichende Basis bieten, kommuniziere
    dies transparent.""",
    
    "media_critique": """Als Medienwissenschaftler analysierst du die Berichterstattung 
    des SPIEGEL kritisch. Achte dabei auf Sprache, Framing, Narrative und implizite
    Wertungen. Belege deine Analyse mit konkreten Zitaten aus den vorliegenden
    Quellen. Bei unzureichender Datenlage kommuniziere dies klar."""

}

# Add configuration for semantic search
ENABLE_SEMANTIC_EXPANSION = True
DEFAULT_SEMANTIC_EXPANSION_FACTOR = 3