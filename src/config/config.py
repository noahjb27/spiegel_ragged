# src/config/config.py
"""
Clean configuration system using Pydantic for validation.
Replaces the messy settings.py with a proper configuration class.
"""
from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseSettings, Field, validator
import os


class Settings(BaseSettings):
    """Application settings with validation and defaults"""
    
    # API Settings
    openai_api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    hu_llm_api_url: str = Field(
        "https://llm3-compute.cms.hu-berlin.de/v1/",
        env='HU_LLM_API_URL'
    )
    
    # Remote ChromaDB Settings
    chroma_db_host: str = Field(
        "dighist.geschichte.hu-berlin.de",
        env='CHROMA_DB_HOST'
    )
    chroma_db_port: int = Field(8000, env='CHROMA_DB_PORT')
    chroma_db_ssl: bool = Field(True, env='CHROMA_DB_SSL')
    
    # Remote Ollama Settings
    ollama_model_name: str = Field(
        "nomic-embed-text",
        env='OLLAMA_MODEL_NAME'
    )
    ollama_base_url: str = Field(
        "https://dighist.geschichte.hu-berlin.de:11434",
        env='OLLAMA_BASE_URL'
    )
    
    # Search Settings
    default_chunk_size: int = Field(3000, env='DEFAULT_CHUNK_SIZE')
    available_chunk_sizes: List[int] = [2000, 3000]
    min_year: int = 1948
    max_year: int = 1979
    default_top_k: int = 10
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    word_embedding_model_path: Optional[Path] = None
    
    # Feature Flags (simplified - removed unused ones)
    enable_semantic_expansion: bool = True
    semantic_expansion_factor: int = 3
    
    # System Prompts
    system_prompts: Dict[str, str] = {
        "default": """Du bist ein hilfreicher Assistent, der Fragen basierend auf 
        historischen Texten aus dem SPIEGEL-Archiv beantwortet. 
        Beziehe dich explizit auf die Quellen und deren Datum.""",
        
        "historical_analysis": """Als Historiker analysierst du die SPIEGEL-Artikel mit 
        Fokus auf historische Kontextualisierung und zeitgenössische Diskurse.""",
        
        "media_critique": """Als Medienwissenschaftler analysierst du die Berichterstattung 
        kritisch mit Blick auf Sprache, Framing und implizite Wertungen."""
    }
    
    # Logging
    log_level: str = Field("INFO", env='LOG_LEVEL')
    debug: bool = Field(False, env='DEBUG')
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
    
    @validator('word_embedding_model_path', pre=True, always=True)
    def set_embedding_path(cls, v, values):
        """Set default embedding model path if not provided"""
        if v is None and 'base_dir' in values:
            return values['base_dir'] / 'models' / 'fasttext_model_spiegel_corpus_neu_50epochs_2.model'
        return v
    
    @validator('available_chunk_sizes')
    def validate_chunk_sizes(cls, v, values):
        """Ensure default chunk size is in available sizes"""
        if 'default_chunk_size' in values and values['default_chunk_size'] not in v:
            raise ValueError(f"Default chunk size {values['default_chunk_size']} not in available sizes {v}")
        return v
    
    def get_collection_name(self, chunk_size: int) -> str:
        """Generate collection name for ChromaDB"""
        # Map chunk sizes to their specific overlaps
        overlap_map = {
            2000: 400,
            3000: 300
        }
        
        overlap = overlap_map.get(chunk_size, chunk_size // 10)
        return f"recursive_chunks_{chunk_size}_{overlap}_TH_cosine_{self.ollama_model_name}"
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured"""
        return bool(self.openai_api_key)


# Create singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings
    
    if _settings is None:
        _settings = Settings()
    
    return _settings


# Convenience exports
settings = get_settings()


# src/config/__init__.py
"""Configuration module"""
from .config import settings, get_settings, Settings

__all__ = ['settings', 'get_settings', 'Settings']