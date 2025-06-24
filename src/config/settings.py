# src/config/settings.py - Updated with new terminology and prompts
"""
Enhanced application settings for Spiegel RAG System (1948-1979).
Updated with new terminology: Heuristik, LLM-Unterstützte Auswahl, Zeitintervall-Suche
"""
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# API SETTINGS
# =============================================================================

# LLM API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# HU LLM Settings - Multiple endpoints
HU_LLM1_API_URL = os.getenv("HU_LLM1_API_URL", "https://llm1-compute.cms.hu-berlin.de/v1/")
HU_LLM3_API_URL = os.getenv("HU_LLM3_API_URL", "https://llm3-compute.cms.hu-berlin.de/v1/")

# Remote ChromaDB Settings
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "dighist.geschichte.hu-berlin.de")
CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT", "8000"))
CHROMA_DB_SSL = os.getenv("CHROMA_DB_SSL", "true").lower() == "true"
CHROMA_DB_CACHE_DIR = os.getenv("CHROMA_DB_CACHE_DIR", "./cache")

# Remote Ollama Settings
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://dighist.geschichte.hu-berlin.de:11434")

# DeepSeek R1 Settings
DEEPSEEK_R1_MODEL_NAME = os.getenv("DEEPSEEK_R1_MODEL_NAME", "deepseek-r1:32b")

# =============================================================================
# PATH SETTINGS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORD_EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "fasttext_model_spiegel_corpus_neu_50epochs_2.model")

# =============================================================================
# SEARCH AND RETRIEVAL SETTINGS
# =============================================================================

# Chunk Settings - Available sizes with their overlap values
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "3000"))
DEFAULT_CHUNK_OVERLAP_PERCENTAGE = int(os.getenv("DEFAULT_CHUNK_OVERLAP_PERCENTAGE", "10"))
AVAILABLE_CHUNK_SIZES = [500, 2000, 3000]

# Time Range Settings - Spiegel Archive Coverage
MIN_YEAR = 1948
MAX_YEAR = 1979

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model selection
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "hu-llm3")

# Available LLM models
AVAILABLE_LLM_MODELS = ["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"]

# LLM Display Names for UI - UPDATED terminology
LLM_DISPLAY_NAMES = {
    "hu-llm1": "HU-LLM 1 (Berlin)",
    "hu-llm3": "HU-LLM 3 (Berlin)", 
    "deepseek-r1": "DeepSeek R1 32B (Ollama)",
    "openai-gpt4o": "OpenAI GPT-4o",
    "gemini-pro": "Google Gemini 2.5 Pro"
}

# Semantic Expansion Settings
ENABLE_SEMANTIC_EXPANSION = True
DEFAULT_SEMANTIC_EXPANSION_FACTOR = 3

# =============================================================================
# LLM-UNTERSTÜTZTE AUSWAHL DEFAULTS (formerly "agent")
# =============================================================================

# Default settings for LLM-assisted search
LLM_ASSISTED_DEFAULT_CHUNKS_PER_WINDOW_INITIAL = 50
LLM_ASSISTED_DEFAULT_CHUNKS_PER_WINDOW_FINAL = 20
LLM_ASSISTED_DEFAULT_TIME_WINDOW_SIZE = 5
LLM_ASSISTED_DEFAULT_USE_TIME_WINDOWS = True
LLM_ASSISTED_DEFAULT_MIN_RETRIEVAL_SCORE = 0.25 
LLM_ASSISTED_DEFAULT_TEMPERATURE = 0.2  

# =============================================================================
# COLLECTION NAME GENERATION
# =============================================================================

def get_collection_name(
    chunk_size: int, 
    chunk_overlap: Optional[int] = None,
    embedding_model: str = "nomic-embed-text"
) -> str:
    """Generate standardized collection name."""
    # Use specific overlap values for available collections
    if chunk_overlap is None:
        overlap_map = {500: 100, 2000: 400, 3000: 300}
        chunk_overlap = overlap_map.get(chunk_size, 300)
    
    return f"recursive_chunks_{chunk_size}_{chunk_overlap}_TH_cosine_{embedding_model}"

# =============================================================================
# LLM CONFIGURATION MAPPING
# =============================================================================

def get_llm_config(model_name: str) -> Dict[str, str]:
    """Get LLM configuration for a given model name."""
    configs = {
        "hu-llm1": {
            "type": "hu-llm",
            "base_url": HU_LLM1_API_URL,
            "model_id": "llm1",
            "api_key": "required-but-not-used"
        },
        "hu-llm3": {
            "type": "hu-llm", 
            "base_url": HU_LLM3_API_URL,
            "model_id": "llm3",
            "api_key": "required-but-not-used"
        },
        "deepseek-r1": {
            "type": "ollama",
            "base_url": OLLAMA_BASE_URL,
            "model_id": DEEPSEEK_R1_MODEL_NAME,
            "api_key": "not-required"
        },
        "openai-gpt4o": {
            "type": "openai",
            "model_id": "gpt-4o",
            "api_key": OPENAI_API_KEY
        },
        "gemini-pro": {
            "type": "gemini",
            "model_id": "gemini-2.5-pro",
            "api_key": GEMINI_API_KEY
        }
    }
    
    return configs.get(model_name, configs["hu-llm3"])

# =============================================================================
# ENHANCED SYSTEM PROMPTS FOR HISTORICAL ANALYSIS
# =============================================================================

SYSTEM_PROMPTS = {
    "default": """Du bist ein erfahrener Historiker mit Expertise in der kritischen Auswertung von SPIEGEL-Artikeln aus den Jahren 1948-1979.

**Hauptaufgabe**: Beantworte die Forschungsfrage präzise und wissenschaftlich fundiert basierend ausschließlich auf den bereitgestellten Textauszügen.

**Methodik**:
* **Quellentreue**: Nutze ausschließlich die bereitgestellten Textauszüge als Grundlage
* **Wissenschaftliche Präzision**: Formuliere analytisch und differenziert
* **Vollständige Integration**: Berücksichtige möglichst viele relevante Textauszüge
* **Transparente Belege**: Verweise präzise mit [Datum, Artikeltitel]

**Antwortstruktur**:
1. Direkte Beantwortung der Forschungsfrage
2. Analytische Auswertung mit Quellenbelegen
3. Bei unzureichender Quellenlage: transparente Kommunikation der Grenzen

Beginne direkt mit der Analyse ohne einleitende Bemerkungen.""",
}

# =============================================================================
# LLM-UNTERSTÜTZTE AUSWAHL SYSTEM PROMPTS (formerly "agent")
# =============================================================================

LLM_ASSISTED_SYSTEM_PROMPTS = {
    "standard_evaluation": """Du bewertest Textabschnitte aus SPIEGEL-Artikeln (1948-1979) für historische Forschung.

**Aufgabe**: Analysiere zunächst jeden Textabschnitt ausführlich im Hinblick auf seine Relevanz für den user retrieval Query. Führe eine differenzierte Argumentation durch, bevor du eine Bewertung abgibst.

{"Vorgehen und Forschungsinteresse spezifizieren"}

**Vorgehen**:
1. **Argumentation**: Begründe ausführlich, inwiefern der Text relevant ist oder nicht. Berücksichtige historische Bedeutung, zeitgenössische Perspektive und Quellenwert.
2. **Bewertung**: Vergib anschließend einen Score auf einer Skala von 0-10, basierend auf deiner Argumentation.

**Bewertungsskala**:
- 9-10: Direkt relevant mit substanziellen Informationen
- 7-8: Stark relevant mit wichtigem Kontext
- 5-6: Mäßig relevant mit ergänzenden Aspekten
- 3-4: Schwach relevant mit entferntem Bezug
- 0-2: Nicht relevant für die Fragestellung

**Antwortformat**: 
"Text X: Argumentation: [Begründung] Score: Y/10"

Bewerte kritisch und stelle sicher, dass die Bewertung und Argumentation auf der Quelle und der user Query basiert sind.""",

    "negative_reranking": """Du bewertest Textabschnitte durch systematische Relevanzprüfung für historische Forschung.

**Vorgehen**:
1. **Contra-Argumente**: Erläutere danach, welche Faktoren gegen die Relevanz sprechen.
2. **Pro-Argumente**: Führe zuerst aus, welche Aspekte für die Relevanz des Textes sprechen.
3. **Abwägung**: Bewerte die Pro- und Contra-Argumente und ziehe eine Schlussfolgerung.
4. **Score**: Vergib abschließend einen Score (0-10), der sich aus deiner Argumentation ergibt.

**Antwortformat**: 
"Text X: Pro: [Argumente] Contra: [Argumente] Bewertung: [Schlussfolgerung] Score: Y/10"

Stelle sicher, dass die Argumentation und Abwägung immer vor der Vergabe des Scores erfolgt und die Bewertung nachvollziehbar ist."""
}

# Combine both prompt collections for easy access
ALL_SYSTEM_PROMPTS = {**SYSTEM_PROMPTS, **LLM_ASSISTED_SYSTEM_PROMPTS}

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Feature Flags (simplified)
ENABLE_QUERY_REFINEMENT = os.getenv("ENABLE_QUERY_REFINEMENT", "false").lower() == "true"
ENABLE_CITATIONS = os.getenv("ENABLE_CITATIONS", "false").lower() == "true"

# Logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_time_intervals(start_year: int, end_year: int, interval_size: int) -> Dict[str, any]:
    """Calculate time intervals and expected results."""
    intervals = []
    for interval_start in range(start_year, end_year + 1, interval_size):
        interval_end = min(interval_start + interval_size - 1, end_year)
        intervals.append((interval_start, interval_end))
    
    return {
        "intervals": intervals,
        "count": len(intervals),
        "coverage": f"{start_year}-{end_year} in {len(intervals)} Intervallen à {interval_size} Jahre"
    }