"""
Enhanced application settings for Spiegel RAG System (1948-1979).
Updated version with multiple LLM options and expanded chunk sizes.
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

# Remote Ollama Embedding Settings
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://dighist.geschichte.hu-berlin.de:11434")

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

# Default Search Settings
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "hu-llm3")
AVAILABLE_LLM_MODELS = ["hu-llm1", "hu-llm3", "openai-gpt4o", "gemini-pro"]

# LLM Display Names for UI
LLM_DISPLAY_NAMES = {
    "hu-llm1": "HU-LLM 1 (Berlin)",
    "hu-llm3": "HU-LLM 3 (Berlin)", 
    "openai-gpt4o": "OpenAI GPT-4o",
    "gemini-pro": "Google Gemini Pro"
}

# Semantic Expansion Settings
ENABLE_SEMANTIC_EXPANSION = True
DEFAULT_SEMANTIC_EXPANSION_FACTOR = 3

# =============================================================================
# COLLECTION NAME GENERATION
# =============================================================================

def get_collection_name(
    chunk_size: int, 
    chunk_overlap: Optional[int] = None,
    embedding_model: str = "nomic-embed-text"
) -> str:
    """Generate collection name based on chunk size and overlap."""
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
        "openai-gpt4o": {
            "type": "openai",
            "model_id": "gpt-4o",
            "api_key": OPENAI_API_KEY
        },
        "gemini-pro": {
            "type": "gemini",
            "model_id": "gemini-pro",
            "api_key": GEMINI_API_KEY
        }
    }
    
    return configs.get(model_name, configs["hu-llm3"])

# =============================================================================
# ENHANCED SYSTEM PROMPTS FOR HISTORICAL ANALYSIS
# =============================================================================

SYSTEM_PROMPTS = {
    "default": """Du bist ein spezialisierter Assistent für die historische Analyse von SPIEGEL-Artikeln aus den Jahren 1948-1979. 

Deine Aufgabe:
- Beantworte Fragen ausschließlich basierend auf den bereitgestellten Textauszügen
- Berücksichtige den historischen Kontext der Nachkriegszeit und frühen Bundesrepublik
- Verweise präzise auf Quellen mit Datum und Titel
- Achte auf zeitgenössische Perspektiven und Terminologie
- Bei unzureichender Quellenlage: Kommuniziere dies transparent

Format: Beginne mit einer direkten Antwort, gefolgt von quellenbasierten Belegen.""",

    "historical_analysis": """Du bist ein Historiker, der SPIEGEL-Artikel aus der Nachkriegszeit (1948-1979) analysiert.

Fokussiere auf:
- Historische Kontextualisierung innerhalb der deutschen Nachkriegsgeschichte
- Entwicklungslinien und Wandel von Diskursen über die Jahrzehnte
- Zeitgenössische Wahrnehmungen vs. heutige Bewertungen
- Politische, gesellschaftliche und kulturelle Kontinuitäten/Brüche
- Einordnung in die Geschichte der frühen Bundesrepublik

Methodik:
- Nutze ausschließlich die bereitgestellten Quellen
- Zitiere präzise mit Datum und Artikeltitel
- Unterscheide zwischen Beschreibung und Interpretation
- Benenne Grenzen der Quellenauswahl explizit""",

    "media_critique": """Du bist ein Medienwissenschaftler, der die SPIEGEL-Berichterstattung von 1948-1979 kritisch analysiert.

Analysiere:
- Sprachliche Mittel und journalistische Strategien der SPIEGEL-Redaktion
- Framing von Ereignissen und Akteuren
- Narrative Strukturen und wiederkehrende Deutungsmuster
- Implizite Wertungen und ideologische Positionierungen
- Entwicklung des SPIEGEL-Stils über die Jahrzehnte

Methodik:
- Belege Aussagen mit konkreten Textstellen und Zitaten
- Kontextualisiere innerhalb der westdeutschen Medienlandschaft
- Unterscheide zwischen Nachricht, Kommentar und Meinungsäußerung
- Berücksichtige die spezifische Rolle des SPIEGEL im Mediensystem""",

    "discourse_analysis": """Du analysierst als Diskursanalytiker SPIEGEL-Texte von 1948-1979.

Untersuche:
- Dominante Diskurse und deren Wandel über die Zeit
- Sprachliche Konstruktion von Realität und Bedeutung
- Macht-/Wissensstrukturen in der Berichterstattung  
- Inklusions-/Exklusionsmechanismen in der Darstellung
- Kontinuitäten und Brüche in Diskursformationen

Vorgehen:
- Identifiziere diskursive Strategien und Topoi
- Analysiere Begrifflichkeiten und deren historische Semantik
- Zeige Interdiskursivität und intertextuelle Bezüge auf
- Berücksichtige gesellschaftliche Machtverhältnisse der Zeit""",

    "social_history": """Du bist Sozialhistoriker und untersuchst SPIEGEL-Artikel von 1948-1979 auf gesellschaftliche Aspekte.

Fokus:
- Darstellung sozialer Gruppen und Schichten
- Geschlechterrollen und Familienbilder
- Generationenkonflikte und -erfahrungen
- Urbanisierung, Modernisierung, Wohlstandsgesellschaft
- Alltagskultur und Mentalitäten

Methodik:
- Arbeite heraus, wie soziale Realitäten konstruiert werden
- Identifiziere Ein- und Ausschlüsse in der Berichterstattung
- Kontextualisiere innerhalb der Sozialgeschichte der BRD
- Nutze die SPIEGEL-Artikel als Spiegel zeitgenössischer Wahrnehmungen""",

    "political_history": """Du analysierst als Politikhistoriker die politische Berichterstattung des SPIEGEL von 1948-1979.

Schwerpunkte:
- Darstellung politischer Akteure und Institutionen
- Demokratisierungsprozesse und politische Kultur
- Ost-West-Konflikt und deutsche Teilung
- Außenpolitik und internationale Beziehungen
- Politische Skandale und Krisen

Analyse:
- Politische Positionierungen und Parteinahmen des SPIEGEL
- Entwicklung der politischen Sprache und Begrifflichkeiten
- Kontinuitäten zu Weimarer Republik und NS-Zeit
- Rolle des SPIEGEL als politischer Akteur (vierte Gewalt)"""
}

# =============================================================================
# HISTORICAL PERIOD CONTEXTS
# =============================================================================

PERIOD_CONTEXTS = {
    "1948-1949": """Gründungsjahre der Bundesrepublik Deutschland:
- Währungsreform und Marshall-Plan
- Parlamentarischer Rat und Grundgesetz
- Beginn des Kalten Krieges
- Entnazifizierung und demokratischer Neubeginn""",
    
    "1950-1955": """Frühe Bundesrepublik und Westintegration:
- Wiederbewaffnungsdebatte und NATO-Beitritt
- Montanunion und erste europäische Integration
- Wirtschaftswunder nimmt Fahrt auf
- Adenauer-Ära beginnt""",
    
    "1956-1960": """Konsolidierung und gesellschaftlicher Wandel:
- Römische Verträge und EWG-Gründung
- Spiegel-Affäre 1962 kündigt sich an
- Wohlstandsgesellschaft etabliert sich
- Generationswechsel bahnt sich an""",
    
    "1961-1965": """Krisenjahre und Umbruch:
- Mauerbau in Berlin (1961)
- Spiegel-Affäre (1962) 
- Ende der Adenauer-Ära
- Erste Gastarbeiter-Generation""",
    
    "1966-1970": """Große Koalition und gesellschaftlicher Aufbruch:
- NPD-Erfolge und demokratische Krise
- Studentenbewegung und 68er-Proteste
- Außerparlamentarische Opposition
- Gesellschaftliche Liberalisierung""",
    
    "1971-1975": """Sozial-liberale Koalition und Reformen:
- Ostpolitik und Entspannung
- Bildungsreform und gesellschaftliche Modernisierung
- Erste Ölkrise (1973)
- Terrorismus der RAF""",
    
    "1976-1979": """Krisenzeit und Deutsche Herbst:
- Terrorismus erreicht Höhepunkt (1977)
- Wirtschaftliche Stagnation
- NATO-Doppelbeschluss (1979)
- Ende der Reformeuphorie"""
}

# =============================================================================
# AGENT SCORING PROMPT FOR HISTORICAL DOCUMENTS
# =============================================================================

AGENT_SCORING_PROMPT = """Du bist ein Experte für die Bewertung historischer Quellen und arbeitest mit SPIEGEL-Artikeln aus der Zeit von 1948-1979.

Bewertungskriterien für Relevanz zur Forschungsfrage (Skala 0-10):

9-10: Außergewöhnlich relevant
- Direkter, substantieller Bezug zur Forschungsfrage
- Einzigartige Informationen oder Perspektiven
- Schlüsseldokument für das Thema

7-8: Hoch relevant  
- Klarer Bezug zur Forschungsfrage
- Wichtige Informationen oder Kontext
- Gute Illustration des Themas

5-6: Mäßig relevant
- Teilbezug zur Forschungsfrage
- Ergänzende Informationen
- Allgemeiner historischer Kontext

3-4: Gering relevant
- Schwacher Bezug zur Forschungsfrage  
- Minimale verwertbare Informationen
- Hauptsächlich tangentiale Erwähnung

0-2: Nicht relevant
- Kein erkennbarer Bezug zur Forschungsfrage
- Keine verwertbaren Informationen

Berücksichtige dabei:
- Historische Bedeutung des Dokuments
- Quellenwert für die spezifische Fragestellung
- Typizität oder Besonderheit der Darstellung
- Zeitgenössische vs. nachträgliche Perspektive

Antworte für jeden Text:
Text X: Score Y - Historische Begründung unter Nennung spezifischer Aspekte"""

# =============================================================================
# QUALITY INDICATORS FOR HISTORICAL ANALYSIS
# =============================================================================

HISTORICAL_QUALITY_INDICATORS = [
    "Präzise Quellenangaben (Datum, Titel, Autor wenn verfügbar)",
    "Kontextualisierung innerhalb der Zeitperiode", 
    "Unterscheidung zwischen zeitgenössischer und heutiger Sicht",
    "Berücksichtigung der SPIEGEL-spezifischen Perspektive",
    "Transparenz bei unzureichender Quellenlage",
    "Methodische Reflexion der Quellenauswahl",
    "Einordnung in größere historische Entwicklungen"
]

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

def get_period_context(start_year: int, end_year: int) -> str:
    """Get historical context for a specific time period."""
    # Find the most appropriate period context
    for period, context in PERIOD_CONTEXTS.items():
        period_start, period_end = map(int, period.split('-'))
        if period_start <= start_year <= period_end or period_start <= end_year <= period_end:
            return context
    
    # Fallback to general period description
    return f"Nachkriegszeit und frühe Bundesrepublik ({start_year}-{end_year})"

def get_system_prompt_with_context(prompt_type: str = "default", 
                                 start_year: int = None, 
                                 end_year: int = None) -> str:
    """Get system prompt with optional historical context."""
    base_prompt = SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])
    
    if start_year and end_year:
        context = get_period_context(start_year, end_year)
        base_prompt += f"\n\nHistorischer Kontext für den Zeitraum {start_year}-{end_year}:\n{context}"
    
    return base_prompt