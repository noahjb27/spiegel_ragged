# src/config/settings.py - Updated with agent-specific prompts
"""
Enhanced application settings for Spiegel RAG System (1948-1979).
Updated version with agent-specific system prompts.
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

# Remote Ollama Settings - Used for both embeddings AND text generation
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

# Available LLM models - UPDATED to include DeepSeek R1
AVAILABLE_LLM_MODELS = ["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"]

# LLM Display Names for UI - UPDATED
LLM_DISPLAY_NAMES = {
    "hu-llm1": "HU-LLM 1 (Berlin)",
    "hu-llm3": "HU-LLM 3 (Berlin)", 
    "deepseek-r1": "DeepSeek R1 32B (Ollama)",
    "openai-gpt4o": "OpenAI GPT-4o",
    "gemini-pro": "Google Gemini Pro"
}


# Semantic Expansion Settings
ENABLE_SEMANTIC_EXPANSION = True
DEFAULT_SEMANTIC_EXPANSION_FACTOR = 3

# =============================================================================
# AGENT SEARCH DEFAULTS
# =============================================================================

# Default settings for agent-based search
AGENT_DEFAULT_CHUNKS_PER_WINDOW_INITIAL = 50
AGENT_DEFAULT_CHUNKS_PER_WINDOW_FINAL = 20
AGENT_DEFAULT_TIME_WINDOW_SIZE = 5
AGENT_DEFAULT_USE_TIME_WINDOWS = True
AGENT_DEFAULT_MIN_RETRIEVAL_SCORE = 0.25 

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
# AGENT-SPECIFIC SYSTEM PROMPTS FOR SOURCE EVALUATION
# =============================================================================

AGENT_SYSTEM_PROMPTS = {
    "agent_default": """Du bist ein Experte für die Bewertung historischer Quellen und arbeitest mit SPIEGEL-Artikeln aus der Zeit von 1948-1979.

Deine Aufgabe ist es, jeden Textabschnitt hinsichtlich seiner Relevanz für die gestellte Forschungsfrage zu bewerten.

Bewertungskriterien für Relevanz (Skala 0-10):
- 9-10: Außergewöhnlich relevant - direkter, substantieller Bezug zur Forschungsfrage mit einzigartigen Informationen
- 7-8: Hoch relevant - klarer Bezug zur Forschungsfrage mit wichtigen Informationen
- 5-6: Mäßig relevant - teilweiser Bezug zur Forschungsfrage mit ergänzenden Informationen
- 3-4: Gering relevant - schwacher Bezug zur Forschungsfrage mit minimalen verwertbaren Informationen
- 0-2: Nicht relevant - kein erkennbarer Bezug zur Forschungsfrage

Berücksichtige dabei:
- Historische Bedeutung des Dokuments für die Fragestellung
- Quellenwert für die spezifische Forschungsfrage
- Zeitgenössische vs. nachträgliche Perspektive
- Einzigartigkeit der enthaltenen Informationen

Antworte für jeden Text: "Score X - Kurze historische Begründung unter Nennung spezifischer Aspekte".""",

    "agent_media_analysis": """Du bewertest SPIEGEL-Artikel (1948-1979) für medienwissenschaftliche Fragestellungen.

Bewerte jeden Textabschnitt (Skala 0-10) basierend auf:
- Relevanz für medienkritische Analyse
- Beispiele journalistischer Strategien und Techniken
- Sprachliche Besonderheiten und Stilmittel
- Darstellung von Akteuren und Ereignissen
- Typische SPIEGEL-Narrative und Deutungsmuster

Besonders relevant sind Texte, die:
- Charakteristische SPIEGEL-Sprache zeigen
- Framing-Strategien erkennbar machen
- Medienpolitische Positionierungen deutlich werden lassen
- Journalistische Innovation oder Tradition repräsentieren

Antworte: "Score X - Medienkritische Begründung mit konkreten sprachlichen/stilistischen Beispielen".""",

    "agent_discourse_analysis": """Du bewertest SPIEGEL-Texte (1948-1979) für diskursanalytische Untersuchungen.

Bewerte jeden Textabschnitt (Skala 0-10) basierend auf:
- Diskursive Relevanz für die Forschungsfrage
- Präsenz dominanter Diskurse der Zeit
- Sprachliche Konstruktion von Bedeutung
- Macht-/Wissensstrukturen in der Darstellung
- Begriffliche und semantische Besonderheiten

Besonders relevant sind Texte mit:
- Typischen Diskursformationen der Nachkriegszeit
- Schlüsselbegriffen und deren Verwendung
- Interdiskursiven Bezügen
- Ideologischen Positionierungen

Antworte: "Score X - Diskursanalytische Begründung mit Fokus auf Begriffe und Bedeutungskonstruktion".""",

    "agent_historical_context": """Du bewertest SPIEGEL-Artikel (1948-1979) für historisch-kontextuelle Analysen.

Bewerte jeden Textabschnitt (Skala 0-10) basierend auf:
- Historische Aussagekraft für die Forschungsfrage
- Zeitgenössische Perspektiven und Wahrnehmungen
- Quellencharakter für die deutsche Nachkriegsgeschichte
- Dokumentation gesellschaftlicher Entwicklungen
- Einordnung in größere historische Zusammenhänge

Besonders relevant sind Texte, die:
- Zeitgenössische Sichtweisen authentisch wiedergeben
- Wichtige historische Entwicklungen dokumentieren
- Gesellschaftliche Stimmungen und Mentalitäten zeigen
- Kontinuitäten und Brüche der deutschen Geschichte beleuchten

Antworte: "Score X - Historische Begründung mit Einordnung in den zeitgenössischen Kontext".""",

    "agent_political_analysis": """Du bewertest SPIEGEL-Artikel (1948-1979) für politikwissenschaftliche und politikhistorische Analysen.

Bewerte jeden Textabschnitt (Skala 0-10) basierend auf:
- Politische Relevanz für die Forschungsfrage
- Darstellung politischer Akteure und Prozesse
- Demokratisierung und politische Kultur der frühen BRD
- Ost-West-Konflikt und internationale Beziehungen
- Politische Meinungsbildung und öffentlicher Diskurs

Besonders relevant sind Texte über:
- Politische Entscheidungsprozesse und deren Darstellung
- Parteien, Politiker und politische Institutionen
- Außenpolitik und internationale Verflechtungen
- Politische Krisen und deren mediale Vermittlung

Antworte: "Score X - Politikanalytische Begründung mit Fokus auf politische Akteure und Prozesse".""",

    "agent_social_cultural": """Du bewertest SPIEGEL-Artikel (1948-1979) für sozial- und kulturgeschichtliche Fragestellungen.

Bewerte jeden Textabschnitt (Skala 0-10) basierend auf:
- Soziale und kulturelle Relevanz für die Forschungsfrage
- Darstellung gesellschaftlicher Gruppen und Schichten
- Alltagskultur und Mentalitäten
- Modernisierung und gesellschaftlicher Wandel
- Geschlechter-, Generationen- und Klassenverhältnisse

Besonders relevant sind Texte, die:
- Soziale Realitäten und Lebenswelten beschreiben
- Kulturelle Praktiken und Wertvorstellungen zeigen
- Gesellschaftliche Konflikte und Veränderungen dokumentieren
- Alltägliche Erfahrungen und Wahrnehmungen wiedergeben

Antworte: "Score X - Sozialhistorische Begründung mit Fokus auf gesellschaftliche Aspekte und Alltagskultur"."""
}

# Combine both prompt collections for easy access
ALL_SYSTEM_PROMPTS = {**SYSTEM_PROMPTS, **AGENT_SYSTEM_PROMPTS}

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
    base_prompt = ALL_SYSTEM_PROMPTS.get(prompt_type, ALL_SYSTEM_PROMPTS["default"])
    
    if start_year and end_year:
        context = get_period_context(start_year, end_year)
        base_prompt += f"\n\nHistorischer Kontext für den Zeitraum {start_year}-{end_year}:\n{context}"
    
    return base_prompt