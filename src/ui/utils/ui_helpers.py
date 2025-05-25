# src/ui/utils/ui_helpers.py
"""
UI Helper functions for the Spiegel RAG application.
Contains functions that are used across multiple UI components.
"""
import json
import logging
from typing import Any, Dict, List, Tuple, Optional
import gradio as gr

# Configure logging
logger = logging.getLogger(__name__)

def toggle_api_key_visibility(model_choice: str) -> Dict:
    """
    Show or hide the API key input field based on the selected model.
    
    Args:
        model_choice: The selected model name
        
    Returns:
        Dict with update parameters for Gradio
    """
    if model_choice.startswith("openai"):
        return gr.update(visible=True)
    return gr.update(visible=False)

def format_expanded_keywords(expanded_words: Dict) -> str:
    """
    Format expanded keywords for display in the UI.
    
    Args:
        expanded_words: Dictionary mapping original words to expanded words
        
    Returns:
        Formatted markdown string
    """
    if not expanded_words:
        return "No expanded keywords available."
    
    result = "## Expanded Keywords\n\n"
    
    for original, similar in expanded_words.items():
        result += f"**{original}** → {', '.join(similar)}\n\n"
    
    return result

def format_search_metadata(
    model: str,
    query: str,
    question: str,
    chunk_size: int,
    year_start: int,
    year_end: int,
    search_time: float,
    num_chunks: int,
    keywords: Optional[str],
    search_fields: List[str],
    enforce_keywords: bool,
    use_semantic_expansion: bool,
    use_time_windows: bool = False,
    time_window_size: int = 5,
    expanded_words: Optional[Dict] = None,
    time_windows: Optional[List[Tuple[int, int]]] = None,
    window_counts: Optional[Dict[str, int]] = None
) -> str:
    """
    Format search metadata for display in the UI.
    
    Args:
        Various search parameters and results
        
    Returns:
        Formatted markdown string with search metadata
    """
    metadata_text = f"""
## Suchparameter
- **Model**: {model}
- **Suchanfrage**: {query}
- **Frage**: {question}
- **Chunk-Größe**: {chunk_size} Zeichen
- **Zeitraum**: {year_start} - {year_end}
- **Suchzeit**: {search_time:.2f} Sekunden
- **Gefundene Texte**: {num_chunks}

## Schlagwort-Filter
- **Schlagwörter**: {keywords or "Keine"}
- **Suchbereiche**: {', '.join(search_fields)}
- **Strikte Filterung**: {"Ja" if enforce_keywords else "Nein"}
- **Semantische Erweiterung**: {"Aktiviert" if use_semantic_expansion and keywords else "Deaktiviert"}
"""
    if use_time_windows and time_windows:
        metadata_text += f"\n## Zeitfenster-Suche\n- **Aktiviert**: Ja\n- **Fenstergröße**: {time_window_size} Jahre\n"
        
        # Display window distribution if available
        if window_counts:
            metadata_text += "\n**Verteilung der Ergebnisse nach Zeitfenstern:**\n"
            for window_start, window_end in time_windows:
                window_key = f"{window_start}-{window_end}"
                count = window_counts.get(window_key, 0)
                metadata_text += f"- **{window_key}**: {count} Texte\n"
    
    if use_semantic_expansion and keywords and expanded_words:
        metadata_text += "\n## Erweiterte Schlagwörter\n"
        for original, similar in expanded_words.items():
            metadata_text += f"- **{original}** → {', '.join(similar)}\n"
    
    return metadata_text
def format_analysis_metadata(
    question: str,
    model: str,
    analysis_time: float,
    retrieved_info: Dict[str, Any],
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Format analysis metadata for display in the UI.
    
    Args:
        question: The question that was asked
        model: The LLM model used
        analysis_time: Time taken for analysis in seconds
        retrieved_info: Metadata from the retrieval step
        temperature: Generation temperature used
        max_tokens: Maximum tokens set for generation
        system_prompt: System prompt used (if custom)
        
    Returns:
        Formatted markdown string with analysis metadata
    """
    metadata_text = f"""
## Analyseparameter
- **Model**: {model}
- **Frage**: {question}
- **Analysezeit**: {analysis_time:.2f} Sekunden
- **Temperatur**: {temperature}
- **Max Tokens**: {max_tokens or "Standardwert"}

## Quellen-Metadaten
- **Inhaltsbeschreibung**: {retrieved_info.get('content_description', 'Nicht angegeben')}
- **Chunk-Größe**: {retrieved_info.get('chunk_size', 'Unbekannt')} Zeichen
- **Zeitraum**: {retrieved_info.get('year_range', ['Unbekannt', 'Unbekannt'])[0]} - {retrieved_info.get('year_range', ['Unbekannt', 'Unbekannt'])[1]}
- **Anzahl Quellen**: {retrieved_info.get('chunks_count', 0)}
"""
    
    # Add system prompt info if available
    if system_prompt:
        # Truncate if too long
        prompt_display = system_prompt
        if len(prompt_display) > 300:
            prompt_display = prompt_display[:297] + "..."
        metadata_text += f"\n## System Prompt\n```\n{prompt_display}\n```"
    
    return metadata_text