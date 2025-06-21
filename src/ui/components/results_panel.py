# src/ui/components/results_panel.py - Updated with metadata at bottom and TXT download
"""
Updated results panel component with metadata moved to bottom and download functionality.
"""
import gradio as gr
from typing import Dict, Any
from datetime import datetime

def create_results_panel() -> Dict[str, Any]:
    """
    Create the updated results panel UI components using main CSS classes.
    """
    with gr.Group():
        # Removed duplicate CSS - now uses main CSS from app.py
        
        # Analysis info box using main CSS classes
        gr.HTML("""
        <div class="analysis-info">
            <h4>📊 Analyse-Ergebnisse</h4>
         </div>
        """)
        
        # Main answer output using CSS classes from main app
        with gr.Column(elem_classes=["results-container"]):
            answer_output = gr.Markdown(
                value="Die Analyse erscheint hier...",
                label="Analyse-Ergebnisse"
            )
        
        # Metadata in collapsible accordion using main CSS
        with gr.Accordion("Ausgabe-Metadaten", open=False):
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gr.Markdown(f"**Erstellt am**: {current_date}")
            
            metadata_output = gr.Markdown(
                value="Detaillierte Metadaten zur Analyse erscheinen hier...",
                elem_classes=["metadata-section"]
            )
    
    components = {
        "answer_output": answer_output,
        "metadata_output": metadata_output
    }
    
    return components

def format_analysis_result(answer: str, metadata: Dict[str, Any], user_prompt: str, model: str) -> tuple:
    """
    Format analysis results with enhanced presentation using main CSS styling.
    """
    formatted_answer = f"""# Analyse-Ergebnisse

## Forschungsfrage
> {user_prompt}

## Antwort

{answer}

---

*Generiert mit {model} • Basierend auf ausgewählten historischen Quellen aus dem SPIEGEL-Archiv (1948-1979)*
"""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chunks_analyzed = metadata.get('chunks_analyzed', 0)
    analysis_time = metadata.get('search_time', 0)
    temperature = metadata.get('temperature', 0.3)
    
    formatted_metadata = f"""## Analyse-Parameter

- **Modell**: {model}
- **Forschungsfrage**: {user_prompt}
- **Analysezeit**: {analysis_time:.2f} Sekunden
- **Temperatur**: {temperature} (Determinismus-Grad)
- **Erstellt am**: {current_time}

## Quellen-Information

- **Anzahl analysierter Quellen**: {chunks_analyzed}
- **Zeitraum**: SPIEGEL-Archiv 1948-1979
- **Auswahlmethode**: {metadata.get('selection_method', 'Benutzerauswahl')}

## System-Konfiguration

- **RAG-System**: SPIEGEL RAG System
- **Embedding-Modell**: nomic-embed-text (Ollama)
- **Vektor-Datenbank**: ChromaDB
"""
    
    return formatted_answer, formatted_metadata

def create_analysis_download_content(answer: str, metadata_text: str, user_prompt: str, model: str) -> str:
    """
    Create properly formatted content for TXT download.
    
    Args:
        answer: Analysis answer
        metadata_text: Metadata text
        user_prompt: User's question
        model: Model used
        
    Returns:
        Formatted text content for download
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""SPIEGEL RAG SYSTEM - ANALYSE-ERGEBNISSE
Erstellt am: {timestamp}

{'='*60}
FORSCHUNGSFRAGE
{'='*60}

{user_prompt}

{'='*60}
ANALYSE-ERGEBNISSE
{'='*60}

{answer}

{'='*60}
SYSTEM-INFORMATIONEN
{'='*60}

Modell: {model}
System: SPIEGEL RAG System (1948-1979)

{'='*60}
METADATEN
{'='*60}

{metadata_text}
"""
    
    return content