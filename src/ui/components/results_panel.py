# src/ui/components/results_panel.py - Updated with metadata at bottom and TXT download
"""
Updated results panel component with metadata moved to bottom and download functionality.
"""
import gradio as gr
from typing import Dict, Any
from datetime import datetime

def create_results_panel() -> Dict[str, Any]:
    """
    Create the updated results panel UI components with improved styling and download functionality.
    """
    with gr.Group():
        # Add CSS for better formatting
        gr.HTML("""
        <style>
            /* UPDATED: Results section styling with new color scheme */
            .results-container {
                padding: 20px !important;
                border-radius: 8px !important;
                border: 1px solid #968d84 !important;  /* NEW: Updated border */
                background-color: #ffffff !important;
            }
            
            /* Better typography */
            .results-container p, .results-container li {
                font-size: 16px !important;
                line-height: 1.6 !important;
                color: #2c3e50 !important;  /* Better contrast */
            }
            
            /* Better heading styles with new colors */
            .results-container h1, .results-container h2, .results-container h3 {
                margin-top: 1em !important;
                margin-bottom: 0.5em !important;
                color: #5a5248 !important;  /* NEW: Darker shade of #968d84 */
            }
            
            .results-container h1 {
                color: #d75425 !important;  /* NEW: Orange for main headings */
                border-bottom: 2px solid #d75425 !important;
                padding-bottom: 5px !important;
            }
            
            /* Better list styling */
            .results-container ul, .results-container ol {
                padding-left: 2em !important;
                margin-bottom: 1em !important;
            }
            
            /* Quote styling with new colors */
            .results-container blockquote {
                border-left: 4px solid #b2b069 !important;  /* NEW: Yellow-green accent */
                padding-left: 1em !important;
                margin-left: 0 !important;
                font-style: italic !important;
                background-color: #f9f8f4 !important;  /* NEW: Light background */
                padding: 10px 15px !important;
                border-radius: 0 5px 5px 0 !important;
            }
            
            /* Source citation styling */
            .results-container .citation {
                background-color: #f4f1ee !important;  /* NEW: Light gray background */
                padding: 5px 10px !important;
                border-radius: 3px !important;
                border-left: 3px solid #968d84 !important;  /* NEW: Gray accent */
                margin: 5px 0 !important;
                font-size: 0.9em !important;
            }
            
            /* Metadata section styling */
            .metadata-section {
                background-color: #fafafa !important;
                padding: 15px !important;
                border-radius: 8px !important;
                border: 1px solid #968d84 !important;  /* NEW: Updated border */
                margin-top: 20px !important;
            }
            
            .metadata-section h3 {
                color: #5a5248 !important;  /* NEW: Updated heading color */
                margin-top: 0 !important;
            }
            
            /* Analysis info box */
            .analysis-info {
                background: linear-gradient(135deg, #f9f8f4 0%, #f4f1ee 100%) !important;  /* NEW: Gradient */
                padding: 15px !important;
                border-radius: 8px !important;
                border-left: 4px solid #b2b069 !important;  /* NEW: Yellow-green accent */
                margin-bottom: 20px !important;
            }
            
            .analysis-info h4 {
                color: #6b6840 !important;  /* NEW: Darker shade */
                margin-bottom: 10px !important;
                font-weight: bold !important;
            }
        </style>
        """)
        
        # Analysis info box
        gr.HTML("""
        <div class="analysis-info">
            <h4>📊 Analyse-Ergebnisse</h4>
         </div>
        """)
        
        # Main answer output with custom class for styling
        with gr.Column(elem_classes=["results-container"]):
            answer_output = gr.Markdown(
                value="Die Analyse erscheint hier...",
                label="Analyse-Ergebnisse"
            )
        
        # UPDATED: Metadata moved to bottom in collapsible accordion
        with gr.Accordion("Ausgabe-Metadaten", open=False):
            # Add date information
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gr.Markdown(f"**Erstellt am**: {current_date}")
            
            metadata_output = gr.Markdown(
                value="Detaillierte Metadaten zur Analyse erscheinen hier...",
                elem_classes=["metadata-section"]
            )
    
    # Define all components to be returned
    components = {
        "answer_output": answer_output,
        "metadata_output": metadata_output
    }
    
    return components

def format_analysis_result(answer: str, metadata: Dict[str, Any], user_prompt: str, model: str) -> tuple:
    """
    Format analysis results with enhanced presentation.
    
    Args:
        answer: The LLM-generated answer
        metadata: Analysis metadata
        user_prompt: The user's question
        model: Model used for analysis
        
    Returns:
        Tuple of (formatted_answer, formatted_metadata)
    """
    # UPDATED: Enhanced answer formatting with new styling
    formatted_answer = f"""# Analyse-Ergebnisse

## Forschungsfrage
> {user_prompt}

## Antwort

{answer}

---

*Generiert mit {model} • Basierend auf ausgewählten historischen Quellen aus dem SPIEGEL-Archiv (1948-1979)*
"""
    
    # UPDATED: Enhanced metadata formatting
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