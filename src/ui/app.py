# src/ui/app.py
"""
Main application entry point for the Spiegel RAG UI.
This file initializes the application and assembles the components.
Refactored to support separate retrieval and analysis steps.
"""
import os
import sys
import logging
from typing import Any, Callable, Dict

import gradio as gr

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.rag_engine import SpiegelRAGEngine
from src.config import settings
from src.ui.components.search_panel import create_search_panel
from src.ui.components.results_panel import create_results_panel
from src.ui.components.keyword_analysis_panel import create_keyword_analysis_panel
from src.ui.components.info_panel import create_info_panel
from src.ui.handlers.search_handlers import (
    perform_search_with_keywords, 
    perform_retrieval,
    perform_analysis,
    set_rag_engine
)
from src.ui.handlers.keyword_handlers import find_similar_words, expand_boolean_expression, set_embedding_service
from src.ui.utils.ui_helpers import toggle_api_key_visibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def initialize_services():
    """
    Initialize the RAG engine and embedding service.
    
    Returns:
        Tuple of (rag_engine, embedding_service)
    """
    try:
        logger.info("Initializing RAG Engine...")
        rag_engine = SpiegelRAGEngine()
        embedding_service = rag_engine.embedding_service
        logger.info("RAG Engine and Embedding Service initialized successfully")
        return rag_engine, embedding_service
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        return None, None

def create_app():
    """
    Create and configure the Gradio application.
    
    Returns:
        Gradio Blocks application
    """
    # Initialize services
    rag_engine, embedding_service = initialize_services()
    
    # Set global service references in handler modules
    set_rag_engine(rag_engine)
    set_embedding_service(embedding_service)
    
    # Create Gradio app
    with gr.Blocks(title="Der Spiegel RAG (1948-1979)") as app:
        gr.Markdown(
            """
            # Der Spiegel RAG (1948-1979)
            
            **Ein Retrieval-Augmented Generation System für die Analyse historischer Artikel des Spiegel-Archivs.**
            
            Mit diesem Tool können Sie das Spiegel-Archiv durchsuchen, relevante Inhalte abrufen und 
            KI-gestützte Analysen zu historischen Fragestellungen erhalten.
            """
        )
        
        with gr.Tabs():
            # Main RAG tab with two-step process
            with gr.TabItem("Zweistufige RAG-Suche", id="two_step_search"):
                with gr.Row():
                    # Search panel (left side)
                    with gr.Column(scale=1):
                        search_components = create_search_panel(
                            retrieve_callback=perform_retrieval,
                            analyze_callback=perform_analysis,
                            preview_callback=expand_boolean_expression,
                            toggle_api_key_callback=toggle_api_key_visibility
                        )
                    
                    # Results panel (right side)
                    with gr.Column(scale=1):
                        results_components = create_results_panel()
                
                # Connect retrieve button to retrieval function
                search_components["retrieve_btn"].click(
                    perform_retrieval,
                    inputs=[
                        search_components["content_description"],
                        search_components["chunk_size"],
                        search_components["year_start"],
                        search_components["year_end"],
                        search_components["keywords"],
                        search_components["search_in"],
                        search_components["use_semantic_expansion"],
                        search_components["semantic_expansion_factor"],
                        search_components["expanded_words_state"],
                        search_components["enforce_keywords"],
                        search_components["use_time_windows"],
                        search_components["time_window_size"]
                    ],
                    outputs=[
                        search_components["retrieved_info"],
                        search_components["retrieved_chunks_state"]
                    ]
                )
                
                # Connect analyze button to analysis function
                search_components["analyze_btn"].click(
                    perform_analysis,
                    inputs=[
                        search_components["question"],
                        search_components["retrieved_chunks_state"],
                        search_components["model_selection"],
                        search_components["openai_api_key"]
                    ],
                    outputs=[
                        results_components["answer_output"],
                        results_components["chunks_output"],
                        results_components["metadata_output"]
                    ]
                )
                
                # When successful retrieval happens, switch to analyze tab
                search_components["retrieve_btn"].click(
                    lambda: gr.update(selected="analyze_tab"),
                    outputs=search_components["tabs"]
                )
            
            # Legacy combined search tab (for backward compatibility)
            with gr.TabItem("Einstufige Suche (Legacy)", id="legacy_search"):
                with gr.Row():
                    # Legacy search panel (left side)
                    with gr.Column(scale=1):
                        legacy_search_components = create_legacy_search_panel(
                            search_callback=perform_search_with_keywords,
                            preview_callback=expand_boolean_expression,
                            toggle_api_key_callback=toggle_api_key_visibility
                        )
                    
                    # Legacy results panel (right side)
                    with gr.Column(scale=1):
                        legacy_results_components = create_results_panel()
                
                # Connect search button to search function
                legacy_search_components["search_btn"].click(
                    perform_search_with_keywords,
                    inputs=[
                        legacy_search_components["query"],
                        legacy_search_components["question"],
                        legacy_search_components["chunk_size"],
                        legacy_search_components["year_start"],
                        legacy_search_components["year_end"],
                        legacy_search_components["keywords"],
                        legacy_search_components["search_in"],
                        legacy_search_components["use_semantic_expansion"],
                        legacy_search_components["semantic_expansion_factor"],
                        legacy_search_components["expanded_words_state"],
                        legacy_search_components["enforce_keywords"],
                        legacy_search_components["use_time_windows"],
                        legacy_search_components["time_window_size"],
                        legacy_search_components["model_selection"],
                        legacy_search_components["openai_api_key"]
                    ],
                    outputs=[
                        legacy_results_components["answer_output"],
                        legacy_results_components["chunks_output"],
                        legacy_results_components["metadata_output"]
                    ]
                )
            
            # Keyword analysis tab
            with gr.TabItem("Schlagwort-Analyse", id="keyword_analysis"):
                keyword_components = create_keyword_analysis_panel(
                    find_similar_words_callback=find_similar_words
                )
            
            # Info tab
            with gr.TabItem("Info", id="info"):
                create_info_panel()
    
    return app

def create_legacy_search_panel(
    search_callback: Callable,
    preview_callback: Callable,
    toggle_api_key_callback: Callable
) -> Dict[str, Any]:
    """
    Create the legacy search panel UI components (for backward compatibility).
    
    This is a simplified version of the original search panel function, kept for
    backward compatibility. It creates a single-step search interface.
    
    Args:
        search_callback: Function to call when search button is clicked
        preview_callback: Function to call when preview button is clicked
        toggle_api_key_callback: Function to toggle API key visibility
        
    Returns:
        Dictionary of UI components
    """
    # Hidden state for expanded words
    expanded_words_state = gr.State("")
    
    # Main search panel
    with gr.Group():
        gr.Markdown("## Suchanfrage")
        
        query = gr.Textbox(
            label="Suchanfrage (welche Inhalte gesucht werden sollen)",
            placeholder="Beispiel: Berichterstattung über die Berliner Mauer",
            lines=2,
            value="Berlin Mauer",
            info="Beschreiben Sie, welche Art von Inhalten Sie im Archiv finden möchten."
        )
        
        question = gr.Textbox(
            label="Frage (was Sie über die gefundenen Inhalte wissen möchten)",
            placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
            lines=2,
            value="Wie wurde die Berliner Mauer in den Medien dargestellt?",
            info="Formulieren Sie Ihre Frage, die anhand der gefundenen Texte beantwortet werden soll."
        )
    
    # Basic settings in an accordion
    with gr.Accordion("Grundeinstellungen", open=True):
        with gr.Row():
            year_start = gr.Slider(
                minimum=settings.MIN_YEAR,
                maximum=settings.MAX_YEAR,
                value=1960,
                step=1,
                label="Startjahr",
                info="Beginn des zu durchsuchenden Zeitraums"
            )
            
            year_end = gr.Slider(
                minimum=settings.MIN_YEAR,
                maximum=settings.MAX_YEAR,
                value=1970,
                step=1,
                label="Endjahr",
                info="Ende des zu durchsuchenden Zeitraums"
            )
        
        with gr.Row():
            chunk_size = gr.Dropdown(
                choices=[2000, 3000],
                value=3000,
                label="Textgröße",
                info="Größe der Textabschnitte in Zeichen. Kleinere Abschnitte sind präziser, größere bieten mehr Kontext."
            )
    
    # Keyword filtering options
    with gr.Accordion("Schlagwort-Filterung", open=False):
        gr.Markdown("""
        ### Schlagwort-Filterung
        
        Filtern Sie die Suchergebnisse nach bestimmten Schlagwörtern. Sie können auch boolesche Ausdrücke verwenden (AND, OR, NOT).
        
        **Beispiele:**
        - `mauer` - Findet Texte, die "mauer" enthalten
        - `berlin AND mauer` - Findet Texte, die sowohl "berlin" als auch "mauer" enthalten
        - `berlin AND (mauer OR grenze) NOT sowjet` - Komplexere Ausdrücke sind möglich
        """)
        
        keywords = gr.Textbox(
            label="Schlagwörter (boolescher Ausdruck)",
            placeholder="berlin AND mauer",
            lines=2
        )
        
        with gr.Row():
            search_in = gr.CheckboxGroup(
                choices=["Text", "Artikeltitel", "Schlagworte"],
                value=["Text"],
                label="Suche in"
            )
            
            enforce_keywords = gr.Checkbox(
                label="Strikte Filterung",
                value=True,
                info="Wenn aktiviert, werden nur Texte angezeigt, die die angegebenen Schlagwörter enthalten."
            )
    
        with gr.Row():
            use_semantic_expansion = gr.Checkbox(
                label="Semantische Erweiterung",
                value=True,
                info="Findet und berücksichtigt auch semantisch ähnliche Wörter"
            )
            
            semantic_expansion_factor = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Anzahl ähnlicher Wörter"
            )
        
        with gr.Row():
            preview_btn = gr.Button("Vorschau ähnlicher Wörter")
        
        expansion_output = gr.Markdown(label="Ähnliche Wörter")
    
    # Time window search
    with gr.Accordion("Zeitfenster-Suche", open=False):
        gr.Markdown("""
        ### Zeitfenster-Suche
        
        Die Zeitfenster-Suche unterteilt den Suchzeitraum in kleinere Abschnitte und sorgt dafür, 
        dass Ergebnisse aus verschiedenen Zeitperioden berücksichtigt werden.
        """)
        
        with gr.Row():
            use_time_windows = gr.Checkbox(
                label="Zeitfenster-Suche aktivieren",
                value=False,
                info="Sucht in definierten Zeitfenstern anstatt im gesamten Zeitraum"
            )
            
            time_window_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Fenstergröße (Jahre)",
                info="Größe der einzelnen Zeitfenster in Jahren"
            )
    
    search_btn = gr.Button("Suchen", variant="primary")

    # Model selection settings
    with gr.Accordion("Erweiterte Einstellungen", open=False):
        gr.Markdown("""
        ### Modellauswahl
        
        Sie können zwischen verschiedenen LLM-Modellen wählen:
        - **HU-LLM**: Lokales Modell 
                    (kein API-Schlüssel erforderlich, HU-Netzwerk erforderlich)
        - **OpenAI GPT-4o**: Leistungsstärkstes OpenAI-Modell 
                    (erfordert API-Schlüssel)
        - **OpenAI GPT-3.5 Turbo**: Schnelles OpenAI-Modell 
                    (erfordert API-Schlüssel)
        """)
        
        with gr.Row():
            model_selection = gr.Radio(
                choices=["hu-llm", "openai-gpt4o", "openai-gpt35"],
                value="hu-llm",
                label="LLM-Modell",
                info="Wählen Sie das zu verwendende Sprachmodell"
            )
        
        with gr.Row(visible=False) as openai_key_row:
            openai_api_key = gr.Textbox(
                label="OpenAI API-Schlüssel",
                placeholder="sk-...",
                type="password",
                info="Ihr OpenAI API-Schlüssel wird nur für diese Sitzung gespeichert"
            )

    # Connect events
    model_selection.change(
        toggle_api_key_callback,
        inputs=[model_selection],
        outputs=[openai_key_row]
    )
    
    preview_btn.click(
        preview_callback,
        inputs=[keywords, semantic_expansion_factor],
        outputs=[expansion_output, expanded_words_state]
    )
    
    # Define all components to be returned
    components = {
        "query": query,
        "question": question,
        "chunk_size": chunk_size,
        "year_start": year_start,
        "year_end": year_end,
        "keywords": keywords,
        "search_in": search_in,
        "use_semantic_expansion": use_semantic_expansion,
        "semantic_expansion_factor": semantic_expansion_factor,
        "expanded_words_state": expanded_words_state,
        "enforce_keywords": enforce_keywords,
        "use_time_windows": use_time_windows,
        "time_window_size": time_window_size,
        "model_selection": model_selection,
        "openai_api_key": openai_api_key,
        "search_btn": search_btn,
        "expansion_output": expansion_output,
        "openai_key_row": openai_key_row
    }
    
    return components

# Run the app
if __name__ == "__main__":
    logger.info("Starting Spiegel RAG UI...")
    app = create_app()
    app.launch(share=False)