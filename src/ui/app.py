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
    perform_analysis_and_update_ui,
    perform_retrieval_and_update_ui,
    perform_retrieval,
    perform_analysis,
    set_rag_engine
)
from src.ui.components.question_panel import create_question_panel

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
        
        # Main RAG interface with accordion-based layout
        with gr.Tab("RAG-Suche"):
            # Search panel (top section - always visible)
            with gr.Accordion("1. Quellen abrufen", open=True) as retrieval_accordion:
                search_components = create_search_panel(
                    retrieve_callback=perform_retrieval,
                    analyze_callback=perform_analysis,
                    preview_callback=expand_boolean_expression,
                    toggle_api_key_callback=toggle_api_key_visibility
                )
            
            # Retrieved texts section (shows after retrieval)
            with gr.Accordion("Gefundene Texte", open=False) as retrieved_texts_accordion:
                retrieved_texts_output = gr.Markdown("Noch keine Texte abgerufen.")
            
            # Question section (middle - becomes visible after retrieval)
            with gr.Accordion("2. Quellen analysieren", open=False) as question_accordion:
                question_components = create_question_panel()
                
            # Results section (bottom - expands after analysis)
            with gr.Accordion("Ergebnisse", open=False) as results_accordion:
                results_components = create_results_panel()
            

    
        # Connect search button to control accordion states
        search_components["retrieve_btn"].click(
            fn=perform_retrieval_and_update_ui,
            inputs=[
                # Retrieval parameters
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
                search_components["time_window_size"],
                search_components["top_k"]
            ],
            outputs=[
                # Results and UI state updates
                search_components["retrieved_info"],
                search_components["retrieved_chunks_state"],
                retrieved_texts_output,
                retrieval_accordion,
                retrieved_texts_accordion,
                question_accordion
            ]
        )
        
        # Connect analyze button to results and update UI
        question_components["analyze_btn"].click(
            fn=perform_analysis_and_update_ui,
            inputs=[
                # Analysis parameters
                question_components["question"],
                search_components["retrieved_chunks_state"],
                question_components["model_selection"],
                question_components["openai_api_key"],
                question_components["system_prompt_template"],
                question_components["custom_system_prompt"],
                question_components["temperature"],
                question_components["max_tokens"]
            ],
            outputs=[
                # Results and UI state updates
                results_components["answer_output"],
                results_components["metadata_output"],
                question_accordion,
                results_accordion
            ]
        )

        # Other tabs remain the same
        with gr.Tab("Schlagwort-Analyse"):
            keyword_components = create_keyword_analysis_panel(
                find_similar_words_callback=find_similar_words
            )
        
        with gr.Tab("Info"):
            create_info_panel()
    
    return app

# Run the app
if __name__ == "__main__":
    logger.info("Starting Spiegel RAG UI...")
    app = create_app()
    app.launch(share=False)