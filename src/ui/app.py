# src/ui/app.py - Fixed initialization for handlers
"""
Updated app initialization to properly connect handlers with the RAG engine
"""
import gradio as gr
import logging
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.engine import SpiegelRAG
from src.ui.handlers.search_handlers import set_rag_engine
from src.ui.components.search_panel import create_search_panel
from src.ui.components.question_panel import create_question_panel
from src.ui.components.results_panel import create_results_panel
from src.ui.components.keyword_analysis_panel import create_keyword_analysis_panel
from src.ui.components.info_panel import create_info_panel
from src.ui.handlers.search_handlers import (
    perform_retrieval_and_update_ui,
    perform_analysis_and_update_ui
)
from src.ui.handlers.keyword_handlers import (
    set_embedding_service,
    find_similar_words,
    expand_boolean_expression
)
from src.ui.utils.ui_helpers import toggle_api_key_visibility
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create the main Gradio application with proper handler integration."""
    
    # Initialize RAG engine
    logger.info("Initializing RAG engine...")
    try:
        rag_engine = SpiegelRAG()
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise
    
    # Set the engine reference for handlers
    set_rag_engine(rag_engine)
    
    # Set embedding service for keyword handlers
    if rag_engine.embedding_service:
        set_embedding_service(rag_engine.embedding_service)
        logger.info("Embedding service connected to keyword handlers")
    else:
        logger.warning("Embedding service not available for keyword analysis")
    
    # Create the Gradio interface
    with gr.Blocks(
        title="Der Spiegel RAG System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # Der Spiegel RAG System (1948-1979)
        
        Ein Retrieval Augmented Generation (RAG) System zur Analyse und Durchsuchung des Spiegel-Archivs.
        
        **Systemstatus:** ✅ Verbunden mit ChromaDB und Ollama Embedding Service
        """)
        
        with gr.Tab("1. Quellen abrufen"):
            with gr.Accordion("Inhalt suchen", open=True) as retrieval_accordion:
                # Create search panel components
                search_components = create_search_panel(
                    retrieve_callback=None,  # Will be set below
                    analyze_callback=None,   # Not used in this tab
                    preview_callback=expand_boolean_expression,
                    toggle_api_key_callback=toggle_api_key_visibility
                )
            
            with gr.Accordion("Gefundene Texte", open=False) as retrieved_texts_accordion:
                retrieved_chunks_display = gr.Markdown("Die gefundenen Texte werden hier angezeigt...")
        
        with gr.Tab("2. Quellen analysieren"):
            with gr.Accordion("Frage stellen", open=True) as question_accordion:
                # Create question panel components
                question_components = create_question_panel()
            
            with gr.Accordion("Ergebnisse", open=False) as results_accordion:
                # Create results panel components
                results_components = create_results_panel()
        
        with gr.Tab("3. Agenten-Suche"):
            # Import agent components
            from src.ui.components.agent_panel import create_agent_panel
            from src.ui.components.agent_results_panel import create_agent_results_panel
            from src.ui.handlers.agent_handlers import (
                set_rag_engine as set_agent_rag_engine,
                perform_agent_search_and_analysis
            )
            
            # Set RAG engine for agent handlers too
            set_agent_rag_engine(rag_engine)
            
            # Create agent panel components
            agent_components = create_agent_panel(
                agent_search_callback=None,  # Will be set below
                toggle_api_key_callback=toggle_api_key_visibility
            )
            
            # Create agent results panel
            agent_results_components = create_agent_results_panel()
        
        with gr.Tab("Schlagwort-Analyse"):
            # Create keyword analysis panel
            keyword_components = create_keyword_analysis_panel(
                find_similar_words_callback=find_similar_words
            )
        
        with gr.Tab("Info"):
            create_info_panel()
        
        # Connect event handlers
        
        # Retrieval button click
        search_components["retrieve_btn"].click(
            perform_retrieval_and_update_ui,
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
                search_components["time_window_size"],
                search_components["top_k"]
            ],
            outputs=[
                search_components["retrieved_info"],
                search_components["retrieved_chunks_state"],
                retrieved_chunks_display,
                retrieval_accordion,
                retrieved_texts_accordion,
                question_accordion
            ]
        )
        
        # Analysis button click
        question_components["analyze_btn"].click(
            perform_analysis_and_update_ui,
            inputs=[
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
                results_components["answer_output"],
                results_components["metadata_output"],
                question_accordion,
                results_accordion
            ]
        )
        
        # Agent search button click
        agent_components["agent_search_btn"].click(
            perform_agent_search_and_analysis,
            inputs=[
                agent_components["agent_question"],
                agent_components["agent_content_description"],
                agent_components["agent_year_start"],
                agent_components["agent_year_end"],
                agent_components["agent_chunk_size"],
                agent_components["agent_keywords"],
                agent_components["agent_search_in"],
                agent_components["agent_enforce_keywords"],
                agent_components["agent_initial_count"],
                agent_components["agent_filter_stage1"],
                agent_components["agent_filter_stage2"],
                agent_components["agent_filter_stage3"],
                agent_components["agent_model"],
                agent_components["agent_openai_api_key"],
                agent_components["agent_system_prompt_template"],
                agent_components["agent_custom_system_prompt"]
            ],
            outputs=[
                agent_components["agent_results_state"],
                agent_components["agent_status"],
                agent_results_components["agent_answer_output"],
                agent_results_components["agent_process_output"],
                agent_results_components["agent_evaluations_output"],
                agent_results_components["agent_chunks_output"],
                agent_results_components["agent_metadata_output"]
            ]
        )
        
        logger.info("Gradio interface created successfully")
    
    return app

def main():
    """Main entry point for the application."""
    try:
        app = create_app()
        logger.info("Starting Gradio application...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()