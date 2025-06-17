# src/ui/app.py - Updated with integrated agent search
"""
Enhanced app with redesigned agent search integrated into main search flow
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
from src.ui.handlers.agent_handlers import (
    set_rag_engine as set_agent_rag_engine,
    perform_agent_search_threaded,
    cancel_agent_search,
    create_agent_download_comprehensive
)
from src.ui.handlers.keyword_handlers import (
    set_embedding_service,
    find_similar_words,
    expand_boolean_expression
)
from src.ui.handlers.download_handlers import create_download_json, create_download_csv
from src.ui.utils.ui_helpers import toggle_api_key_visibility
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create the main Gradio application with integrated agent search."""
    
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
    set_agent_rag_engine(rag_engine)
    
    # Set embedding service for keyword handlers
    if rag_engine.embedding_service:
        set_embedding_service(rag_engine.embedding_service)
        logger.info("Embedding service connected to keyword handlers")
    else:
        logger.warning("Embedding service not available for keyword analysis")
    
    # Enhanced CSS
    enhanced_css = """
    /* Main container styling */
    .gradio-container {
        max-width: 1400px !important;
    }
    
    /* Search mode radio styling */
    .search-mode-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #dee2e6 !important;
        margin-bottom: 20px !important;
    }
    
    /* ENHANCED ACCORDION BUTTON STYLING */
    .label-wrap {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        border: 2px solid transparent !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    .label-wrap:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b5b95 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
        border: 2px solid #4a5568 !important;
    }
    
    .label-wrap.open {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
        border: 2px solid #2d7d3b !important;
    }
    
    /* Agent search specific styling */
    .agent-progress {
        background-color: #e3f2fd !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border-left: 4px solid #2196f3 !important;
        margin: 10px 0 !important;
    }
    
    .agent-progress h4 {
        color: #1565c0 !important;
        margin-bottom: 10px !important;
    }
    
    /* Results container styling */
    .results-container {
        padding: 20px !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        background-color: #ffffff !important;
    }
    
    /* Download button styling */
    .download-button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .download-button:hover {
        background: linear-gradient(135deg, #27ae60 0%, #219a52 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Cancel button styling */
    .cancel-button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        border: none !important;
        cursor: pointer !important;
    }
    
    .cancel-button:hover {
        background: linear-gradient(135deg, #c0392b 0%, #a93226 100%) !important;
    }
    """
    
    # Create the Gradio interface
    with gr.Blocks(
        title="Der Spiegel RAG System",
        theme=gr.themes.Soft(),
        css=enhanced_css
    ) as app:
        
        gr.Markdown("""
        # Der Spiegel RAG System (1948-1979)
        
        Ein Retrieval Augmented Generation (RAG) System zur Analyse und Durchsuchung des Spiegel-Archivs.
        
        **Systemstatus:** ✅ Verbunden mit ChromaDB und Ollama Embedding Service
        """)
        
        with gr.Tab("Quellen abrufen"):
            with gr.Accordion("Suchmethode wählen", open=True) as search_method_accordion:
                # Create search panel components with both callbacks
                search_components = create_search_panel(
                    retrieve_callback=None,  # Will be set below
                    agent_search_callback=None,  # Will be set below
                    preview_callback=expand_boolean_expression,
                    toggle_api_key_callback=toggle_api_key_visibility
                )
            
            with gr.Accordion("Gefundene Texte", open=False) as retrieved_texts_accordion:
                retrieved_chunks_display = gr.Markdown("Die gefundenen Texte werden hier angezeigt...")
                
                # Download functionality
                with gr.Row():
                    download_json_btn = gr.Button("📥 Als JSON herunterladen", elem_classes=["download-button"])
                    download_csv_btn = gr.Button("📊 Als CSV herunterladen", elem_classes=["download-button"])
                    
                    # Agent-specific comprehensive download (conditionally visible)
                    download_comprehensive_btn = gr.Button(
                        "📋 Umfassender Agent-Download", 
                        elem_classes=["download-button"],
                        visible=False
                    )
                
                # Download status and files
                download_status = gr.Markdown("", visible=False)
                download_json_file = gr.File(label="JSON Download", visible=False)
                download_csv_file = gr.File(label="CSV Download", visible=False)
                download_comprehensive_file = gr.File(label="Comprehensive Agent Download", visible=False)
        
        with gr.Tab("Quellen analysieren"):
            with gr.Accordion("Frage stellen", open=True) as question_accordion:
                # Create question panel components
                question_components = create_question_panel()
            
            with gr.Accordion("Ergebnisse", open=False) as results_accordion:
                # Create results panel components
                results_components = create_results_panel()
        
        with gr.Tab("Schlagwort-Analyse"):
            # Create keyword analysis panel
            keyword_components = create_keyword_analysis_panel(
                find_similar_words_callback=find_similar_words
            )
        
        with gr.Tab("Info"):
            create_info_panel()
        
        # Connect event handlers
        
        # Helper function to determine which search to use
        def route_search_request(*args):
            """Route search request based on search mode."""
            search_mode = args[0]  # First argument is search mode
            
            if search_mode == "standard":
                # Use standard search (skip search_mode argument)
                return perform_retrieval_and_update_ui(*args[1:])
            else:
                # Use agent search
                return perform_agent_search_threaded(*args[1:])
        
        # Standard search button click
        search_components["standard_search_btn"].click(
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
                search_components["search_status"],
                search_components["retrieved_chunks_state"],
                retrieved_chunks_display,
                search_method_accordion,
                retrieved_texts_accordion,
                question_accordion
            ]
        ).then(
            lambda: gr.update(visible=False),  # Hide comprehensive download for standard
            outputs=[download_comprehensive_btn]
        )
        
        # Agent search button click
        search_components["agent_search_btn"].click(
            perform_agent_search_threaded,
            inputs=[
                search_components["content_description"],
                search_components["chunk_size"],
                search_components["year_start"],
                search_components["year_end"],
                search_components["agent_use_time_windows"],
                search_components["agent_time_window_size"],
                search_components["chunks_per_window_initial"],
                search_components["chunks_per_window_final"],
                search_components["agent_keywords"],
                search_components["agent_search_in"],
                search_components["agent_enforce_keywords"],
                search_components["agent_model"],
                search_components["agent_system_prompt_template"],
                search_components["agent_custom_system_prompt"]
            ],
            outputs=[
                search_components["search_status"],
                search_components["retrieved_chunks_state"],
                retrieved_chunks_display,
                search_components["search_mode"],  # No change
                search_components["agent_search_btn"],
                search_components["agent_cancel_btn"],
                search_components["agent_progress"],
                retrieved_texts_accordion,
                question_accordion
            ]
        ).then(
            lambda: gr.update(visible=True),  # Show comprehensive download for agent
            outputs=[download_comprehensive_btn]
        )
        
        # Agent search cancellation
        search_components["agent_cancel_btn"].click(
            cancel_agent_search,
            outputs=[search_components["agent_progress"]]
        )
        
        # Analysis button click (works for both search types)
        question_components["analyze_btn"].click(
            perform_analysis_and_update_ui,
            inputs=[
                question_components["question"],
                search_components["retrieved_chunks_state"],
                question_components["model_selection"],
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
        
        # Download handlers
        def handle_json_download(retrieved_chunks_state):
            """Handle JSON download with proper status updates."""
            try:
                file_path = create_download_json(retrieved_chunks_state)
                if file_path:
                    return (
                        gr.update(value="✅ JSON-Datei wurde erstellt und kann heruntergeladen werden.", visible=True),
                        gr.update(value=file_path, visible=True)
                    )
                else:
                    return (
                        gr.update(value="❌ Fehler: Keine Daten zum Herunterladen verfügbar.", visible=True),
                        gr.update(visible=False)
                    )
            except Exception as e:
                logger.error(f"JSON download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der JSON-Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        def handle_csv_download(retrieved_chunks_state):
            """Handle CSV download with proper status updates."""
            try:
                file_path = create_download_csv(retrieved_chunks_state)
                if file_path:
                    return (
                        gr.update(value="✅ CSV-Datei wurde erstellt und kann heruntergeladen werden.", visible=True),
                        gr.update(value=file_path, visible=True)
                    )
                else:
                    return (
                        gr.update(value="❌ Fehler: Keine Daten zum Herunterladen verfügbar.", visible=True),
                        gr.update(visible=False)
                    )
            except Exception as e:
                logger.error(f"CSV download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der CSV-Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        def handle_comprehensive_download(retrieved_chunks_state):
            """Handle comprehensive agent download."""
            try:
                file_path = create_agent_download_comprehensive(retrieved_chunks_state)
                if file_path:
                    return (
                        gr.update(value="✅ Umfassende Agent-Datei wurde erstellt.", visible=True),
                        gr.update(value=file_path, visible=True)
                    )
                else:
                    return (
                        gr.update(value="❌ Fehler: Keine Agent-Daten zum Herunterladen verfügbar.", visible=True),
                        gr.update(visible=False)
                    )
            except Exception as e:
                logger.error(f"Comprehensive download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der umfassenden Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        # Connect download events
        download_json_btn.click(
            handle_json_download,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[download_status, download_json_file]
        )
        
        download_csv_btn.click(
            handle_csv_download,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[download_status, download_csv_file]
        )
        
        download_comprehensive_btn.click(
            handle_comprehensive_download,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[download_status, download_comprehensive_file]
        )
        
        logger.info("Enhanced Gradio interface with integrated agent search created successfully")
    
    return app

def main():
    """Main entry point for the application."""
    try:
        app = create_app()
        logger.info("Starting enhanced Gradio application with integrated agent search...")
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