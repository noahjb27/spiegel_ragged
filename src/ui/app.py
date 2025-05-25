# src/ui/app.py
"""
Enhanced app with improved button styling, download functionality, and fixed text visibility
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
from src.ui.handlers.download_handlers import create_download_json, create_download_csv  # New import
from src.ui.utils.ui_helpers import toggle_api_key_visibility
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create the main Gradio application with enhanced styling and download functionality."""
    
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
    
    # Enhanced CSS with better button styling and fixed text visibility
    enhanced_css = """
    /* Main container styling */
    .gradio-container {
        max-width: 1200px !important;
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
    
    /* Icon styling within buttons */
    .label-wrap .icon {
        font-weight: bold !important;
        font-size: 18px !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Button text styling */
    .label-wrap span:first-child {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        letter-spacing: 0.5px !important;
    }
    
    /* FIXED TEXT VISIBILITY FOR EVALUATIONS */
    .evaluation-card {
        border-left: 4px solid #3498db !important;
        padding: 15px !important;
        margin-bottom: 15px !important;
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    .evaluation-card h4 {
        color: #2c3e50 !important;
        margin-bottom: 10px !important;
        font-weight: bold !important;
    }
    
    .evaluation-card p, .evaluation-card div {
        color: #34495e !important;
        line-height: 1.6 !important;
    }
    
    .evaluation-card strong {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* High relevance styling */
    .evaluation-card.high-relevance {
        background-color: #f1f8e9 !important;
        border-left-color: #4caf50 !important;
    }
    
    /* Medium relevance styling */
    .evaluation-card.medium-relevance {
        background-color: #fff8e1 !important;
        border-left-color: #ff9800 !important;
    }
    
    /* Lower relevance styling */
    .evaluation-card.low-relevance {
        background-color: #ffebee !important;
        border-left-color: #f44336 !important;
    }
    
    /* Progress visualization */
    .filter-stage {
        margin-bottom: 20px !important;
        background-color: #f8f9fa !important;
        padding: 10px !important;
        border-radius: 8px !important;
    }
    
    .filter-stage-title {
        font-weight: bold !important;
        margin-bottom: 8px !important;
        color: #2c3e50 !important;
    }
    
    .filter-progress {
        height: 30px !important;
        background-color: #e9ecef !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        margin-bottom: 5px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    .filter-bar {
        height: 100% !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        text-align: center !important;
        color: white !important;
        line-height: 30px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
        transition: width 0.5s ease !important;
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
    
    /* Results container styling */
    .results-container {
        padding: 20px !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        background-color: #ffffff !important;
    }
    
    /* Better typography */
    .results-container p, .results-container li {
        font-size: 16px !important;
        line-height: 1.6 !important;
        color: #2c3e50 !important;
    }
    
    /* Better heading styles */
    .results-container h1, .results-container h2, .results-container h3 {
        margin-top: 1em !important;
        margin-bottom: 0.5em !important;
        color: #2c3e50 !important;
    }
    
    /* Quote styling */
    .results-container blockquote {
        border-left: 4px solid #667eea !important;
        padding-left: 1em !important;
        margin-left: 0 !important;
        font-style: italic !important;
        background-color: #f8f9fa !important;
        padding: 10px 15px !important;
        border-radius: 0 6px 6px 0 !important;
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
                
                # ADD DOWNLOAD FUNCTIONALITY
                with gr.Row():
                    download_json_btn = gr.Button("📥 Als JSON herunterladen", elem_classes=["download-button"])
                    download_csv_btn = gr.Button("📊 Als CSV herunterladen", elem_classes=["download-button"])
                
                # Download status and files - FIXED
                download_status = gr.Markdown("", visible=False)
                download_json_file = gr.File(label="JSON Download", visible=False)
                download_csv_file = gr.File(label="CSV Download", visible=False)
        
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
        
        # JSON Download
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
        
        # CSV Download  
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
        
        # Connect download events - FIXED
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
        
        logger.info("Enhanced Gradio interface created successfully")
    
    return app

def main():
    """Main entry point for the application."""
    try:
        app = create_app()
        logger.info("Starting enhanced Gradio application...")
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