# src/ui/app.py
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
from src.ui.components.info_panel import create_info_panel
from src.ui.handlers.search_handlers import (
    perform_retrieval_and_update_ui,
    perform_analysis_and_update_ui
)
from src.ui.handlers.agent_handlers import (
    set_rag_engine as set_llm_assisted_rag_engine,
    perform_llm_assisted_search_threaded,
    cancel_llm_assisted_search,
    create_llm_assisted_download_comprehensive
)
from src.ui.handlers.keyword_handlers import (
    set_embedding_service,
    find_similar_words,
    expand_boolean_expression
)
from src.ui.handlers.download_handlers import (
    create_download_json, 
    create_download_csv,
    format_download_summary
)
from src.ui.utils.ui_helpers import toggle_api_key_visibility
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create the updated Gradio application with new terminology and design."""
    
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
    set_llm_assisted_rag_engine(rag_engine)
    
    # Set embedding service for keyword handlers
    if rag_engine.embedding_service:
        set_embedding_service(rag_engine.embedding_service)
        logger.info("Embedding service connected to keyword handlers")
    else:
        logger.warning("Embedding service not available")
    
    # UPDATED: CSS with new color scheme and improved styling
    updated_css = """
    /* Main container styling */
    .gradio-container {
        max-width: 1400px !important;
    }
    
    .search-mode-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #968d84 !important;  /* NEW: Updated border color */
        margin-bottom: 20px !important;
    }
    
    .chunk-selection-container {
        background: linear-gradient(135deg, #f4f1ee 0%, #faf8f6 100%) !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #968d84 !important;  /* NEW: Updated color */
        margin: 10px 0 !important;
    }
    
    .chunk-selection-container h4 {
        color: #5a5248 !important;  /* NEW: Darker shade of #968d84 */
        margin-bottom: 10px !important;
        font-weight: bold !important;
    }
    
    .interval-info {
        background: #fef7f0 !important;  /* NEW: Light shade of #d75425 */
        padding: 10px !important;
        border-radius: 6px !important;
        border-left: 4px solid #d75425 !important;  /* NEW: Orange accent */
        margin: 10px 0 !important;
    }
    
    .llm-assisted-prompt-container {
        background: #f9f8f4 !important;  /* NEW: Light shade of #b2b069 */
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #b2b069 !important;  /* NEW: Yellow-green accent */
        margin: 10px 0 !important;
    }
    
    .llm-assisted-prompt-container h4 {
        color: #6b6840 !important;  /* NEW: Darker shade of #b2b069 */
        margin-bottom: 10px !important;
        font-weight: bold !important;
    }
    
    .label-wrap {
        background: linear-gradient(135deg, #968d84 0%, #b2b069 100%) !important;  /* NEW: Custom gradient */
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
        background: linear-gradient(135deg, #7d7469 0%, #9d9d5c 100%) !important;  /* NEW: Darker on hover */
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
        border: 2px solid #5a5248 !important;
    }
    
    .label-wrap.open {
        background: linear-gradient(135deg, #d75425 0%, #b85820 100%) !important;  /* NEW: Orange when open */
        border: 2px solid #a0471c !important;
    }
    
    /* UPDATED: Primary button styling with new color scheme */
    .btn-primary {
        background: linear-gradient(135deg, #d75425 0%, #c04a20 100%) !important;  /* NEW: Orange gradient */
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    
    .btn-primary:hover {
        background: linear-gradient(135deg, #c04a20 0%, #a63f1b 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* UPDATED: Secondary button styling */
    .btn-secondary {
        background: linear-gradient(135deg, #968d84 0%, #857c73 100%) !important;  /* NEW: Gray gradient */
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        border: none !important;
        cursor: pointer !important;
    }
    
    .btn-secondary:hover {
        background: linear-gradient(135deg, #857c73 0%, #756c63 100%) !important;
    }
    
    /* UPDATED: Download button styling with new colors */
    .download-button {
        background: linear-gradient(135deg, #b2b069 0%, #a0a05c 100%) !important;  /* NEW: Yellow-green */
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .download-button:hover {
        background: linear-gradient(135deg, #a0a05c 0%, #8f8f52 100%) !important;
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
    
    /* UPDATED: Progress indicators with new colors */
    .llm-assisted-progress {
        background-color: #f9f8f4 !important;  /* NEW: Light yellow-green */
        padding: 15px !important;
        border-radius: 8px !important;
        border-left: 4px solid #b2b069 !important;  /* NEW: Yellow-green accent */
        margin: 10px 0 !important;
    }
    
    .llm-assisted-progress h4 {
        color: #6b6840 !important;  /* NEW: Darker shade */
        margin-bottom: 10px !important;
    }
    
    /* Results container styling */
    .results-container {
        padding: 20px !important;
        border-radius: 8px !important;
        border: 1px solid #968d84 !important;  /* NEW: Updated border */
    }
    
    /* UPDATED: Tab styling to distinguish from buttons */
    .tab-nav {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border: 1px solid #968d84 !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .tab-nav button {
        background: transparent !important;
        color: #5a5248 !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border: none !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(135deg, #d75425 0%, #c04a20 100%) !important;
        color: white !important;
    }
    
    /* Improved form styling */
    .form-container {
        background: #fafafa !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #968d84 !important;
        margin: 10px 0 !important;
    }
    
    /* Better text contrast for accessibility */
    .dark-text {
        color: #2c3e50 !important;
    }
    
    /* Chunk selection checkboxes */
    .chunk-checkbox {
        margin-right: 8px !important;
        transform: scale(1.2) !important;
    }
    
    .chunk-item {
        padding: 10px !important;
        margin: 5px 0 !important;
        border: 1px solid #ddd !important;
        border-radius: 5px !important;
        background: #fafafa !important;
        transition: background-color 0.2s ease !important;
    }
    
    .chunk-item:hover {
        background: #f0f0f0 !important;
    }
    
    .chunk-item.selected {
        background: #f9f8f4 !important;
        border-color: #b2b069 !important;
    }
    """
    
    # Create the Gradio interface
    with gr.Blocks(
        title="SPIEGEL RAG System",  # UPDATED: Simplified title
        theme=gr.themes.Soft(),
        css=updated_css
    ) as app:
        
        # UPDATED: Main header with new terminology
        gr.Markdown("""
        # SPIEGEL RAG System (1948-1979)
        
        Ein Retrieval Augmented Generation (RAG) System zur Analyse und Durchsuchung des Spiegel-Archivs.
        
        **Systemstatus:** ✅ Verbunden mit ChromaDB und Ollama Embedding Service
        
        **Neue Features:**
        - **Heuristik**: Getrennte Retrieval- und Analyse-Phasen für bessere Kontrolle
        - **LLM-Unterstützte Auswahl**: KI-gestützte Quellenbewertung mit anpassbaren Prompts
        - **Zeit-Interval-Suche**: Gleichmäßige zeitliche Verteilung der Quellen
        - **Quellenauswahl**: Interaktive Auswahl der zu analysierenden Texte
        """)
        
        # UPDATED: Tab structure with new names
        with gr.Tab("Heuristik"):  # UPDATED: Changed from "Quellen abrufen"
            with gr.Accordion("Suchmethode wählen", open=True) as search_method_accordion:
                # Create updated search panel components
                search_components = create_search_panel(
                    retrieve_callback=None,  # Will be set below
                    llm_assisted_search_callback=None,  # Will be set below
                    preview_callback=expand_boolean_expression,
                    toggle_api_key_callback=toggle_api_key_visibility
                )
            
            with gr.Accordion("Gefundene Texte", open=False) as retrieved_texts_accordion:
                retrieved_chunks_display = gr.Markdown("Die gefundenen Texte werden hier angezeigt...")
                
                # UPDATED: Download functionality without template option
                with gr.Row():
                    download_json_btn = gr.Button("📥 Als JSON herunterladen", elem_classes=["download-button"])
                    download_csv_btn = gr.Button("📊 Als CSV herunterladen", elem_classes=["download-button"])
                    
                    # LLM-assisted-specific comprehensive download (conditionally visible)
                    download_comprehensive_btn = gr.Button(
                        "📋 Umfassender LLM-Download", 
                        elem_classes=["download-button"],
                        visible=False
                    )
                
                # Download status and files
                download_status = gr.Markdown("", visible=False)
                download_json_file = gr.File(label="JSON Download", visible=False)
                download_csv_file = gr.File(label="CSV Download", visible=False)
                download_comprehensive_file = gr.File(label="Comprehensive LLM Download", visible=False)
        
        with gr.Tab("Analyse"):  # UPDATED: Changed from "Quellen analysieren"
            with gr.Accordion("Analyse konfigurieren", open=True) as analysis_accordion:
                # Create updated question panel components
                question_components = create_question_panel()
            
            with gr.Accordion("Ergebnisse", open=False) as results_accordion:
                # Create results panel components
                results_components = create_results_panel()
                
                # NEW: Download analysis results as TXT
                with gr.Row():
                    download_analysis_btn = gr.Button("📄 Analyse als TXT herunterladen", elem_classes=["download-button"])
                    download_analysis_file = gr.File(label="Analysis TXT Download", visible=False)
        
        # REMOVED: Schlagwort-Analyse tab (as requested)
        
        with gr.Tab("Info"):
            create_info_panel()
        
        # Connect event handlers
        
        # UPDATED: Standard search button click
        search_components["standard_search_btn"].click(
            perform_retrieval_and_update_ui,
            inputs=[
                search_components["retrieval_query"], 
                search_components["chunk_size"],
                search_components["year_start"],
                search_components["year_end"],
                search_components["keywords"],
                search_components["search_in"],
                search_components["use_semantic_expansion"],
                search_components["semantic_expansion_factor"],
                search_components["expanded_words_state"],
                search_components["use_time_intervals"], 
                search_components["time_interval_size"], 
                search_components["top_k"],
                search_components["chunks_per_interval"]  
            ],
            outputs=[
                search_components["search_status"],
                search_components["retrieved_chunks_state"],
                retrieved_chunks_display,
                search_method_accordion,
                retrieved_texts_accordion,
                analysis_accordion
            ]
        ).then(
            lambda: gr.update(visible=False),  # Hide comprehensive download for standard
            outputs=[download_comprehensive_btn]
        ).then(
            # NEW: Update chunk selection display
            question_components["update_chunk_selection_display"],
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[
                question_components["chunk_selection_area"],
                question_components["selection_summary"],
                question_components["transfer_selection_btn"],
                question_components["selected_chunks_state"]
            ]
        )
        
        # UPDATED: LLM-assisted search button click
        search_components["llm_assisted_search_btn"].click(
            perform_llm_assisted_search_threaded,
            inputs=[
                search_components["retrieval_query"],  # UPDATED: from content_description
                search_components["chunk_size"],
                search_components["year_start"],
                search_components["year_end"],
                search_components["llm_assisted_use_time_intervals"], 
                search_components["llm_assisted_time_interval_size"],  
                search_components["chunks_per_interval_initial"],  
                search_components["chunks_per_interval_final"],  
                search_components["llm_assisted_min_retrieval_score"],  
                search_components["llm_assisted_keywords"],  
                search_components["llm_assisted_search_in"],  
                gr.State(True),  # enforce_keywords - always true 
                search_components["llm_assisted_model"], 
                search_components["llm_assisted_system_prompt_template"], 
                search_components["llm_assisted_system_prompt_text"] 
            ],
            outputs=[
                search_components["search_status"],
                search_components["retrieved_chunks_state"],
                retrieved_chunks_display,
                search_components["search_mode"],
                search_components["llm_assisted_search_btn"],  # UPDATED: from agent_search_btn
                search_components["llm_assisted_cancel_btn"],  # UPDATED: from agent_cancel_btn
                search_components["llm_assisted_progress"],  # UPDATED: from agent_progress
                retrieved_texts_accordion,
                analysis_accordion
            ]
        ).then(
            lambda: gr.update(visible=True),  # Show comprehensive download for LLM-assisted
            outputs=[download_comprehensive_btn]
        ).then(
            # NEW: Update chunk selection display for LLM-assisted
            question_components["update_chunk_selection_display"],
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[
                question_components["chunk_selection_area"],
                question_components["selection_summary"],
                question_components["transfer_selection_btn"],
                question_components["selected_chunks_state"]
            ]
        )
        
        # LLM-assisted search cancellation
        search_components["llm_assisted_cancel_btn"].click(
            cancel_llm_assisted_search,
            outputs=[search_components["llm_assisted_progress"]]
        )
        
        # UPDATED: Analysis button click
        question_components["analyze_btn"].click(
            perform_analysis_and_update_ui,
            inputs=[
                question_components["user_prompt"],  # UPDATED: from question
                search_components["retrieved_chunks_state"],
                question_components["model_selection"],
                question_components["system_prompt_template"],
                question_components["system_prompt_text"],
                question_components["temperature"],
                gr.State("selected"),  # chunk_selection_mode - using selected chunks
                question_components["selected_chunks_state"]  # selected chunk IDs
            ],
            outputs=[
                results_components["answer_output"],
                results_components["metadata_output"],
                analysis_accordion,
                results_accordion
            ]
        )
        
        # Download handlers (updated but keeping functionality)
        def handle_json_download(retrieved_chunks_state):
            """Handle JSON download with proper status updates."""
            try:
                file_path = create_download_json(retrieved_chunks_state)
                if file_path:
                    has_dual_scores = any(
                        'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
                        for chunk in retrieved_chunks_state.get('chunks', [])
                    ) if retrieved_chunks_state else False
                    
                    summary = format_download_summary(
                        len(retrieved_chunks_state.get('chunks', [])), 
                        "JSON", 
                        has_dual_scores
                    )
                    return (
                        gr.update(value=summary, visible=True),
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
            """Handle CSV download with improved German text encoding."""
            try:
                file_path = create_download_csv(retrieved_chunks_state)
                if file_path:
                    has_dual_scores = any(
                        'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
                        for chunk in retrieved_chunks_state.get('chunks', [])
                    ) if retrieved_chunks_state else False
                    
                    summary = format_download_summary(
                        len(retrieved_chunks_state.get('chunks', [])), 
                        "CSV", 
                        has_dual_scores
                    )
                    return (
                        gr.update(value=summary, visible=True),
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
            """Handle comprehensive LLM-assisted download."""
            try:
                file_path = create_llm_assisted_download_comprehensive(retrieved_chunks_state)
                if file_path:
                    return (
                        gr.update(value="✅ Umfassende LLM-Datei wurde erstellt.", visible=True),
                        gr.update(value=file_path, visible=True)
                    )
                else:
                    return (
                        gr.update(value="❌ Fehler: Keine LLM-Daten zum Herunterladen verfügbar.", visible=True),
                        gr.update(visible=False)
                    )
            except Exception as e:
                logger.error(f"Comprehensive download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der umfassenden Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        # NEW: Analysis TXT download
        def handle_analysis_txt_download(answer_output, metadata_output):
            """Handle analysis results download as TXT."""
            try:
                import tempfile
                from datetime import datetime
                
                # Create TXT content
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_content = f"""SPIEGEL RAG System - Analyse-Ergebnisse
Erstellt am: {timestamp}

{'='*60}
ANALYSE
{'='*60}

{answer_output}

{'='*60}
METADATEN
{'='*60}

{metadata_output}
"""
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix='.txt', 
                    prefix='spiegel_analysis_', 
                    delete=False,
                    encoding='utf-8'
                )
                
                temp_file.write(txt_content)
                temp_file.close()
                
                return (
                    gr.update(value="✅ Analyse als TXT-Datei erstellt.", visible=True),
                    gr.update(value=temp_file.name, visible=True)
                )
                
            except Exception as e:
                logger.error(f"Analysis TXT download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der TXT-Datei: {str(e)}", visible=True),
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
        
        # NEW: Analysis TXT download
        download_analysis_btn.click(
            handle_analysis_txt_download,
            inputs=[results_components["answer_output"], results_components["metadata_output"]],
            outputs=[download_status, download_analysis_file]
        )
        
        logger.info("Gradio interface created successfully with updated terminology and design")
    
    return app

def main():
    """Main entry point for the updated application."""
    try:
        app = create_app()
        logger.info("Starting updated Gradio application with new terminology and design...")
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