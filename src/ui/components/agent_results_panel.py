# src/ui/components/llm_assisted_results_panel.py - Updated with new terminology
"""
Updated LLM-assisted results panel component with new terminology.
UPDATED: Changed from "Agenten-Ergebnisse" to "LLM-Unterstützte Auswahl Ergebnisse"
This component displays the results from the LLM-assisted search with readable text.
"""
import gradio as gr
from typing import Dict, Any

def create_llm_assisted_results_panel() -> Dict[str, Any]:
    """
    Create the LLM-assisted results panel UI components with updated terminology.
    
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        # Enhanced CSS for proper text visibility with new color scheme
        gr.HTML("""
        <style>
            /* UPDATED: Better styling for LLM-assisted results with new color scheme */
            .llm-assisted-results {
                padding: 20px !important;
                border-radius: 8px !important;
                border: 1px solid #968d84 !important;  /* NEW: Updated border color */
                margin-top: 20px !important;
                background-color: #ffffff !important;
            }
            
            /* UPDATED: Evaluation card styling with new color scheme */
            .evaluation-card {
                border-left: 4px solid #d75425 !important;  /* NEW: Orange accent */
                padding: 15px !important;
                margin-bottom: 15px !important;
                background-color: #faf8f6 !important;  /* NEW: Light background */
                border-radius: 8px !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }
            
            /* UPDATED: Ensure all text in evaluation cards is dark and readable */
            .evaluation-card h4 {
                color: #5a5248 !important;  /* NEW: Darker shade of #968d84 */
                margin-bottom: 10px !important;
                font-weight: bold !important;
            }
            
            .evaluation-card p {
                color: #34495e !important;
                line-height: 1.6 !important;
                margin: 8px 0 !important;
            }
            
            .evaluation-card div {
                color: #34495e !important;
                line-height: 1.6 !important;
            }
            
            .evaluation-card strong {
                color: #2c3e50 !important;
                font-weight: 600 !important;
            }
            
            /* UPDATED: Specific styling for different relevance levels with new colors */
            .evaluation-card.high-relevance {
                background-color: #f9f8f4 !important;  /* NEW: Light yellow-green */
                border-left-color: #b2b069 !important;  /* NEW: Yellow-green accent */
            }
            
            .evaluation-card.high-relevance h4,
            .evaluation-card.high-relevance p,
            .evaluation-card.high-relevance div,
            .evaluation-card.high-relevance strong {
                color: #6b6840 !important;  /* NEW: Darker yellow-green */
            }
            
            .evaluation-card.medium-relevance {
                background-color: #fef7f0 !important;  /* NEW: Light orange */
                border-left-color: #d75425 !important;  /* NEW: Orange accent */
            }
            
            .evaluation-card.medium-relevance h4,
            .evaluation-card.medium-relevance p,
            .evaluation-card.medium-relevance div,
            .evaluation-card.medium-relevance strong {
                color: #a0471c !important;  /* NEW: Darker orange */
            }
            
            .evaluation-card.low-relevance {
                background-color: #f4f1ee !important;  /* NEW: Light gray */
                border-left-color: #968d84 !important;  /* NEW: Gray accent */
            }
            
            .evaluation-card.low-relevance h4,
            .evaluation-card.low-relevance p,
            .evaluation-card.low-relevance div,
            .evaluation-card.low-relevance strong {
                color: #5a5248 !important;  /* NEW: Darker gray */
            }
            
            /* UPDATED: Progress visualization with new color scheme */
            .filter-stage {
                margin-bottom: 20px !important;
                background-color: #faf8f6 !important;  /* NEW: Light background */
                padding: 15px !important;
                border-radius: 8px !important;
                border: 1px solid #968d84 !important;  /* NEW: Gray border */
            }
            
            .filter-stage-title {
                font-weight: bold !important;
                margin-bottom: 8px !important;
                color: #5a5248 !important;  /* NEW: Dark gray */
                font-size: 16px !important;
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
                background: linear-gradient(90deg, #d75425 0%, #b2b069 100%) !important;  /* NEW: Custom gradient */
                text-align: center !important;
                color: white !important;
                line-height: 30px !important;
                font-weight: bold !important;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            /* UPDATED: Explanation box with new colors */
            .explanation-box {
                background-color: #f9f8f4 !important;  /* NEW: Light yellow-green */
                padding: 15px !important;
                margin-bottom: 20px !important;
                border-radius: 8px !important;
                border-left: 4px solid #b2b069 !important;  /* NEW: Yellow-green accent */
            }
            
            .explanation-box h4 {
                color: #6b6840 !important;  /* NEW: Darker yellow-green */
                margin-bottom: 10px !important;
                font-weight: bold !important;
            }
            
            .explanation-box ul, .explanation-box li {
                color: #6b6840 !important;  /* NEW: Darker yellow-green */
                line-height: 1.6 !important;
            }
            
            .explanation-box strong {
                color: #5a5248 !important;  /* NEW: Dark gray */
                font-weight: 600 !important;
            }
            
            .explanation-box em {
                color: #6b6840 !important;  /* NEW: Yellow-green */
                font-style: italic !important;
            }
        </style>
        """)
        
        # Answer section - UPDATED terminology
        llm_assisted_answer_output = gr.Markdown(
            value="Die Antwort wird hier angezeigt.",
            label="Antwort der LLM-Unterstützten Auswahl"
        )
        
        # Process visualization - UPDATED terminology
        with gr.Accordion("LLM-Bewertungsprozess", open=True):
            llm_assisted_process_output = gr.HTML(
                value="Der LLM-Bewertungsprozess wird hier visualisiert, wenn die Suche abgeschlossen ist."
            )
        
        # Chunk evaluations - UPDATED terminology
        with gr.Accordion("LLM-Textbewertungen", open=False):
            llm_assisted_evaluations_output = gr.HTML(
                value="Die LLM-Bewertungen der Textabschnitte werden hier angezeigt."
            )
        
        # Retrieved chunks - UPDATED terminology
        with gr.Accordion("Ausgewählte Texte", open=False):
            llm_assisted_chunks_output = gr.Markdown(
                value="Die durch LLM ausgewählten Texte werden hier angezeigt."
            )
            
            # Add download functionality for LLM-assisted results
            with gr.Row():
                llm_assisted_download_json_btn = gr.Button(
                    "📥 LLM-Ergebnisse als JSON", 
                    elem_classes=["download-button"]
                )
            
            llm_assisted_download_json_file = gr.File(visible=False)
        
        # Metadata - UPDATED terminology
        with gr.Accordion("Metadaten", open=False):
            llm_assisted_metadata_output = gr.Markdown(
                value="Metadaten zur LLM-Unterstützten Auswahl werden hier angezeigt."
            )
    
    # Define all components to be returned - UPDATED component names
    components = {
        "llm_assisted_answer_output": llm_assisted_answer_output,
        "llm_assisted_process_output": llm_assisted_process_output,
        "llm_assisted_evaluations_output": llm_assisted_evaluations_output,
        "llm_assisted_chunks_output": llm_assisted_chunks_output,
        "llm_assisted_metadata_output": llm_assisted_metadata_output,
        "llm_assisted_download_json_btn": llm_assisted_download_json_btn,
        "llm_assisted_download_json_file": llm_assisted_download_json_file,
    }
    
    return components