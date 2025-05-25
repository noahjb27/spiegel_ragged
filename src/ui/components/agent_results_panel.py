# src/ui/components/agent_results_panel.py
"""
Fixed Agent results panel component with proper text visibility.
This component displays the results from the agent-based search with readable text.
"""
import gradio as gr
from typing import Dict, Any

def create_agent_results_panel() -> Dict[str, Any]:
    """
    Create the agent results panel UI components with fixed text visibility.
    
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        # Enhanced CSS for proper text visibility
        gr.HTML("""
        <style>
            /* FIXED: Better styling for agent results with proper contrast */
            .agent-results {
                padding: 20px !important;
                border-radius: 8px !important;
                border: 1px solid #e0e0e0 !important;
                margin-top: 20px !important;
                background-color: #ffffff !important;
            }
            
            /* FIXED: Evaluation card styling with proper text contrast */
            .evaluation-card {
                border-left: 4px solid #3498db !important;
                padding: 15px !important;
                margin-bottom: 15px !important;
                background-color: #f8f9fa !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }
            
            /* FIXED: Ensure all text in evaluation cards is dark and readable */
            .evaluation-card h4 {
                color: #2c3e50 !important;
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
            
            /* FIXED: Specific styling for different relevance levels */
            .evaluation-card.high-relevance {
                background-color: #f1f8e9 !important;
                border-left-color: #4caf50 !important;
            }
            
            .evaluation-card.high-relevance h4,
            .evaluation-card.high-relevance p,
            .evaluation-card.high-relevance div,
            .evaluation-card.high-relevance strong {
                color: #1b5e20 !important;
            }
            
            .evaluation-card.medium-relevance {
                background-color: #fff8e1 !important;
                border-left-color: #ff9800 !important;
            }
            
            .evaluation-card.medium-relevance h4,
            .evaluation-card.medium-relevance p,
            .evaluation-card.medium-relevance div,
            .evaluation-card.medium-relevance strong {
                color: #e65100 !important;
            }
            
            .evaluation-card.low-relevance {
                background-color: #ffebee !important;
                border-left-color: #f44336 !important;
            }
            
            .evaluation-card.low-relevance h4,
            .evaluation-card.low-relevance p,
            .evaluation-card.low-relevance div,
            .evaluation-card.low-relevance strong {
                color: #b71c1c !important;
            }
            
            /* FIXED: Progress visualization with readable text */
            .filter-stage {
                margin-bottom: 20px !important;
                background-color: #f8f9fa !important;
                padding: 15px !important;
                border-radius: 8px !important;
                border: 1px solid #dee2e6 !important;
            }
            
            .filter-stage-title {
                font-weight: bold !important;
                margin-bottom: 8px !important;
                color: #2c3e50 !important;
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
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
                text-align: center !important;
                color: white !important;
                line-height: 30px !important;
                font-weight: bold !important;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            /* FIXED: Ensure explanation box has proper styling */
            .explanation-box {
                background-color: #e3f2fd !important;
                padding: 15px !important;
                margin-bottom: 20px !important;
                border-radius: 8px !important;
                border-left: 4px solid #2196f3 !important;
            }
            
            .explanation-box h4 {
                color: #1565c0 !important;
                margin-bottom: 10px !important;
                font-weight: bold !important;
            }
            
            .explanation-box ul, .explanation-box li {
                color: #0d47a1 !important;
                line-height: 1.6 !important;
            }
            
            .explanation-box strong {
                color: #0d47a1 !important;
                font-weight: 600 !important;
            }
            
            .explanation-box em {
                color: #1565c0 !important;
                font-style: italic !important;
            }
        </style>
        """)
        
        # Answer section
        agent_answer_output = gr.Markdown(
            value="Die Antwort wird hier angezeigt.",
            label="Antwort des Agenten"
        )
        
        # Process visualization
        with gr.Accordion("Filterungsprozess", open=True):
            agent_process_output = gr.HTML(
                value="Der Filterungsprozess wird hier visualisiert, wenn die Suche abgeschlossen ist."
            )
        
        # Chunk evaluations
        with gr.Accordion("Textbewertungen", open=False):
            agent_evaluations_output = gr.HTML(
                value="Die Bewertungen der Textabschnitte werden hier angezeigt."
            )
        
        # Retrieved chunks
        with gr.Accordion("Gefundene Texte", open=False):
            agent_chunks_output = gr.Markdown(
                value="Die gefundenen Texte werden hier angezeigt."
            )
            
            # Add download functionality for agent results
            with gr.Row():
                agent_download_json_btn = gr.Button("📥 Agenten-Ergebnisse als JSON", elem_classes=["download-button"])
            
            agent_download_json_file = gr.File(visible=False)
        
        # Metadata
        with gr.Accordion("Metadaten", open=False):
            agent_metadata_output = gr.Markdown(
                value="Metadaten zur Suche werden hier angezeigt."
            )
    
    # Define all components to be returned
    components = {
        "agent_answer_output": agent_answer_output,
        "agent_process_output": agent_process_output,
        "agent_evaluations_output": agent_evaluations_output,
        "agent_chunks_output": agent_chunks_output,
        "agent_metadata_output": agent_metadata_output,
        "agent_download_json_btn": agent_download_json_btn,
        "agent_download_json_file": agent_download_json_file
    }
    
    return components