# src/ui/components/agent_results_panel.py
"""
Agent results panel component for the Spiegel RAG application.
This component displays the results from the agent-based search.
"""
import gradio as gr
from typing import Dict, Any

def create_agent_results_panel() -> Dict[str, Any]:
    """
    Create the agent results panel UI components.
    
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        # Add some CSS for better formatting
        gr.HTML("""
        <style>
            /* Better styling for agent results */
            .agent-results {
                padding: 20px !important;
                border-radius: 8px !important;
                border: 1px solid #e0e0e0 !important;
                margin-top: 20px !important;
            }
            
            /* Evaluation card styling */
            .evaluation-card {
                border-left: 4px solid #3498db !important;
                padding: 10px !important;
                margin-bottom: 10px !important;
            }
            
            /* Progress visualization */
            .filter-stage {
                margin-bottom: 20px !important;
            }
            
            .filter-stage-title {
                font-weight: bold !important;
                margin-bottom: 5px !important;
            }
            
            .filter-progress {
                height: 25px !important;
                background-color: #e9ecef !important;
                border-radius: 5px !important;
                overflow: hidden !important;
                margin-bottom: 10px !important;
            }
            
            .filter-bar {
                height: 100% !important;
                background-color: #3498db !important;
                text-align: center !important;
                color: white !important;
                line-height: 25px !important;
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
        "agent_metadata_output": agent_metadata_output
    }
    
    return components