# src/ui/components/results_panel.py
"""
Results panel component for the Spiegel RAG application.
This component defines the UI elements for displaying analysis results.
"""
import gradio as gr
from typing import Dict, Any

def create_results_panel() -> Dict[str, Any]:
    """
    Create the results panel UI components.
    
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        # Main answer output - always visible
        answer_output = gr.Markdown(
            value="Die Antwort erscheint hier...",
            label="Analyse"
        )
        
        # Metadata in collapsible accordion - hidden by default
        with gr.Accordion("Metadaten zur Analyse", open=False):
            metadata_output = gr.Markdown(
                value="Detaillierte Metadaten zur Analyse erscheinen hier..."
            )
    
    # Define all components to be returned
    components = {
        "answer_output": answer_output,
        "metadata_output": metadata_output
    }
    
    return components