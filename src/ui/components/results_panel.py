# src/ui/components/results_panel.py
"""
Results panel component for the Spiegel RAG application.
This component defines the UI elements for displaying search results.
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
        gr.Markdown("## Ergebnisse")
        
        with gr.Tabs():
            with gr.TabItem("Analyse"):
                answer_output = gr.Markdown(
                    value="Die Antwort erscheint hier...",
                    label="Analyse"
                )
            
            with gr.TabItem("Gefundene Texte"):
                chunks_output = gr.Markdown(
                    value="Gefundene Textabschnitte erscheinen hier...",
                    label="Gefundene Texte"
                )
            
            with gr.TabItem("Metadaten"):
                metadata_output = gr.Markdown(
                    value="Metadaten zur Suche erscheinen hier...",
                    label="Metadaten"
                )
    
    # Define all components to be returned
    components = {
        "answer_output": answer_output,
        "chunks_output": chunks_output,
        "metadata_output": metadata_output
    }
    
    return components