# src/ui/components/results_panel.py
"""
Results panel component for the Spiegel RAG application.
This component defines the UI elements for displaying analysis results.
"""
import gradio as gr
from typing import Dict, Any

def create_results_panel() -> Dict[str, Any]:
    """
    Create the results panel UI components with improved styling.
    """
    with gr.Group():
        # Add some CSS for better formatting
        gr.HTML("""
        <style>
            /* Improve results section styling */
            .results-container {
                padding: 20px !important;
                border-radius: 8px !important;
                border: 1px solid #e0e0e0 !important;
            }
            
            /* Better typography */
            .results-container p, .results-container li {
                font-size: 16px !important;
                line-height: 1.6 !important;
            }
            
            /* Better heading styles */
            .results-container h1, .results-container h2, .results-container h3 {
                margin-top: 1em !important;
                margin-bottom: 0.5em !important;
                color: #1f1f1f !important;
            }
            
            /* Better list styling */
            .results-container ul, .results-container ol {
                padding-left: 2em !important;
                margin-bottom: 1em !important;
            }
            
            /* Quote styling */
            .results-container blockquote {
                border-left: 4px solid #b0b0b0 !important;
                padding-left: 1em !important;
                margin-left: 0 !important;
                font-style: italic !important;
            }
        </style>
        """)
        
        # Main answer output with custom class for styling
        with gr.Column(elem_classes=["results-container"]):
            answer_output = gr.Markdown(
            value="Die Antwort erscheint hier...",
            label="Analyse"
            )
        
        # Metadata in collapsible accordion
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