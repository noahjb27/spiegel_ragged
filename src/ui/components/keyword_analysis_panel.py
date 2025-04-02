# src/ui/components/keyword_analysis_panel.py
"""
Keyword analysis panel component for the Spiegel RAG application.
This component defines the UI elements for keyword analysis.
"""
import gradio as gr
from typing import Dict, Any, Callable

def create_keyword_analysis_panel(find_similar_words_callback: Callable) -> Dict[str, Any]:
    """
    Create the keyword analysis panel UI components.
    
    Args:
        find_similar_words_callback: Function to call when searching for similar words
        
    Returns:
        Dictionary of UI components
    """
    gr.Markdown("""
    # Schlagwort-Analyse
    
    Hier können Sie ähnliche Wörter zu einem Suchbegriff finden, um Ihre Suchanfragen zu verbessern.
    
    Der FastText-Algorithmus findet Wörter mit ähnlicher Bedeutung basierend auf dem Kontext, 
    in dem sie im Spiegel-Archiv verwendet werden.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            single_keyword = gr.Textbox(
                label="Suchbegriff",
                placeholder="Geben Sie einen Begriff ein...",
                lines=1
            )
            
            single_expansion_factor = gr.Slider(
                minimum=1,
                maximum=20,
                value=10,
                step=1,
                label="Anzahl ähnlicher Wörter"
            )
            
            single_word_btn = gr.Button("Ähnliche Wörter finden")
        
        with gr.Column(scale=1):
            single_word_output = gr.Markdown(
                label="Ähnliche Wörter",
                value="Die Ergebnisse erscheinen hier..."
            )
    
    # Connect events
    single_word_btn.click(
        find_similar_words_callback,
        inputs=[single_keyword, single_expansion_factor],
        outputs=[single_word_output]
    )
    
    # Define all components to be returned
    components = {
        "single_keyword": single_keyword,
        "single_expansion_factor": single_expansion_factor,
        "single_word_btn": single_word_btn,
        "single_word_output": single_word_output
    }
    
    return components