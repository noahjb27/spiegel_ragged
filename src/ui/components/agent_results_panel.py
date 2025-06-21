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
    Create the LLM-assisted results panel UI components using main CSS classes.
    Removed all duplicate CSS - now relies entirely on main app CSS.
    """
    with gr.Group():
        # Answer section using main CSS classes
        llm_assisted_answer_output = gr.Markdown(
            value="Die Antwort wird hier angezeigt.",
            label="Antwort der LLM-Unterstützten Auswahl"
        )
        
        # Process visualization using main CSS classes
        with gr.Accordion("LLM-Bewertungsprozess", open=True):
            llm_assisted_process_output = gr.HTML(
                value="<div class='llm-assisted-progress'>Der LLM-Bewertungsprozess wird hier visualisiert, wenn die Suche abgeschlossen ist.</div>"
            )
        
        # Chunk evaluations using main CSS classes
        with gr.Accordion("LLM-Textbewertungen", open=False):
            llm_assisted_evaluations_output = gr.HTML(
                value="<div class='evaluation-card'>Die LLM-Bewertungen der Textabschnitte werden hier angezeigt.</div>"
            )
        
        # Retrieved chunks using main CSS classes
        with gr.Accordion("Ausgewählte Texte", open=False):
            llm_assisted_chunks_output = gr.Markdown(
                value="Die durch LLM ausgewählten Texte werden hier angezeigt."
            )
            
            # Download functionality using main CSS button classes
            with gr.Row():
                llm_assisted_download_json_btn = gr.Button(
                    "📥 LLM-Ergebnisse als JSON", 
                    elem_classes=["download-button"]
                )
            
            llm_assisted_download_json_file = gr.File(visible=False)
        
        # Metadata using main CSS classes
        with gr.Accordion("Metadaten", open=False):
            llm_assisted_metadata_output = gr.Markdown(
                value="Metadaten zur LLM-Unterstützten Auswahl werden hier angezeigt."
            )
    
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
