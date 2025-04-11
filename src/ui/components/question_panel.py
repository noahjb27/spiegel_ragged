# src/ui/components/question_panel.py
"""
Question panel component for the Spiegel RAG application.
This component defines the UI elements for asking questions about retrieved content.
"""
import gradio as gr
from typing import Dict, Any

from src.config import settings

def create_question_panel() -> Dict[str, Any]:
    """
    Create the question panel UI components.
    
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        gr.Markdown("## Frage stellen")
        
        question = gr.Textbox(
            label="Frage (was Sie über die gefundenen Inhalte wissen möchten)",
            placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
            lines=2,
            info="Formulieren Sie Ihre Frage, die anhand der gefundenen Texte beantwortet werden soll."
        )
        
        # Model selection settings
        with gr.Accordion("LLM-Einstellungen", open=False):
            gr.Markdown("""
            ### Modellauswahl
            
            Sie können zwischen verschiedenen LLM-Modellen wählen:
            - **HU-LLM**: Lokales Modell (kein API-Schlüssel erforderlich, HU-Netzwerk erforderlich)
            - **OpenAI GPT-4o**: Leistungsstärkstes OpenAI-Modell (erfordert API-Schlüssel)
            - **OpenAI GPT-3.5 Turbo**: Schnelles OpenAI-Modell (erfordert API-Schlüssel)
            """)
            
            with gr.Row():
                model_selection = gr.Radio(
                    choices=["hu-llm", "openai-gpt4o", "openai-gpt35"],
                    value="hu-llm",
                    label="LLM-Modell",
                    info="Wählen Sie das zu verwendende Sprachmodell"
                )
            
            with gr.Row(visible=False) as openai_key_row:
                openai_api_key = gr.Textbox(
                    label="OpenAI API-Schlüssel",
                    placeholder="sk-...",
                    type="password",
                    info="Ihr OpenAI API-Schlüssel wird nur für diese Sitzung gespeichert"
                )
            
            # System prompt selection
            with gr.Row():
                system_prompt_template = gr.Dropdown(
                    choices=list(settings.SYSTEM_PROMPTS.keys()),
                    value="default",
                    label="System Prompt Vorlage",
                    info="Wählen Sie eine vordefinierte Vorlage oder passen Sie den Prompt manuell an."
                )
            
            # Add custom system prompt input
            with gr.Row():
                custom_system_prompt = gr.Textbox(
                    label="Eigener System Prompt",
                    placeholder="Anpassen des System Prompts für spezifische Anweisungen an das LLM...",
                    value="",
                    lines=5,
                    info="Leer lassen für die gewählte Vorlage oder eigenen Prompt eingeben."
                )
            
            # Add temperature slider
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Temperatur",
                    info="Kontrolliert die Kreativität der Antworten. Höhere Werte = kreativere Antworten."
                )
            
            # Add max tokens slider
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=100,
                    maximum=4000,
                    value=1000,
                    step=100,
                    label="Maximale Antwortlänge",
                    info="Maximale Anzahl der Token in der Antwort."
                )
        
        analyze_btn = gr.Button("Frage beantworten", variant="primary")
        
    # Connect model selection to API key visibility
    model_selection.change(
        fn=lambda model_choice: gr.update(visible=model_choice.startswith("openai")),
        inputs=[model_selection],
        outputs=[openai_key_row]
    )
    
    # Define all components to be returned
    components = {
        "question": question,
        "model_selection": model_selection,
        "openai_api_key": openai_api_key,
        "system_prompt_template": system_prompt_template,
        "custom_system_prompt": custom_system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "analyze_btn": analyze_btn,
        "openai_key_row": openai_key_row
    }
    
    return components