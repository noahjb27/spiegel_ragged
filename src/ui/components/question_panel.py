# src/ui/components/question_panel.py - Updated with DeepSeek R1 support
"""
Question panel component for the Spiegel RAG application.
This component defines the UI elements for asking questions about retrieved content.
Updated to include DeepSeek R1 model option.
"""
import gradio as gr
from typing import Dict, Any

from src.config import settings

def create_question_panel() -> Dict[str, Any]:
    """
    Create the question panel UI components with DeepSeek R1 support.
    
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
            # UPDATED: Model description with DeepSeek R1
            gr.Markdown("""
            ### Modellauswahl
            
            Sie können zwischen verschiedenen LLM-Modellen wählen:
            - **HU-LLM 1**: Lokales Modell (HU-Netzwerk erforderlich)
            - **HU-LLM 3**: Lokales Modell (HU-Netzwerk erforderlich)  
            - **DeepSeek R1 32B**: Fortschrittliches Reasoning-Modell via Ollama (HU-Netzwerk erforderlich)
            - **OpenAI GPT-4o**: Leistungsstärkstes OpenAI-Modell
            - **Google Gemini Pro**: Google's neuestes Sprachmodell mit großem Kontextfenster
            
            **Empfehlung**: DeepSeek R1 für komplexe analytische Aufgaben, die mehrstufiges Denken erfordern.
            """)
            
            # UPDATED: Model choices to include DeepSeek R1
            with gr.Row():
                model_selection = gr.Radio(
                    choices=["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"],
                    value="hu-llm3",
                    label="LLM-Modell",
                    info="Wählen Sie das zu verwendende Sprachmodell. DeepSeek R1 ist besonders gut für analytische Aufgaben."
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
                    info="Kontrolliert die Kreativität der Antworten. Höhere Werte = kreativere Antworten. DeepSeek R1 funktioniert oft gut mit niedrigeren Werten (0.1-0.4)."
                )
            
            # Add max tokens slider
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=100,
                    maximum=4000,
                    value=1000,
                    step=100,
                    label="Maximale Antwortlänge",
                    info="Maximale Anzahl der Token in der Antwort. DeepSeek R1 kann längere, detailliertere Antworten generieren."
                )
        
        analyze_btn = gr.Button("Frage beantworten", variant="primary")
        
    # Define all components to be returned
    components = {
        "question": question,
        "model_selection": model_selection,
        "system_prompt_template": system_prompt_template,
        "custom_system_prompt": custom_system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "analyze_btn": analyze_btn
    }
    
    return components