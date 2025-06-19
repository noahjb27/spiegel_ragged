# src/ui/components/question_panel.py - Updated with editable system prompts
"""
Question panel component for the Spiegel RAG application.
Updated to use editable system prompt templates.
"""
import gradio as gr
from typing import Dict, Any

from src.config import settings

def create_question_panel() -> Dict[str, Any]:
    """
    Create the question panel UI components with editable system prompt templates.
    
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
            - **HU-LLM 1**: Lokales Modell (HU-Netzwerk erforderlich)
            - **HU-LLM 3**: Lokales Modell (HU-Netzwerk erforderlich)  
            - **DeepSeek R1 32B**: Fortschrittliches Reasoning-Modell via Ollama (HU-Netzwerk erforderlich)
            - **OpenAI GPT-4o**: Leistungsstärkstes OpenAI-Modell
            - **Google Gemini Pro**: Google's neuestes Sprachmodell mit großem Kontextfenster
            
            **Empfehlung**: DeepSeek R1 für komplexe analytische Aufgaben, die mehrstufiges Denken erfordern.
            """)
            
            with gr.Row():
                model_selection = gr.Radio(
                    choices=["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"],
                    value="hu-llm3",
                    label="LLM-Modell",
                    info="Wählen Sie das zu verwendende Sprachmodell. DeepSeek R1 ist besonders gut für analytische Aufgaben."
                )
            
            # System prompt template selection and editing
            gr.Markdown("""
            ### System Prompt
            
            Wählen Sie eine Vorlage und bearbeiten Sie sie nach Ihren Bedürfnissen. Der System Prompt steuert, 
            wie das LLM die Analyse durchführt.
            """)
            
            with gr.Row():
                system_prompt_template = gr.Dropdown(
                    choices=list(settings.SYSTEM_PROMPTS.keys()),
                    value="default",
                    label="System Prompt Vorlage",
                    info="Wählen Sie eine Vorlage als Ausgangspunkt"
                )
                
                reset_system_prompt_btn = gr.Button("Auf Vorlage zurücksetzen", size="sm")
            
            # Editable system prompt text area - initialized with default template
            system_prompt_text = gr.Textbox(
                label="System Prompt (bearbeitbar)",
                value=settings.SYSTEM_PROMPTS["default"],
                lines=8,
                info="Bearbeiten Sie den System Prompt nach Ihren Bedürfnissen. Dieser Text wird an das LLM gesendet."
            )
            
            # Add temperature and max tokens
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Temperatur",
                    info="Bestimmt die Wahrscheinlichkeitsverteilung, aus der Tokens ausgewählt werden – höher bedeutet kreativere, aber potenziell weniger kohärente Texte."
                )
            
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
    
    # Event handlers for system prompt template management
    def load_system_prompt_template(template_name: str) -> str:
        """Load the selected template into the text area."""
        return settings.SYSTEM_PROMPTS.get(template_name, settings.SYSTEM_PROMPTS["default"])
    
    def reset_to_template(template_name: str) -> str:
        """Reset the text area to the selected template."""
        return settings.SYSTEM_PROMPTS.get(template_name, settings.SYSTEM_PROMPTS["default"])
    
    # Connect template dropdown to text area
    system_prompt_template.change(
        load_system_prompt_template,
        inputs=[system_prompt_template],
        outputs=[system_prompt_text]
    )
    
    # Connect reset button
    reset_system_prompt_btn.click(
        reset_to_template,
        inputs=[system_prompt_template],
        outputs=[system_prompt_text]
    )
    
    # Define all components to be returned
    components = {
        "question": question,
        "model_selection": model_selection,
        "system_prompt_template": system_prompt_template,
        "system_prompt_text": system_prompt_text,  # New: always use this
        "reset_system_prompt_btn": reset_system_prompt_btn,  # New: reset button
        "temperature": temperature,
        "max_tokens": max_tokens,
        "analyze_btn": analyze_btn,
        
        # Keep these for backward compatibility but mark as deprecated
        "custom_system_prompt": system_prompt_text  # Alias for backward compatibility
    }
    
    return components