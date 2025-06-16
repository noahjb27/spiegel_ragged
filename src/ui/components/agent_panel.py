# src/ui/components/agent_panel.py - Fixed version
"""
Agent panel component for the Spiegel RAG application.
This component defines the UI elements for the agent-based search approach.
"""
import gradio as gr
from typing import Dict, Any, Callable

from src.config import settings

def create_agent_panel(
    agent_search_callback: Callable,
    toggle_api_key_callback: Callable
) -> Dict[str, Any]:
    """
    Create the agent panel UI components.
    
    Args:
        agent_search_callback: Function to call for agent-based search
        toggle_api_key_callback: Function to toggle API key visibility (not used anymore)
        
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        gr.Markdown("""
        # Agenten-basierte RAG-Suche
        
        Diese Funktion kombiniert Retrieval und Analyse in einem mehrstufigen Prozess:
        
        1. Zunächst werden mehr Quellen abgerufen als bei der Standard-Suche (100 statt 10)
        2. Das Sprachmodell bewertet dann die Relevanz jedes Textabschnitts für Ihre Frage
        3. In mehreren Filterstufen werden die relevantesten Texte identifiziert
        4. Schließlich wird eine fundierte Antwort auf Basis der besten Texte generiert
        
        **Vorteile:**
        - Tiefere Analyse und bessere Auswahl relevanter Quellen
        - Transparente Bewertung der Relevanz jedes Textabschnitts
        - Kombination von Vektorsimilarität und semantischer Bewertung
        
        **Hinweis:** Diese Methode ist aufgrund der mehrstufigen Analyse langsamer als die Standard-Suche.
        """)
        
        # Basic input fields
        agent_question = gr.Textbox(
            label="Frage",
            placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
            lines=2,
            info="Die Frage, die Sie mit den gefundenen Texten beantworten möchten."
        )
        
        agent_content_description = gr.Textbox(
            label="Inhaltsbeschreibung (optional)",
            placeholder="Leer lassen, um die Frage für die Suche zu verwenden, oder spezifizieren Sie den zu suchenden Inhalt",
            lines=2,
            info="Beschreibt, welche Art von Inhalten gesucht werden sollen. Wenn leer, wird die Frage verwendet."
        )
        
        # Advanced settings in accordions
        with gr.Accordion("Zeitraum und Textgröße", open=True):
            with gr.Row():
                agent_year_start = gr.Slider(
                    minimum=settings.MIN_YEAR,
                    maximum=settings.MAX_YEAR,
                    value=1960,
                    step=1,
                    label="Startjahr"
                )
                
                agent_year_end = gr.Slider(
                    minimum=settings.MIN_YEAR,
                    maximum=settings.MAX_YEAR,
                    value=1970,
                    step=1,
                    label="Endjahr"
                )
            
            with gr.Row():
                # FIXED: Added 500 chunk size option
                agent_chunk_size = gr.Dropdown(
                    choices=[500, 2000, 3000],
                    value=3000,
                    label="Textgröße",
                    info="Größe der Textabschnitte in Zeichen."
                )
                
        # Keyword filtering
        with gr.Accordion("Schlagwort-Filterung", open=False):
            agent_keywords = gr.Textbox(
                label="Schlagwörter (boolescher Ausdruck)",
                placeholder="berlin AND mauer",
                lines=2
            )
            
            with gr.Row():
                agent_search_in = gr.CheckboxGroup(
                    choices=["Text", "Artikeltitel", "Schlagworte"],
                    value=["Text"],
                    label="Suche in"
                )
                
                agent_enforce_keywords = gr.Checkbox(
                    label="Strikte Filterung",
                    value=True,
                    info="Wenn aktiviert, werden nur Texte angezeigt, die die angegebenen Schlagwörter enthalten."
                )
        
        # Agent settings
        with gr.Accordion("Agenten-Einstellungen", open=True):
            gr.Markdown("""
            ### Filtereinstellungen
            
            Konfigurieren Sie, wie der Agent Texte filtert und bewertet.
            """)
            
            with gr.Row():
                agent_initial_count = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Initiale Textmenge",
                    info="Anzahl der Texte, die zunächst abgerufen werden."
                )
            
            with gr.Row():
                agent_filter_stage1 = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Filterstufe 1",
                    info="Anzahl der Texte nach der ersten Bewertung."
                )
                
                agent_filter_stage2 = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Filterstufe 2",
                    info="Anzahl der Texte nach der zweiten Bewertung."
                )
                
                agent_filter_stage3 = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Filterstufe 3 (Endauswahl)",
                    info="Finale Anzahl der Texte für die Antwortgenerierung."
                )
        
        # Model settings
        with gr.Accordion("LLM-Einstellungen", open=False):
            # FIXED: Updated model choices
            with gr.Row():
                agent_model = gr.Radio(
                    choices=["hu-llm1", "hu-llm3", "openai-gpt4o", "gemini-pro"],
                    value="hu-llm3",
                    label="LLM-Modell",
                    info="Wählen Sie das zu verwendende Sprachmodell."
                )
            
            # REMOVED: API key input field completely
            
            with gr.Row():
                agent_system_prompt_template = gr.Dropdown(
                    choices=list(settings.SYSTEM_PROMPTS.keys()),
                    value="default",
                    label="System Prompt Vorlage"
                )
            
            with gr.Row():
                agent_custom_system_prompt = gr.Textbox(
                    label="Eigener System Prompt",
                    placeholder="Anpassen des System Prompts für spezifische Anweisungen an das LLM...",
                    value="",
                    lines=5,
                    info="Leer lassen für die gewählte Vorlage oder eigenen Prompt eingeben."
                )
        
        # Search button
        agent_search_btn = gr.Button("Agenten-Suche starten", variant="primary")
        
        # Results containers
        agent_status = gr.Markdown("Klicken Sie auf 'Agenten-Suche starten', um den Prozess zu beginnen.")
        
        # State for storing results
        agent_results_state = gr.State(None)
    
    # REMOVED: Model selection API key visibility toggle
    
    # Define all components to be returned - FIXED: Removed API key related components
    components = {
        "agent_question": agent_question,
        "agent_content_description": agent_content_description,
        "agent_year_start": agent_year_start,
        "agent_year_end": agent_year_end,
        "agent_chunk_size": agent_chunk_size,
        "agent_keywords": agent_keywords,
        "agent_search_in": agent_search_in,
        "agent_enforce_keywords": agent_enforce_keywords,
        "agent_initial_count": agent_initial_count,
        "agent_filter_stage1": agent_filter_stage1,
        "agent_filter_stage2": agent_filter_stage2,
        "agent_filter_stage3": agent_filter_stage3,
        "agent_model": agent_model,
        "agent_system_prompt_template": agent_system_prompt_template,
        "agent_custom_system_prompt": agent_custom_system_prompt,
        "agent_search_btn": agent_search_btn,
        "agent_status": agent_status,
        "agent_results_state": agent_results_state
    }
    
    return components