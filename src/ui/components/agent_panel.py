# src/ui/components/llm_assisted_panel.py - Updated with new terminology
"""
LLM-Assisted panel component for the Spiegel RAG application.
"""
import gradio as gr
from typing import Dict, Any, Callable

from src.config import settings

def create_llm_assisted_panel(
    llm_assisted_search_callback: Callable,
    toggle_api_key_callback: Callable
) -> Dict[str, Any]:
    """
    Create the LLM-assisted panel UI components.
    
    Args:
        llm_assisted_search_callback: Function to call for LLM-assisted search
        toggle_api_key_callback: Function to toggle API key visibility (deprecated)
        
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        gr.Markdown("""
        # LLM-Unterstützte Auswahl
        
        Diese Funktion kombiniert Retrieval und Analyse in einem mehrstufigen Prozess:
        
        1. Zunächst werden mehr Quellen abgerufen als bei der Standard-Suche (z.B. 50 pro Zeitintervall)
        2. Das Sprachmodell bewertet dann die Relevanz jedes Textabschnitts für Ihre Query
        3. In mehreren Filterstufen werden die relevantesten Texte identifiziert
        4. Schließlich wird eine fundierte Antwort auf Basis der besten Texte generiert
        
        **Vorteile:**
        - Tiefere Analyse und bessere Auswahl relevanter Quellen
        - Transparente Bewertung der Relevanz jedes Textabschnitts
        - Kombination von Vektorsimilarität und semantischer Bewertung
        - Gleichmäßige zeitliche Verteilung durch Zeitintervall-Suche
        
        **Hinweis:** Diese Methode ist aufgrund der mehrstufigen Analyse langsamer als die Standard-Suche.
        """)
        
        # Basic input fields - UPDATED terminology
        llm_assisted_question = gr.Textbox(
            label="Forschungsfrage",
            placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
            lines=2,
            info="Die Frage, die Sie mit den gefundenen Texten beantworten möchten."
        )
        
        llm_assisted_retrieval_query = gr.Textbox(
            label="Retrieval-Query (optional)",
            placeholder="Leer lassen, um die Frage für die Suche zu verwenden, oder spezifizieren Sie den zu suchenden Inhalt",
            lines=2,
            info="Beschreibt, welche Art von Inhalten gesucht werden sollen. Wenn leer, wird die Frage verwendet."
        )
        
        # Advanced settings in accordions
        with gr.Accordion("Zeitraum und Chunking-Größe", open=True):
            with gr.Row():
                llm_assisted_year_start = gr.Slider(
                    minimum=settings.MIN_YEAR,
                    maximum=settings.MAX_YEAR,
                    value=1960,
                    step=1,
                    label="Startjahr"
                )
                
                llm_assisted_year_end = gr.Slider(
                    minimum=settings.MIN_YEAR,
                    maximum=settings.MAX_YEAR,
                    value=1970,
                    step=1,
                    label="Endjahr"
                )
            
            with gr.Row():
                llm_assisted_chunk_size = gr.Dropdown(
                    choices=[500, 2000, 3000],
                    value=3000,
                    label="Chunking-Größe",
                    info="Größe der Textabschnitte in Zeichen."
                )
        
        # Keyword filtering - UPDATED: strikte Filterung always enabled
        with gr.Accordion("Schlagwort-Filterung", open=False):
            gr.Markdown("""
            Filtern Sie die Suchergebnisse nach bestimmten Schlagwörtern. 
            Die Filterung ist standardmäßig strikt aktiviert.
            """)
            
            llm_assisted_keywords = gr.Textbox(
                label="Schlagwörter (boolescher Ausdruck)",
                placeholder="berlin AND mauer",
                lines=2
            )
            
            llm_assisted_search_in = gr.CheckboxGroup(
                choices=["Text", "Artikeltitel", "Schlagworte"],
                value=["Text"],
                label="Suche in"
            )
        
        # LLM-assisted settings - UPDATED terminology
        with gr.Accordion("LLM-Unterstützte Auswahl Einstellungen", open=True):
            gr.Markdown("""
            ### Zeitintervall-Einstellungen
            
            Konfigurieren Sie, wie das LLM Texte über verschiedene Zeit-Intervalle hinweg filtert und bewertet.
            """)
            
            with gr.Row():
                llm_assisted_use_time_intervals = gr.Checkbox(
                    label="Zeit-Intervalle verwenden",
                    value=settings.LLM_ASSISTED_DEFAULT_USE_TIME_WINDOWS,
                    info="Teilt den Zeitraum in kleinere Intervalle auf für gleichmäßige zeitliche Verteilung"
                )
                
                llm_assisted_time_interval_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=settings.LLM_ASSISTED_DEFAULT_TIME_WINDOW_SIZE,
                    step=1,
                    label="Intervall-Größe (Jahre)",
                    info="Größe der einzelnen Zeit-Intervalle in Jahren"
                )
            
            with gr.Row():
                llm_assisted_initial_count = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=settings.LLM_ASSISTED_DEFAULT_CHUNKS_PER_WINDOW_INITIAL,
                    step=10,
                    label="Initial pro Intervall",
                    info="Anzahl der Texte, die zunächst pro Zeitintervall abgerufen werden."
                )
            
            with gr.Row():
                llm_assisted_filter_stage1 = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Filterstufe 1",
                    info="Anzahl der Texte nach der ersten LLM-Bewertung."
                )
                
                llm_assisted_filter_stage2 = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Filterstufe 2",
                    info="Anzahl der Texte nach der zweiten LLM-Bewertung."
                )
                
                llm_assisted_filter_stage3 = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Filterstufe 3 (Endauswahl)",
                    info="Finale Anzahl der Texte für die Antwortgenerierung."
                )
            
            # NEW: Minimum retrieval score
            llm_assisted_min_retrieval_score = gr.Slider(
                minimum=0.1,
                maximum=0.8,
                value=0.25,
                step=0.05,
                label="Mindest-Retrieval-Score",
                info="Minimale Ähnlichkeitsschwelle für die initiale Quellenauswahl"
            )
        
        # Model settings - UPDATED terminology and added temperature
        with gr.Accordion("LLM-Einstellungen", open=False):
            with gr.Row():
                llm_assisted_model = gr.Radio(
                    choices=["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"],
                    value="hu-llm3",
                    label="LLM-Modell",
                    info="Wählen Sie das zu verwendende Sprachmodell."
                )
            
            # NEW: Temperature control for LLM evaluation
            llm_assisted_temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=settings.LLM_ASSISTED_DEFAULT_TEMPERATURE,
                step=0.1,
                label="Temperatur",
                info="Bestimmt den Determinismus der LLM-Bewertung. Niedrigere Werte = konsistentere Bewertungen."
            )
            
            with gr.Row():
                llm_assisted_system_prompt_template = gr.Dropdown(
                    choices=list(settings.LLM_ASSISTED_SYSTEM_PROMPTS.keys()),
                    value="standard_evaluation",
                    label="System-Prompt Vorlage"
                )
            
            with gr.Row():
                llm_assisted_custom_system_prompt = gr.Textbox(
                    label="Eigener System-Prompt",
                    placeholder="Anpassen des System Prompts für spezifische Anweisungen an das LLM...",
                    value="",
                    lines=5,
                    info="Leer lassen für die gewählte Vorlage oder eigenen Prompt eingeben."
                )
        
        # Search button - UPDATED terminology
        llm_assisted_search_btn = gr.Button("LLM-Unterstützte Auswahl starten", variant="primary")
        
        # Results containers
        llm_assisted_status = gr.Markdown("Klicken Sie auf 'LLM-Unterstützte Auswahl starten', um den Prozess zu beginnen.")
        
        # State for storing results
        llm_assisted_results_state = gr.State(None)
    
    # Define all components to be returned - UPDATED component names
    components = {
        "llm_assisted_question": llm_assisted_question,
        "llm_assisted_retrieval_query": llm_assisted_retrieval_query,
        "llm_assisted_year_start": llm_assisted_year_start,
        "llm_assisted_year_end": llm_assisted_year_end,
        "llm_assisted_chunk_size": llm_assisted_chunk_size,
        "llm_assisted_keywords": llm_assisted_keywords,
        "llm_assisted_search_in": llm_assisted_search_in,
        "llm_assisted_use_time_intervals": llm_assisted_use_time_intervals,
        "llm_assisted_time_interval_size": llm_assisted_time_interval_size,
        "llm_assisted_initial_count": llm_assisted_initial_count,
        "llm_assisted_filter_stage1": llm_assisted_filter_stage1,
        "llm_assisted_filter_stage2": llm_assisted_filter_stage2,
        "llm_assisted_filter_stage3": llm_assisted_filter_stage3,
        "llm_assisted_min_retrieval_score": llm_assisted_min_retrieval_score,
        "llm_assisted_model": llm_assisted_model,
        "llm_assisted_temperature": llm_assisted_temperature,  # NEW
        "llm_assisted_system_prompt_template": llm_assisted_system_prompt_template,
        "llm_assisted_custom_system_prompt": llm_assisted_custom_system_prompt,
        "llm_assisted_search_btn": llm_assisted_search_btn,
        "llm_assisted_status": llm_assisted_status,
        "llm_assisted_results_state": llm_assisted_results_state,
    }
    
    return components