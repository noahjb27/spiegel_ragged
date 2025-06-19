# src/ui/components/search_panel.py - Updated with agent minimum retrieval score
"""
Updated search panel component with minimum retrieval relevance score for agent search.
"""
import gradio as gr
from typing import Dict, Any, Callable

from src.config import settings

def create_search_panel(
    retrieve_callback: Callable,
    agent_search_callback: Callable,
    preview_callback: Callable,
    toggle_api_key_callback: Callable
) -> Dict[str, Any]:
    """
    Create the redesigned search panel with stacked standard/agent options.
    
    Returns:
        Dictionary of UI components
    """
    # Hidden state for expanded words and retrieved chunks
    expanded_words_state = gr.State("")
    retrieved_chunks_state = gr.State(None)
    
    # Search mode selection
    with gr.Group():
        gr.Markdown("## Suchmodus auswählen")
        
        search_mode = gr.Radio(
            choices=[
                ("Standard-Suche", "standard"),
                ("Agenten-Suche", "agent")
            ],
            value="standard",
            label="Suchmethode",
            info="Wählen Sie zwischen Standard-Suche (schnell) oder Agenten-Suche (KI-gestützte Bewertung)"
        )
    
    # Common fields (shown for both modes)
    with gr.Group():
        gr.Markdown("## Allgemeine Einstellungen")
        
        content_description = gr.Textbox(
            label="Inhaltsbeschreibung (welche Quellen gesucht werden sollen)",
            placeholder="Beispiel: Berichterstattung über die Berliner Mauer",
            lines=2,
            info="Beschreiben Sie, welche Art von Inhalten Sie im Archiv finden möchten."
        )
        
        with gr.Row():
            year_start = gr.Slider(
                minimum=settings.MIN_YEAR,
                maximum=settings.MAX_YEAR,
                value=1960,
                step=1,
                label="Startjahr",
                info="Beginn des zu durchsuchenden Zeitraums"
            )
            
            year_end = gr.Slider(
                minimum=settings.MIN_YEAR,
                maximum=settings.MAX_YEAR,
                value=1970,
                step=1,
                label="Endjahr",
                info="Ende des zu durchsuchenden Zeitraums"
            )
        
        with gr.Row():
            chunk_size = gr.Dropdown(
                choices=[500, 2000, 3000],
                value=3000,
                label="Textgröße",
                info="Größe der Textabschnitte in Zeichen."
            )
    
    # Standard search specific settings
    with gr.Group(visible=True) as standard_settings:
        gr.Markdown("## Standard-Suche Einstellungen")
        
        top_k = gr.Slider(
            minimum=1,
            maximum=50,
            value=10,
            step=1,
            label="Anzahl Ergebnisse",
            info="Maximale Anzahl der Texte, die abgerufen werden sollen."
        )
        
        with gr.Accordion("Erweiterte Einstellungen", open=False):
            # Keyword filtering
            with gr.Accordion("Schlagwort-Filterung", open=False):
                gr.Markdown("""
                ### Schlagwort-Filterung
                
                Filtern Sie die Suchergebnisse nach bestimmten Schlagwörtern. Sie können auch boolesche Ausdrücke verwenden (AND, OR, NOT).
                
                **Beispiele:**
                - `mauer` - Findet Texte, die "mauer" enthalten
                - `berlin AND mauer` - Findet Texte, die sowohl "berlin" als auch "mauer" enthalten
                - `berlin AND (mauer OR grenze) NOT sowjet` - Komplexere Ausdrücke sind möglich
                """)
                
                keywords = gr.Textbox(
                    label="Schlagwörter (boolescher Ausdruck)",
                    placeholder="berlin AND mauer",
                    lines=2
                )
                
                with gr.Row():
                    search_in = gr.CheckboxGroup(
                        choices=["Text", "Artikeltitel", "Schlagworte"],
                        value=["Text"],
                        label="Suche in"
                    )
                    
                    enforce_keywords = gr.Checkbox(
                        label="Strikte Filterung",
                        value=True,
                        info="Wenn aktiviert, werden nur Texte angezeigt, die die angegebenen Schlagwörter enthalten."
                    )
            
                with gr.Row():
                    use_semantic_expansion = gr.Checkbox(
                        label="Semantische Erweiterung",
                        value=True,
                        info="Findet und berücksichtigt auch semantisch ähnliche Wörter"
                    )
                    
                    semantic_expansion_factor = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Anzahl ähnlicher Wörter"
                    )
                
                with gr.Row():
                    preview_btn = gr.Button("Vorschau ähnlicher Wörter")
                
                expansion_output = gr.Markdown(label="Ähnliche Wörter")
            
            # Time window search
            with gr.Accordion("Zeitfenster-Suche", open=False):
                gr.Markdown("""
                ### Zeitfenster-Suche
                
                Die Zeitfenster-Suche unterteilt den Suchzeitraum in kleinere Abschnitte und sorgt dafür, 
                dass Ergebnisse aus verschiedenen Zeitperioden berücksichtigt werden.
                """)
                
                with gr.Row():
                    use_time_windows = gr.Checkbox(
                        label="Zeitfenster-Suche aktivieren",
                        value=False,
                        info="Sucht in definierten Zeitfenstern anstatt im gesamten Zeitraum"
                    )
                    
                    time_window_size = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Fenstergröße (Jahre)",
                        info="Größe der einzelnen Zeitfenster in Jahren"
                    )
        
        standard_search_btn = gr.Button("Standard-Suche starten", variant="primary")
    
    # Agent search specific settings
    with gr.Group(visible=False) as agent_settings:
        gr.Markdown("## Agenten-Suche Einstellungen")
        
        gr.Markdown("""
        Die Agenten-Suche verwendet KI-gestützte Bewertung zur Auswahl der relevantesten Quellen.
        Sie können Zeitfenster verwenden und die Anzahl der Texte pro Fenster konfigurieren.
        """)
        
        # Agent time windows (default enabled)
        with gr.Row():
            agent_use_time_windows = gr.Checkbox(
                label="Zeitfenster verwenden",
                value=settings.AGENT_DEFAULT_USE_TIME_WINDOWS,
                info="Teilt den Zeitraum in kleinere Fenster auf"
            )
            
            agent_time_window_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=settings.AGENT_DEFAULT_TIME_WINDOW_SIZE,
                step=1,
                label="Fenstergröße (Jahre)",
                info="Größe der einzelnen Zeitfenster"
            )
        
        # Chunks per window configuration
        with gr.Row():
            chunks_per_window_initial = gr.Slider(
                minimum=10,
                maximum=200,
                value=settings.AGENT_DEFAULT_CHUNKS_PER_WINDOW_INITIAL,
                step=5,
                label="Initial pro Fenster",
                info="Anzahl der Texte, die zunächst pro Zeitfenster abgerufen werden"
            )
            
            chunks_per_window_final = gr.Slider(
                minimum=5,
                maximum=100,
                value=settings.AGENT_DEFAULT_CHUNKS_PER_WINDOW_FINAL,
                step=5,
                label="Final pro Fenster",
                info="Anzahl der Texte, die nach KI-Bewertung pro Zeitfenster behalten werden"
            )
        
        # NEW: Agent minimum retrieval relevance score
        with gr.Row():
            agent_min_retrieval_score = gr.Slider(
                minimum=0.1,
                maximum=0.8,
                value=0.25,
                step=0.05,
                label="Mindest-Retrieval-Score",
                info="Minimale Ähnlichkeitsschwelle für die initiale Quellenauswahl (niedrigere Werte = mehr Kandidaten)"
            )
        
        # Agent-specific keyword filtering (simplified)
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
                    info="Nur Texte mit den angegebenen Schlagwörtern"
                )
        
        # Agent LLM settings
        with gr.Accordion("KI-Bewertungseinstellungen", open=True):
            agent_model = gr.Radio(
                choices=["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"],
                value="hu-llm3",
                label="LLM-Modell für Bewertung",
                info="Wählen Sie das Modell für die Quellenbewertung. DeepSeek R1 ist besonders leistungsstark für komplexe Analysen."
            )
            
            agent_system_prompt_template = gr.Dropdown(
                choices=list(settings.AGENT_SYSTEM_PROMPTS.keys()),
                value="agent_default",
                label="Bewertungs-Prompt Vorlage",
                info="Vordefinierte Vorlagen für verschiedene Analysezwecke"
            )
            
            agent_custom_system_prompt = gr.Textbox(
                label="Eigener Bewertungs-Prompt",
                placeholder="Anpassung des System-Prompts für die Quellenbewertung...",
                value="",
                lines=6,
                info="Leer lassen für die gewählte Vorlage oder eigenen Prompt eingeben"
            )
        
        agent_search_btn = gr.Button("Agenten-Suche starten", variant="primary")
        
        # Progress indicator for agent search
        agent_progress = gr.Markdown("", visible=False)
        agent_cancel_btn = gr.Button("Abbrechen", visible=False, variant="stop")
    
    # Results display (common for both modes)
    with gr.Group():
        search_status = gr.Markdown("Noch keine Suche durchgeführt.")
    
    # Connect preview functionality
    preview_btn.click(
        preview_callback,
        inputs=[keywords, semantic_expansion_factor],
        outputs=[expansion_output, expanded_words_state]
    )
    
    # Show/hide settings based on search mode
    def toggle_search_settings(mode):
        if mode == "standard":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    search_mode.change(
        toggle_search_settings,
        inputs=[search_mode],
        outputs=[standard_settings, agent_settings]
    )
    
    # Define all components to be returned
    components = {
        # Common components
        "search_mode": search_mode,
        "content_description": content_description,
        "chunk_size": chunk_size,
        "year_start": year_start,
        "year_end": year_end,
        "search_status": search_status,
        "retrieved_chunks_state": retrieved_chunks_state,
        "expanded_words_state": expanded_words_state,
        
        # Standard search components
        "top_k": top_k,
        "keywords": keywords,
        "search_in": search_in,
        "use_semantic_expansion": use_semantic_expansion,
        "semantic_expansion_factor": semantic_expansion_factor,
        "enforce_keywords": enforce_keywords,
        "use_time_windows": use_time_windows,
        "time_window_size": time_window_size,
        "standard_search_btn": standard_search_btn,
        "expansion_output": expansion_output,
        
        # Agent search components
        "agent_use_time_windows": agent_use_time_windows,
        "agent_time_window_size": agent_time_window_size,
        "chunks_per_window_initial": chunks_per_window_initial,
        "chunks_per_window_final": chunks_per_window_final,
        "agent_min_retrieval_score": agent_min_retrieval_score,  # NEW
        "agent_keywords": agent_keywords,
        "agent_search_in": agent_search_in,
        "agent_enforce_keywords": agent_enforce_keywords,
        "agent_model": agent_model,
        "agent_system_prompt_template": agent_system_prompt_template,
        "agent_custom_system_prompt": agent_custom_system_prompt,
        "agent_search_btn": agent_search_btn,
        "agent_progress": agent_progress,
        "agent_cancel_btn": agent_cancel_btn,
        
        # UI groups for visibility control
        "standard_settings": standard_settings,
        "agent_settings": agent_settings
    }
    
    return components