# src/ui/components/search_panel.py - Updated with new terminology and structure
import gradio as gr
from typing import Dict, Any, Callable

from src.config import settings

def create_search_panel(
    retrieve_callback: Callable,
    llm_assisted_search_callback: Callable,
    preview_callback: Callable,
    toggle_api_key_callback: Callable
) -> Dict[str, Any]:
    """
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
                ("LLM-Unterstützte Auswahl", "llm_assisted") 
            ],
            value="standard",
            label="Suchmethode",
            info="Wählen Sie zwischen Standard-Suche (schnell) oder LLM-Unterstützter Auswahl (KI-gestützte Bewertung)"
        )
    
    # Common fields (shown for both modes)
    with gr.Group():
        gr.Markdown("## Allgemeine Einstellungen")
        
        # UPDATED: Changed to Retrieval-Query with helper text
        retrieval_query = gr.Textbox(
            label="Retrieval-Query (welche Quellen gesucht werden sollen)",
            placeholder="Beispiel: Berichterstattung über die Berliner Mauer",
            lines=2,
            info="Verwenden Sie wenige Stoppwörter und viele Begriffe. Beschreiben Sie, welche Art von Inhalten Sie im Archiv finden möchten."
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
            # UPDATED: Changed to Chunking-Größe
            chunk_size = gr.Dropdown(
                choices=[500, 2000, 3000],
                value=3000,
                label="Chunking-Größe",
                info="Größe der Textabschnitte in Zeichen."
            )
    
    # Standard search specific settings
    with gr.Group(visible=True) as standard_settings:
        gr.Markdown("## Standard-Suche Einstellungen")
        
        # Different options for chunks based on time interval usage
        with gr.Row():
            # Total chunks (shown when time intervals are disabled)
            top_k = gr.Slider(
                minimum=1,
                maximum=150,
                value=10,
                step=1,
                label="Anzahl Ergebnisse (gesamt)",
                info="Maximale Anzahl der Texte insgesamt.",
                visible=True
            )
            
            # Chunks per interval (shown when time intervals are enabled)
            chunks_per_interval = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Ergebnisse pro Zeitintervall",
                info="Anzahl der Texte pro Zeitintervall.",
                visible=False
            )
        
        with gr.Accordion("Erweiterte Einstellungen", open=False):
            # Keyword filtering - UPDATED structure
            with gr.Accordion("Schlagwort-Filterung", open=False):
                gr.Markdown("""
                ### Schlagwort-Filterung
                
                Filtern Sie die Suchergebnisse nach bestimmten Schlagwörtern. 
                Dies kann helfen, die zeitgenössische Sprache der Artikel zu nutzen und die Retrieval-Qualität zu verbessern.
                
                **Beispiele:**
                - `mauer` - Findet Texte, die "mauer" enthalten
                - `berlin AND mauer` - Findet Texte, die sowohl "berlin" als auch "mauer" enthalten
                - `berlin AND (mauer OR grenze) NOT sowjet` - Komplexere Ausdrücke sind möglich
                - Wenn Sie nur Briefe möchten, dann suchen Sie in "Artikeltitel"
                """)
                
                keywords = gr.Textbox(
                    label="Schlagwörter (boolescher Ausdruck)",
                    placeholder="berlin AND mauer",
                    lines=2
                )
                
                search_in = gr.CheckboxGroup(
                    choices=["Text", "Artikeltitel", "Schlagworte"],
                    value=["Text"],
                    label="Suche in",
                    info="Wenn Sie zum Beispiel nur Leserbriefe möchten, dann wählen Sie 'Artikeltitel' mit Brief als Schlagwort."
                )
                
                # UPDATED: Reorganized semantic expansion controls
                with gr.Row():
                    use_semantic_expansion = gr.Checkbox(
                        label="Semantische Erweiterung",
                        value=True,
                        info="Findet und berücksichtigt auch semantisch ähnliche Wörter"
                    )
                    
                    # MOVED: Now under semantic expansion
                    semantic_expansion_factor = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Anzahl ähnlicher Wörter",
                        info="Wie viele ähnliche Begriffe pro Schlagwort gesucht werden"
                    )
                
                with gr.Row():
                    preview_btn = gr.Button("Vorschau ähnlicher Wörter")
                
                # UPDATED: Show frequency and similarity
                expansion_output = gr.Markdown(label="Ähnliche Wörter mit Häufigkeiten")
            
            # UPDATED: Zeitintervall-Suche (formerly Zeitfenster-Suche)
            with gr.Accordion("Zeitintervall-Suche", open=False):
                gr.Markdown("""
                ### Zeitintervall-Suche
                
                Die Zeitintervall-Suche unterteilt den Suchzeitraum in kleinere Abschnitte und sorgt für 
                eine gleichmäßige Verteilung der Ergebnisse über verschiedene Zeitperioden. Dies ermöglicht
                ein diakrones Narrativ durch ausgewogene zeitliche Abdeckung.
                """)
                
                # UPDATED: Moved window size under the checkbox
                use_time_intervals = gr.Checkbox(
                    label="Zeitintervall-Suche aktivieren",
                    value=False,
                    info="Sucht in definierten Zeit-Intervallen für gleichmäßige zeitliche Verteilung"
                )
                
                time_interval_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Intervall-Größe (Jahre)",
                    info="Größe der einzelnen Zeit-Intervalle in Jahren"
                )
                
                # NEW: Automatic calculation display
                interval_calculation = gr.Markdown(
                    value="", 
                    label="Berechnete Aufteilung",
                    visible=False
                )
        
        standard_search_btn = gr.Button("Standard-Suche starten", variant="primary")
    
    # LLM-assisted search specific settings - UPDATED terminology
    with gr.Group(visible=False) as llm_assisted_settings:
        gr.Markdown("## LLM-Unterstützte Auswahl Einstellungen")
        
        gr.Markdown("""
        Die LLM-Unterstützte Auswahl verwendet KI-gestützte Bewertung zur Auswahl der relevantesten Quellen.
        Sie können Zeit-Intervalle verwenden und die Anzahl der Texte pro Intervall konfigurieren.
        """)
        
        # LLM-assisted time intervals (default enabled) - UPDATED terminology
        with gr.Row():
            llm_assisted_use_time_intervals = gr.Checkbox(
                label="Zeit-Intervalle verwenden",
                value=settings.LLM_ASSISTED_DEFAULT_USE_TIME_WINDOWS,
                info="Teilt den Zeitraum in kleinere Intervalle auf"
            )
            
            llm_assisted_time_interval_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=settings.LLM_ASSISTED_DEFAULT_TIME_WINDOW_SIZE,
                step=1,
                label="Intervall-Größe (Jahre)",
                info="Größe der einzelnen Zeit-Intervalle"
            )
        
        # Chunks per interval configuration
        with gr.Row():
            chunks_per_interval_initial = gr.Slider(
                minimum=10,
                maximum=200,
                value=settings.LLM_ASSISTED_DEFAULT_CHUNKS_PER_WINDOW_INITIAL,
                step=5,
                label="Initial pro Intervall",
                info="Anzahl der Texte, die zunächst pro Zeitintervall abgerufen werden"
            )
            
            chunks_per_interval_final = gr.Slider(
                minimum=5,
                maximum=100,
                value=settings.LLM_ASSISTED_DEFAULT_CHUNKS_PER_WINDOW_FINAL,
                step=5,
                label="Final pro Intervall",
                info="Anzahl der Texte, die nach KI-Bewertung pro Zeitintervall behalten werden"
            )
        
        # LLM-assisted minimum retrieval relevance score
        with gr.Row():
            llm_assisted_min_retrieval_score = gr.Slider(
                minimum=0.1,
                maximum=0.8,
                value=0.25,
                step=0.05,
                label="Mindest-Retrieval-Score",
                info="Minimale Ähnlichkeitsschwelle für die initiale Quellenauswahl (niedrigere Werte = mehr Kandidaten)"
            )
        
        # LLM-assisted-specific keyword filtering (simplified)
        with gr.Accordion("Schlagwort-Filterung", open=False):
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
        
        # UPDATED: LLM settings with temperature and new prompts
        with gr.Accordion("KI-Bewertungseinstellungen", open=True):
            llm_assisted_model = gr.Radio(
                choices=["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"],
                value="hu-llm3",
                label="LLM-Modell für Bewertung",
                info="Wählen Sie das Modell für die Quellenbewertung."
            )
            
            # NEW: Temperature control for LLM evaluation
            llm_assisted_temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=settings.LLM_ASSISTED_DEFAULT_TEMPERATURE,
                step=0.1,
                label="Temperatur",
                info="Bestimmt den Determinismus der KI-Bewertung. Niedrigere Werte = konsistentere Bewertungen."
            )
            
            # UPDATED: System prompt template selection and editing
            gr.Markdown("""
            ### System-Prompt für Quellenbewertung
            
            Wählen Sie eine Vorlage und bearbeiten Sie sie nach Ihren Bedürfnissen. Der System-Prompt steuert, 
            wie das LLM die Relevanz der Quellen bewertet.
            """)
            
            with gr.Row():
                llm_assisted_system_prompt_template = gr.Dropdown(
                    choices=list(settings.LLM_ASSISTED_SYSTEM_PROMPTS.keys()),
                    value="standard_evaluation",
                    label="Bewertungs-Prompt Vorlage",
                    info="Wählen Sie eine Vorlage als Ausgangspunkt"
                )
                
                reset_llm_assisted_system_prompt_btn = gr.Button("Auf Vorlage zurücksetzen", size="sm")
            
            # Editable system prompt text area
            llm_assisted_system_prompt_text = gr.Textbox(
                label="System-Prompt für Bewertung (bearbeitbar)",
                value=settings.LLM_ASSISTED_SYSTEM_PROMPTS["standard_evaluation"],
                lines=8,
                info="Bearbeiten Sie den System-Prompt für die Quellenbewertung, nach Ihren Bedürfnissen, spezifizieren Sie (in den Eckigen-Klammern) unbedingt die Forschungsfrage nach der Selektiert werden sollte."
            )
        
        llm_assisted_search_btn = gr.Button("LLM-Unterstützte Auswahl starten", variant="primary")
        
        # Progress indicator for LLM-assisted search
        llm_assisted_progress = gr.Markdown("", visible=False)
        llm_assisted_cancel_btn = gr.Button("Abbrechen", visible=False, variant="stop")
    
    # Results display (common for both modes)
    with gr.Group():
        search_status = gr.Markdown("Noch keine Suche durchgeführt.")
    
    # Connect preview functionality
    preview_btn.click(
        preview_callback,
        inputs=[keywords, semantic_expansion_factor],
        outputs=[expansion_output, expanded_words_state]
    )
    
    # Show/hide chunk options based on time interval selection
    def toggle_chunk_options(use_time_intervals_val):
        if use_time_intervals_val:
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)
    
    use_time_intervals.change(
        toggle_chunk_options,
        inputs=[use_time_intervals],
        outputs=[top_k, chunks_per_interval]
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
        outputs=[standard_settings, llm_assisted_settings]
    )
    
    # NEW: Calculate and display time interval breakdown
    def calculate_intervals(start_year, end_year, interval_size, use_intervals):
        if not use_intervals:
            return gr.update(value="", visible=False)
        
        interval_info = settings.calculate_time_intervals(start_year, end_year, interval_size)
        intervals_text = f"""**Berechnete Aufteilung**: {interval_info['coverage']}
        
**Intervalle**: {', '.join([f"{start}-{end}" for start, end in interval_info['intervals']])}"""
        
        return gr.update(value=intervals_text, visible=True)
    
    # Connect interval calculation to inputs
    for component in [year_start, year_end, time_interval_size, use_time_intervals]:
        component.change(
            calculate_intervals,
            inputs=[year_start, year_end, time_interval_size, use_time_intervals],
            outputs=[interval_calculation]
        )
    
    # LLM-assisted system prompt template management
    def load_llm_assisted_system_prompt_template(template_name: str) -> str:
        """Load the selected LLM-assisted template into the text area."""
        return settings.LLM_ASSISTED_SYSTEM_PROMPTS.get(template_name, settings.LLM_ASSISTED_SYSTEM_PROMPTS["standard_evaluation"])
    
    def reset_llm_assisted_to_template(template_name: str) -> str:
        """Reset the LLM-assisted text area to the selected template."""
        return settings.LLM_ASSISTED_SYSTEM_PROMPTS.get(template_name, settings.LLM_ASSISTED_SYSTEM_PROMPTS["standard_evaluation"])
    
    # Connect LLM-assisted template dropdown to text area
    llm_assisted_system_prompt_template.change(
        load_llm_assisted_system_prompt_template,
        inputs=[llm_assisted_system_prompt_template],
        outputs=[llm_assisted_system_prompt_text]
    )
    
    # Connect LLM-assisted reset button
    reset_llm_assisted_system_prompt_btn.click(
        reset_llm_assisted_to_template,
        inputs=[llm_assisted_system_prompt_template],
        outputs=[llm_assisted_system_prompt_text]
    )
    
    # Define all components to be returned - UPDATED component names
    components = {
        # Common components
        "search_mode": search_mode,
        "retrieval_query": retrieval_query,  # UPDATED from content_description
        "chunk_size": chunk_size,
        "year_start": year_start,
        "year_end": year_end,
        "search_status": search_status,
        "retrieved_chunks_state": retrieved_chunks_state,
        "expanded_words_state": expanded_words_state,
        
        # Standard search components
        "top_k": top_k,
        "chunks_per_interval": chunks_per_interval,  # UPDATED from chunks_per_window
        "keywords": keywords,
        "search_in": search_in,
        "use_semantic_expansion": use_semantic_expansion,
        "semantic_expansion_factor": semantic_expansion_factor,
        "use_time_intervals": use_time_intervals,  # UPDATED from use_time_windows
        "time_interval_size": time_interval_size,  # UPDATED from time_window_size
        "standard_search_btn": standard_search_btn,
        "expansion_output": expansion_output,
        "interval_calculation": interval_calculation,  # NEW
        
        # LLM-assisted search components - UPDATED terminology
        "llm_assisted_use_time_intervals": llm_assisted_use_time_intervals,
        "llm_assisted_time_interval_size": llm_assisted_time_interval_size,
        "chunks_per_interval_initial": chunks_per_interval_initial,
        "chunks_per_interval_final": chunks_per_interval_final,
        "llm_assisted_min_retrieval_score": llm_assisted_min_retrieval_score,
        "llm_assisted_keywords": llm_assisted_keywords,
        "llm_assisted_search_in": llm_assisted_search_in,
        "llm_assisted_model": llm_assisted_model,
        "llm_assisted_temperature": llm_assisted_temperature,  # NEW
        "llm_assisted_system_prompt_template": llm_assisted_system_prompt_template,
        "llm_assisted_system_prompt_text": llm_assisted_system_prompt_text,
        "reset_llm_assisted_system_prompt_btn": reset_llm_assisted_system_prompt_btn,
        "llm_assisted_search_btn": llm_assisted_search_btn,
        "llm_assisted_progress": llm_assisted_progress,
        "llm_assisted_cancel_btn": llm_assisted_cancel_btn,
        
        # UI groups for visibility control
        "standard_settings": standard_settings,
        "llm_assisted_settings": llm_assisted_settings,
        
        # DEPRECATED: Keep for backward compatibility with updated names
        "content_description": retrieval_query,  # Alias
        "chunks_per_window": chunks_per_interval,  # Alias
        "agent_use_time_windows": llm_assisted_use_time_intervals,  # Alias
        "agent_time_window_size": llm_assisted_time_interval_size,  # Alias
        "chunks_per_window_initial": chunks_per_interval_initial,  # Alias
        "chunks_per_window_final": chunks_per_interval_final,  # Alias
        "agent_min_retrieval_score": llm_assisted_min_retrieval_score,  # Alias
        "agent_keywords": llm_assisted_keywords,  # Alias
        "agent_search_in": llm_assisted_search_in,  # Alias
        "agent_model": llm_assisted_model,  # Alias
        "agent_system_prompt_template": llm_assisted_system_prompt_template,  # Alias
        "agent_system_prompt_text": llm_assisted_system_prompt_text,  # Alias
        "agent_search_btn": llm_assisted_search_btn,  # Alias
        "agent_progress": llm_assisted_progress,  # Alias
        "agent_cancel_btn": llm_assisted_cancel_btn,  # Alias
        "agent_settings": llm_assisted_settings  # Alias
    }
    
    return components