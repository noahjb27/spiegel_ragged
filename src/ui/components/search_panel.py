# src/ui/components/search_panel.py
"""
Search panel component for the Spiegel RAG application.
This component defines the UI elements for the search interface.
Refactored to support separate retrieval and analysis steps.
"""
import gradio as gr
from typing import Dict, Any, Callable

from src.config import settings

def create_search_panel(
    retrieve_callback: Callable,
    analyze_callback: Callable,
    preview_callback: Callable,
    toggle_api_key_callback: Callable
) -> Dict[str, Any]:
    """
    Create the search panel UI components.
    
    Args:
        retrieve_callback: Function to call when retrieve button is clicked
        analyze_callback: Function to call when analyze button is clicked
        preview_callback: Function to call when preview button is clicked
        toggle_api_key_callback: Function to toggle API key visibility
        
    Returns:
        Dictionary of UI components
    """
    # Hidden state for expanded words and retrieved chunks
    expanded_words_state = gr.State("")
    retrieved_chunks_state = gr.State(None)
    
    # Create tabs for the two main steps
    with gr.Tabs() as tabs:
        # Step 1: Retrieve Sources
        with gr.TabItem("1. Quellen abrufen", id="retrieve_tab"):
            # Content Description
            with gr.Group():
                gr.Markdown("## Quellen abrufen")
                
                content_description = gr.Textbox(
                    label="Inhaltsbeschreibung (welche Quellen gesucht werden sollen)",
                    placeholder="Beispiel: Berichterstattung über die Berliner Mauer",
                    lines=2,
                    info="Beschreiben Sie, welche Art von Inhalten Sie im Archiv finden möchten."
                )
            
            # Basic settings in an accordion
            with gr.Accordion("Grundeinstellungen", open=True):
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
                        choices=[2000, 3000],
                        value=3000,
                        label="Textgröße",
                        info="Größe der Textabschnitte in Zeichen. Kleinere Abschnitte sind präziser, größere bieten mehr Kontext."
                    )

                    top_k = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Anzahl Ergebnisse",
                        info="Maximale Anzahl der Texte, die abgerufen werden sollen."
                    )
            
            # Keyword filtering options
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
                
                **Vorteile:**
                - Bessere Abdeckung verschiedener Zeiträume
                - Erfassung der zeitlichen Entwicklung von Themen
                - Vermeidung einer Dominanz bestimmter Zeiträume
                
                **Beispiel:** Bei einem Zeitraum von 1960-1970 mit einer Fenstergröße von 5 Jahren wird die Suche in zwei Fenster unterteilt:
                - 1960-1964
                - 1965-1970
                
                Aus jedem Zeitfenster werden die relevantesten Ergebnisse ausgewählt.
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
            
            retrieve_btn = gr.Button("Quellen abrufen", variant="primary")
            retrieved_info = gr.Markdown("Noch keine Quellen abgerufen.")
        
        # Step 2: Analyze Content
        with gr.TabItem("2. Quellen analysieren", id="analyze_tab"):
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
                    - **HU-LLM**: Lokales Modell 
                                (kein API-Schlüssel erforderlich, HU-Netzwerk erforderlich)
                    - **OpenAI GPT-4o**: Leistungsstärkstes OpenAI-Modell 
                                (erfordert API-Schlüssel)
                    - **OpenAI GPT-3.5 Turbo**: Schnelles OpenAI-Modell 
                                (erfordert API-Schlüssel)
                    """)
                    
                    with gr.Row():
                        model_selection = gr.Radio(
                            choices=["hu-llm", "openai-gpt4o", "openai-gpt35"],
                            value="hu-llm",
                            label="LLM-Modell",
                            info="Wählen Sie das zu verwendende Sprachmodell"
                        )

                                        # Add system prompt selection
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
                    
                    with gr.Row(visible=False) as openai_key_row:
                        openai_api_key = gr.Textbox(
                            label="OpenAI API-Schlüssel",
                            placeholder="sk-...",
                            type="password",
                            info="Ihr OpenAI API-Schlüssel wird nur für diese Sitzung gespeichert"
                        )
                
                analyze_btn = gr.Button("Frage beantworten", variant="primary")
            
    # Connect events
    model_selection.change(
        toggle_api_key_callback,
        inputs=[model_selection],
        outputs=[openai_key_row]
    )
    
    preview_btn.click(
        preview_callback,
        inputs=[keywords, semantic_expansion_factor],
        outputs=[expansion_output, expanded_words_state]
    )
    
    # Define all components to be returned
    components = {
        "tabs": tabs,
        "content_description": content_description,
        "question": question,
        "chunk_size": chunk_size,
        "year_start": year_start,
        "year_end": year_end,
        "keywords": keywords,
        "search_in": search_in,
        "use_semantic_expansion": use_semantic_expansion,
        "semantic_expansion_factor": semantic_expansion_factor,
        "expanded_words_state": expanded_words_state,
        "enforce_keywords": enforce_keywords,
        "use_time_windows": use_time_windows,
        "time_window_size": time_window_size,
        "model_selection": model_selection,
        "openai_api_key": openai_api_key,
        "retrieve_btn": retrieve_btn,
        "analyze_btn": analyze_btn,
        "retrieved_info": retrieved_info,
        "retrieved_chunks_state": retrieved_chunks_state,
        "expansion_output": expansion_output,
        "openai_key_row": openai_key_row,
        "top_k": top_k,
        "system_prompt_template": system_prompt_template,
        "custom_system_prompt": custom_system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    return components