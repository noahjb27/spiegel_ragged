# src/ui/components/search_panel.py
"""
Search panel component for the Spiegel RAG application.
This component defines the UI elements for the search interface.
"""
import gradio as gr
from typing import Dict, Any, Callable

from src.config import settings

def create_search_panel(
    search_callback: Callable,
    preview_callback: Callable,
    toggle_api_key_callback: Callable
) -> Dict[str, Any]:
    """
    Create the search panel UI components.
    
    Args:
        search_callback: Function to call when search button is clicked
        preview_callback: Function to call when preview button is clicked
        toggle_api_key_callback: Function to toggle API key visibility
        
    Returns:
        Dictionary of UI components
    """
    # Hidden state for expanded words
    expanded_words_state = gr.State("")
    
    # Main search panel
    with gr.Group():
        gr.Markdown("## Suchanfrage")
        
        query = gr.Textbox(
            label="Suchanfrage (welche Inhalte gesucht werden sollen)",
            placeholder="Beispiel: Berichterstattung über die Berliner Mauer",
            lines=2,
            value="Berlin Mauer",
            info="Beschreiben Sie, welche Art von Inhalten Sie im Archiv finden möchten."
        )
        
        question = gr.Textbox(
            label="Frage (was Sie über die gefundenen Inhalte wissen möchten)",
            placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
            lines=2,
            value="Wie wurde die Berliner Mauer in den Medien dargestellt?",
            info="Formulieren Sie Ihre Frage, die anhand der gefundenen Texte beantwortet werden soll."
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
    
    search_btn = gr.Button("Suchen", variant="primary")

    # Model selection settings
    with gr.Accordion("Erweiterte Einstellungen", open=False):
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
        
        with gr.Row(visible=False) as openai_key_row:
            openai_api_key = gr.Textbox(
                label="OpenAI API-Schlüssel",
                placeholder="sk-...",
                type="password",
                info="Ihr OpenAI API-Schlüssel wird nur für diese Sitzung gespeichert"
            )

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
        "query": query,
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
        "search_btn": search_btn,
        "expansion_output": expansion_output,
        "openai_key_row": openai_key_row
    }
    
    return components