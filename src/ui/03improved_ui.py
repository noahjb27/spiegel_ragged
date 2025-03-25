# src/ui/improved_ui.py
import os
import sys
import logging
import json
from typing import Dict, List, Tuple, Optional
import time

import gradio as gr

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.rag_engine import SpiegelRAGEngine
from src.core.embedding_service import WordEmbeddingService
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize RAG engine and embedding service
try:
    logger.info("Initializing RAG Engine...")
    rag_engine = SpiegelRAGEngine()
    embedding_service = rag_engine.embedding_service
    logger.info("RAG Engine and Embedding Service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {e}")
    rag_engine = None
    embedding_service = None

# Helper functions
def find_similar_words(keyword: str, expansion_factor: int) -> str:
    """Find similar words for a given keyword using FastText embeddings."""
    if not keyword.strip():
        return "Please enter a keyword."
    
    if not embedding_service:
        return "Embedding service not available."
    
    try:
        # Get similar words
        similar_words = embedding_service.find_similar_words(keyword.strip(), top_n=expansion_factor)
        
        # Format results for display
        if not similar_words:
            return f"No similar words found for '{keyword}'"
        
        result = f"### Similar words for '{keyword}':\n\n"
        for word_info in similar_words:
            result += f"- **{word_info['word']}** (similarity: {word_info['similarity']:.4f})\n"
        
        return result
    except Exception as e:
        logger.error(f"Error finding similar words: {e}")
        return f"Error finding similar words: {str(e)}"

def expand_boolean_expression(expression: str, expansion_factor: int) -> Tuple[str, str]:
    """Expand a boolean expression with semantically similar words."""
    if not expression.strip():
        return "Please enter a boolean expression.", ""
    
    if not embedding_service:
        return "Embedding service not available.", ""
    
    try:
        # Parse the boolean expression
        parsed_terms = embedding_service.parse_boolean_expression(expression)
        
        # Expand terms with semantically similar words
        expanded_terms = embedding_service.filter_by_semantic_similarity(
            parsed_terms, 
            expansion_factor=expansion_factor
        )
        
        # Format the expanded terms for display
        display_result = f"## Expanded Keywords\n\n"
        
        # Save expanded words for potential use in search
        expanded_words = {}
        
        for category, terms in expanded_terms.items():
            if terms:
                display_result += f"### {category.capitalize()} Terms:\n\n"
                for term_data in terms:
                    original = term_data.get('original', '')
                    expanded = term_data.get('expanded', {}).get(original, [])
                    
                    display_result += f"**{original}** → "
                    if expanded:
                        expanded_list = [f"{item['word']} ({item['similarity']:.2f})" for item in expanded]
                        display_result += ", ".join(expanded_list)
                        
                        # Add to expanded_words for search
                        if original not in expanded_words:
                            expanded_words[original] = []
                        for item in expanded:
                            expanded_words[original].append(item['word'])
                    else:
                        display_result += "No similar words found."
                    display_result += "\n\n"
        
        # Prepare a JSON structure to be used later in search
        encoded_expanded = json.dumps(expanded_words)
        
        return display_result, encoded_expanded
    except Exception as e:
        logger.error(f"Error expanding boolean expression: {e}")
        return f"Error expanding expression: {str(e)}", ""

def perform_search_with_keywords(
    query: str,
    question: str,
    chunk_size: int,
    year_start: int,
    year_end: int,
    keywords: str,
    search_in: List[str],
    use_semantic_expansion: bool,
    semantic_expansion_factor: int,
    expanded_words_json: str,
    enforce_keywords: bool,
    use_time_windows: bool,
    time_window_size: int,
    model_selection: str,  # Add these parameters
    openai_api_key: str
) -> Tuple[str, str, str]:
    """Perform search with keyword filtering and semantic expansion."""
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", "", "Check the logs for details"
        
        start_time = time.time()
        logger.info(f"Starting search with keywords: query='{query}', question='{question}', keywords='{keywords}'")
        
        # Process keywords - only use if keywords are actually provided
        keywords_to_use = keywords.strip() if keywords and keywords.strip() else None
        
        # Only use expanded words if keywords are provided and expansion is enabled
        expanded_words = None
        if keywords_to_use and use_semantic_expansion and expanded_words_json:
            try:
                expanded_words = json.loads(expanded_words_json)
                logger.info(f"Using expanded words: {expanded_words}")
            except:
                logger.warning("Failed to parse expanded words JSON")
        elif not keywords_to_use:
            # Reset expanded words if no keywords provided
            expanded_words_json = ""
            expanded_words = None
        
        # Set search fields
        search_fields = search_in if search_in else ["Text"]
        
               # Handle model selection
        model_to_use = "hu-llm"  # Default
        if model_selection == "openai-gpt4o":
            model_to_use = "gpt-4o"
        elif model_selection == "openai-gpt35":
            model_to_use = "gpt-3.5-turbo"
        
        # Perform search
        results = rag_engine.search(
            question=question,
            content_description=query,
            year_range=[year_start, year_end],
            chunk_size=chunk_size,
            keywords=keywords_to_use,
            search_in=search_fields,
            model=model_to_use,  # Use selected model
            openai_api_key=openai_api_key,  # Pass API key
            use_query_refinement=False,
            use_iterative_search=use_time_windows,
            time_window_size=time_window_size,
            with_citations=False,
            use_semantic_expansion=use_semantic_expansion and keywords_to_use is not None,
            semantic_expansion_factor=semantic_expansion_factor,
            enforce_keywords=enforce_keywords
        )
        
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f} seconds")
        
        # Get actual number of chunks
        num_chunks = len(results.get('chunks', []))
        logger.info(f"Found {num_chunks} chunks")
        
        # Format results
        answer_text = results.get('answer', 'No answer generated')
        
        # Format retrieved chunks for display
        chunks_text = ""
        if results.get("chunks"):
            # Group chunks by year for better readability
            chunks_by_year = {}
            for chunk in results["chunks"]:
                year = chunk["metadata"].get("Jahrgang", "Unknown")
                if year not in chunks_by_year:
                    chunks_by_year[year] = []
                chunks_by_year[year].append(chunk)
            
            # Calculate time windows for better visualization
            time_windows = []
            if use_time_windows:
                for window_start in range(year_start, year_end + 1, time_window_size):
                    window_end = min(window_start + time_window_size - 1, year_end)
                    time_windows.append((window_start, window_end))
            
            # Display chunks grouped by time window if using time windows
            if use_time_windows and time_windows:
                chunks_text += "# Ergebnisse nach Zeitfenstern\n\n"
                
                # Group years into their respective time windows
                for window_start, window_end in time_windows:
                    window_label = f"## Zeitfenster {window_start}-{window_end}\n\n"
                    window_chunks = []
                    
                    # Collect chunks from years in this window
                    for year in sorted(chunks_by_year.keys()):
                        if isinstance(year, int) and window_start <= year <= window_end:
                            window_chunks.extend([(year, i, chunk) for i, chunk in enumerate(chunks_by_year[year])])
                    
                    # Only add window if it has chunks
                    if window_chunks:
                        chunks_text += window_label
                        
                        # Count chunks per year in this window for statistics
                        window_year_counts = {}
                        for y, _, _ in window_chunks:
                            window_year_counts[y] = window_year_counts.get(y, 0) + 1
                        
                        # Show year distribution within window
                        chunks_text += "**Verteilung:** "
                        chunks_text += ", ".join([f"{y}: {count} Texte" for y, count in sorted(window_year_counts.items())])
                        chunks_text += "\n\n"
                        
                        # Add each chunk under its year heading
                        current_year = None
                        chunk_in_year = 1
                        
                        for year, _, chunk in sorted(window_chunks):
                            # Add year subheading when year changes
                            if year != current_year:
                                chunks_text += f"### {year}\n\n"
                                current_year = year
                                chunk_in_year = 1
                            
                            metadata = chunk["metadata"]
                            chunks_text += f"#### {chunk_in_year}. {metadata.get('Artikeltitel', 'Kein Titel')}\n\n"
                            chunks_text += f"**Datum**: {metadata.get('Datum', 'Unbekannt')} | "
                            chunks_text += f"**Relevanz**: {chunk['relevance_score']:.3f}\n\n"
                            
                            # Highlight the keywords if found
                            if keywords_to_use:
                                content = chunk['content']
                                keywords_list = [k.strip().lower() for k in keywords_to_use.split('AND')]
                                if expanded_words:
                                    all_keywords = []
                                    for k in keywords_list:
                                        all_keywords.append(k)
                                        if k in expanded_words:
                                            all_keywords.extend(expanded_words[k])
                                    
                                    # Add a note about which keywords were found
                                    found_keywords = []
                                    for k in all_keywords:
                                        if k.lower() in content.lower():
                                            found_keywords.append(k)
                                    
                                    if found_keywords:
                                        chunks_text += f"**Schlagwörter gefunden**: {', '.join(found_keywords)}\n\n"
                            
                            chunks_text += f"**Text**:\n{chunk['content']}\n\n"
                            chunks_text += "---\n\n"
                            chunk_in_year += 1
                        
                        chunks_text += "\n"
                    else:
                        # No chunks found for this window
                        chunks_text += window_label
                        chunks_text += "Keine Texte gefunden in diesem Zeitfenster.\n\n"
            else:
                # Regular display by year when not using time windows
                for year in sorted(chunks_by_year.keys()):
                    chunks_text += f"## {year}\n\n"
                    for i, chunk in enumerate(chunks_by_year[year], 1):
                        metadata = chunk["metadata"]
                        chunks_text += f"### {i}. {metadata.get('Artikeltitel', 'Kein Titel')}\n\n"
                        chunks_text += f"**Datum**: {metadata.get('Datum', 'Unbekannt')} | "
                        chunks_text += f"**Relevanz**: {chunk['relevance_score']:.3f}\n\n"
                        
                        # Highlight the keywords if found
                        if keywords_to_use:
                            content = chunk['content']
                            keywords_list = [k.strip().lower() for k in keywords_to_use.split('AND')]
                            if expanded_words:
                                all_keywords = []
                                for k in keywords_list:
                                    all_keywords.append(k)
                                    if k in expanded_words:
                                        all_keywords.extend(expanded_words[k])
                                
                                # Add a note about which keywords were found
                                found_keywords = []
                                for k in all_keywords:
                                    if k.lower() in content.lower():
                                        found_keywords.append(k)
                                
                                if found_keywords:
                                    chunks_text += f"**Schlagwörter gefunden**: {', '.join(found_keywords)}\n\n"
                        
                        chunks_text += f"**Text**:\n{chunk['content']}\n\n"
                        chunks_text += "---\n\n"
        else:
            chunks_text = "Keine passenden Texte gefunden."
        
        # Format metadata
        metadata_text = f"""
## Suchparameter
- **Model**: {model_to_use}
- **Suchanfrage**: {query}
- **Frage**: {question}
- **Chunk-Größe**: {chunk_size} Zeichen
- **Zeitraum**: {year_start} - {year_end}
- **Suchzeit**: {search_time:.2f} Sekunden
- **Gefundene Texte**: {num_chunks}

## Schlagwort-Filter
- **Schlagwörter**: {keywords_to_use or "Keine"}
- **Suchbereiche**: {', '.join(search_fields)}
- **Strikte Filterung**: {"Ja" if enforce_keywords else "Nein"}
- **Semantische Erweiterung**: {"Aktiviert" if use_semantic_expansion and keywords_to_use else "Deaktiviert"}
"""
        if use_time_windows:
            metadata_text += f"\n## Zeitfenster-Suche\n- **Aktiviert**: Ja\n- **Fenstergröße**: {time_window_size} Jahre\n"
            
            # Create time windows
            time_windows = []
            for window_start in range(year_start, year_end + 1, time_window_size):
                window_end = min(window_start + time_window_size - 1, year_end)
                time_windows.append((window_start, window_end))
            
            # Count chunks per window
            window_counts = {}
            for chunk in results.get('chunks', []):
                year = chunk["metadata"].get("Jahrgang")
                if isinstance(year, int):
                    for i, (window_start, window_end) in enumerate(time_windows):
                        if window_start <= year <= window_end:
                            window_key = f"{window_start}-{window_end}"
                            window_counts[window_key] = window_counts.get(window_key, 0) + 1
                            break
            
            # Display window distribution
            metadata_text += "\n**Verteilung der Ergebnisse nach Zeitfenstern:**\n"
            for window_start, window_end in time_windows:
                window_key = f"{window_start}-{window_end}"
                count = window_counts.get(window_key, 0)
                metadata_text += f"- **{window_key}**: {count} Texte\n"
            
        if use_semantic_expansion and keywords_to_use and expanded_words:
            metadata_text += "\n## Erweiterte Schlagwörter\n"
            for original, similar in expanded_words.items():
                metadata_text += f"- **{original}** → {', '.join(similar)}\n"
            
        return answer_text, chunks_text, metadata_text
    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return f"Error: {str(e)}", "", "Search failed, check logs for details."

# Create enhanced Gradio interface with improved UI
with gr.Blocks(title="Der Spiegel RAG (1948-1979)") as app:
    gr.Markdown(
        """
        # Der Spiegel RAG (1948-1979)
        
        **Ein Retrieval-Augmented Generation System für die Analyse historischer Artikel des Spiegel-Archivs.**
        
        Mit diesem Tool können Sie das Spiegel-Archiv durchsuchen, relevante Inhalte abrufen und 
        KI-gestützte Analysen zu historischen Fragestellungen erhalten.
        """
    )
    
    # Hidden state for expanded words
    expanded_words_state = gr.State("")
    
    with gr.Tabs():
        with gr.TabItem("Archiv durchsuchen", id="search"):
            with gr.Row():
                with gr.Column(scale=1):
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
                        
                        with gr.Row():
                            gr.Markdown("""
                            **Hinweis:** Die Zeitfenster-Suche kann die Suchergebnisse ausgewogener gestalten, 
                            erfordert aber mehr Verarbeitungszeit aufgrund mehrerer Suchanfragen.
                            """)
                    search_btn = gr.Button("Suchen", variant="primary")

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

                        # Show the API key row only when OpenAI models are selected
                        def toggle_api_key_visibility(model_choice):
                            if model_choice.startswith("openai"):
                                return gr.update(visible=True)
                            return gr.update(visible=False)
                        
                        model_selection.change(
                            toggle_api_key_visibility,
                            inputs=[model_selection],
                            outputs=[openai_key_row]
                        )

   
                # Results panel
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("## Ergebnisse")
                        
                        with gr.Tabs():
                            with gr.TabItem("Analyse"):
                                answer_output = gr.Markdown(
                                    value="Die Antwort erscheint hier...",
                                    label="Analyse"
                                )
                            
                            with gr.TabItem("Gefundene Texte"):
                                chunks_output = gr.Markdown(
                                    value="Gefundene Textabschnitte erscheinen hier...",
                                    label="Gefundene Texte"
                                )
                            
                            with gr.TabItem("Metadaten"):
                                metadata_output = gr.Markdown(
                                    value="Metadaten zur Suche erscheinen hier...",
                                    label="Metadaten"
                                )
        
        with gr.TabItem("Schlagwort-Analyse", id="keyword_analysis"):
            gr.Markdown("""
            # Schlagwort-Analyse
            
            Hier können Sie ähnliche Wörter zu einem Suchbegriff finden, um Ihre Suchanfragen zu verbessern.
            
            Der FastText-Algorithmus findet Wörter mit ähnlicher Bedeutung basierend auf dem Kontext, 
            in dem sie im Spiegel-Archiv verwendet werden.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    single_keyword = gr.Textbox(
                        label="Suchbegriff",
                        placeholder="Geben Sie einen Begriff ein...",
                        lines=1
                    )
                    
                    single_expansion_factor = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Anzahl ähnlicher Wörter"
                    )
                    
                    single_word_btn = gr.Button("Ähnliche Wörter finden")
                
                with gr.Column(scale=1):
                    single_word_output = gr.Markdown(
                        label="Ähnliche Wörter",
                        value="Die Ergebnisse erscheinen hier..."
                    )
        
        with gr.TabItem("Info", id="info"):
            gr.Markdown("""
            # Über das Spiegel RAG-System
            
            Dieses System ermöglicht die Durchsuchung und Analyse von Der Spiegel-Artikeln aus den Jahren 1948 bis 1979 
            mithilfe von Retrieval-Augmented Generation (RAG).
            
            ## Was ist RAG?
            
            Retrieval-Augmented Generation ist ein Ansatz, der die Stärken von Suchsystemen mit 
            denen von großen Sprachmodellen kombiniert:
            
            1. **Retrieval**: Das System sucht zunächst relevante Textabschnitte aus dem Archiv
            2. **Generation**: Ein Sprachmodell analysiert diese Abschnitte und generiert eine Antwort
            
            ## Hauptfunktionen
            
            - **Semantische Suche**: Findet Inhalte basierend auf ihrer Bedeutung, nicht nur nach Schlüsselwörtern
            - **Schlagwort-Filterung**: Verwenden Sie boolesche Operatoren (AND, OR, NOT) für präzise Filterung
            - **Semantische Erweiterung**: Findet und berücksichtigt automatisch ähnliche Begriffe
            - **Zeitfenster-Suche**: Analysiert Inhalte über verschiedene Zeitperioden hinweg
            - **Anpassbare Textgrößen**: Optimieren Sie die Suche mit verschiedenen Chunk-Größen
            
            ## Tipps für optimale Ergebnisse
            
            1. **Präzise Suchanfragen**: Je genauer Ihre Suchanfrage, desto relevanter die Ergebnisse
            2. **Konkrete Fragen**: Formulieren Sie spezifische Fragen zu den gesuchten Inhalten
            3. **Schlagwort-Filterung**: Nutzen Sie die Filterung, um irrelevante Ergebnisse auszuschließen
            4. **Semantische Erweiterung**: Aktivieren Sie diese Option, um auch Texte mit ähnlichen Begriffen zu finden
            5. **Zeitfenster-Suche**: Besonders nützlich, um zeitliche Entwicklungen zu analysieren
            
            ## Anwendungsbeispiele
            
            - **Historische Analysen**: Wie hat sich die Berichterstattung zu einem Thema über die Zeit verändert?
            - **Medienkritik**: Wie wurden bestimmte Ereignisse im Spiegel dargestellt?
            - **Diskursanalyse**: Welche Begriffe und Konzepte wurden im Zusammenhang mit einem Thema verwendet?
            - **Ereignisrecherche**: Wie wurde über ein spezifisches historisches Ereignis berichtet?
            
            ## Datengrundlage
            
            Die Datenbank enthält Der Spiegel-Artikel aus den Jahren 1948 bis 1979, die in aus dem Spiegel-Archiv gescraped wurden.
            """)
    
    # Set up event handlers
    preview_btn.click(
        expand_boolean_expression,
        inputs=[keywords, semantic_expansion_factor],
        outputs=[expansion_output, expanded_words_state]
    )
    
    # search button click handler
    search_btn.click(
        perform_search_with_keywords,
        inputs=[
            query, question, chunk_size, year_start, year_end,
            keywords, search_in, use_semantic_expansion,
            semantic_expansion_factor, expanded_words_state,
            enforce_keywords, use_time_windows, time_window_size,
            model_selection, openai_api_key  
        ],
        outputs=[answer_output, chunks_output, metadata_output]
    )
    
    single_word_btn.click(
        find_similar_words,
        inputs=[single_keyword, single_expansion_factor],
        outputs=[single_word_output]
    )

# Run the app
if __name__ == "__main__":
    logger.info("Starting improved Gradio UI...")
    app.launch(share=False)