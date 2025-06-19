# src/ui/handlers/search_handlers.py - Enhanced with chunk selection and chunks per window
"""
Enhanced handler functions for search operations with chunk selection and improved time window support.
"""
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any
import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.engine import SpiegelRAG
from src.core.search.strategies import (
    StandardSearchStrategy, 
    TimeWindowSearchStrategy, 
    AgentSearchStrategy,
    SearchConfig
)
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reference to the RAG engine
rag_engine = None

def set_rag_engine(engine: Any) -> None:
    """Set the global RAG engine reference."""
    global rag_engine
    rag_engine = engine

def perform_retrieval(
    content_description: str,
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
    top_k: int,
    chunks_per_window: int = 5  # ENHANCED: New parameter for chunks per window
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform source retrieval using the enhanced strategy-based approach with chunks per window support.
    """
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", None
        
        start_time = time.time()
        
        # Clean and validate parameters
        content_description = content_description.strip() if content_description else ""
        if not content_description:
            return "Error: Content description is required", None
        
        keywords_cleaned = keywords.strip() if keywords else None
        search_fields = search_in if search_in else ["Text"]
        
        logger.info(f"Starting retrieval with time windows: {use_time_windows}")
        logger.info(f"Content: '{content_description}', Keywords: '{keywords_cleaned}'")
        
        # ENHANCED: Determine effective top_k based on time window usage
        if use_time_windows:
            # Calculate expected number of windows
            year_span = year_end - year_start + 1
            num_windows = max(1, (year_span + time_window_size - 1) // time_window_size)
            effective_top_k = chunks_per_window * num_windows
            logger.info(f"Time windows: {num_windows} windows × {chunks_per_window} chunks = {effective_top_k} total target")
        else:
            effective_top_k = top_k
            logger.info(f"Standard search: {effective_top_k} chunks total")
        
        # Create search configuration
        config = SearchConfig(
            content_description=content_description,
            year_range=(year_start, year_end),
            chunk_size=chunk_size,
            keywords=keywords_cleaned,
            search_fields=search_fields,
            enforce_keywords=enforce_keywords,
            top_k=effective_top_k
        )
        
        # Choose strategy based on user selection
        if use_time_windows:
            logger.info(f"Using TimeWindowSearchStrategy with window size: {time_window_size}")
            # ENHANCED: Create custom strategy with chunks per window logic
            strategy = EnhancedTimeWindowSearchStrategy(
                window_size=time_window_size,
                chunks_per_window=chunks_per_window
            )
        else:
            logger.info("Using StandardSearchStrategy")
            strategy = StandardSearchStrategy()
        
        # Execute search with strategy
        search_result = rag_engine.search(
            strategy=strategy,
            config=config,
            use_semantic_expansion=use_semantic_expansion and keywords_cleaned is not None
        )
        
        retrieval_time = time.time() - start_time
        
        # Convert SearchResult to the format expected by the UI
        chunks_for_ui = []
        for doc, score in search_result.chunks:
            chunks_for_ui.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': score
            })
        
        # Build result dictionary in expected format
        results = {
            'chunks': chunks_for_ui,
            'metadata': {
                'search_time': retrieval_time,
                'strategy': search_result.metadata.get('strategy', 'unknown'),
                'chunk_size': chunk_size,
                'year_range': [year_start, year_end],
                'keywords': keywords_cleaned,
                'retrieval_method': 'time_window' if use_time_windows else 'standard',
                'chunks_per_window': chunks_per_window if use_time_windows else None,  # ENHANCED
                'effective_top_k': effective_top_k  # ENHANCED
            }
        }
        
        # Add strategy-specific metadata
        results['metadata'].update(search_result.metadata)
        
        num_chunks = len(chunks_for_ui)
        logger.info(f"Retrieval completed: {num_chunks} chunks in {retrieval_time:.2f}s")
        
        # Create info message
        method_name = "Zeitfenster-Suche" if use_time_windows else "Standard-Suche"
        info_text = f"""
        ### Quellen erfolgreich abgerufen ({method_name})
        
        **Inhaltsbeschreibung**: {content_description}  
        **Zeitraum**: {year_start} - {year_end}  
        **Anzahl gefundener Quellen**: {num_chunks}  
        **Abrufzeit**: {retrieval_time:.2f} Sekunden
        """
        
        if use_time_windows:
            windows = search_result.metadata.get('windows', [])
            info_text += f"""
            **Zeitfenster**: {len(windows)} Fenster à {time_window_size} Jahre
            **Chunks pro Fenster**: {chunks_per_window}
            **Ziel-Gesamtzahl**: {effective_top_k}
            **Fenster**: {', '.join([f"{w[0]}-{w[1]}" for w in windows[:5]])}{'...' if len(windows) > 5 else ''}
            """
        
        if keywords_cleaned:
            info_text += f"""
            **Schlagwörter**: {keywords_cleaned}
            **Semantische Erweiterung**: {'Ja' if use_semantic_expansion else 'Nein'}
            """
        
        if num_chunks == 0:
            info_text = f"""
            ### Keine passenden Quellen gefunden
            
            Versuchen Sie es mit einer anderen Inhaltsbeschreibung oder erweitern Sie die Filter.
            - Zeitraum: {year_start} - {year_end}
            - Methode: {method_name}
            """
        
        return info_text, results
        
    except Exception as e:
        logger.error(f"Error in retrieval: {e}", exc_info=True)
        return f"Error: {str(e)}", None

def perform_analysis(
    question: str,
    retrieved_chunks: Dict[str, Any],
    model_selection: str,
    system_prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    selected_chunk_ids: Optional[List[int]] = None  # ENHANCED: New parameter for chunk selection
) -> Tuple[str, str, str]:
    """
    Perform analysis on retrieved chunks with optional chunk selection.
    ENHANCED: Now supports filtering chunks by selected IDs.
    """
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", "", "Check the logs for details"
        
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            return "Bitte rufen Sie zuerst Quellen ab", "", "Keine Quellen abgerufen"
        
        start_time = time.time()
        logger.info(f"Starting analysis with question: '{question}'")
        logger.info(f"Using system prompt: {system_prompt[:100]}...")
        
        # Convert UI chunks back to Document format
        from langchain.docstore.document import Document
        
        chunks_data = retrieved_chunks.get('chunks', [])
        
        # ENHANCED: Filter chunks if specific IDs are selected
        if selected_chunk_ids is not None:
            logger.info(f"Filtering chunks by selected IDs: {selected_chunk_ids}")
            filtered_chunks = []
            
            for chunk_id in selected_chunk_ids:
                # Convert to 0-based index
                index = chunk_id - 1
                if 0 <= index < len(chunks_data):
                    filtered_chunks.append(chunks_data[index])
                else:
                    logger.warning(f"Chunk ID {chunk_id} out of range (1-{len(chunks_data)})")
            
            if not filtered_chunks:
                return "Error: Keine der ausgewählten Chunk-IDs sind gültig", "", "Chunk-Auswahl fehlgeschlagen"
            
            chunks_data = filtered_chunks
            logger.info(f"Using {len(chunks_data)} selected chunks for analysis")
        else:
            logger.info(f"Using all {len(chunks_data)} retrieved chunks for analysis")
        
        # Convert to Document objects
        documents = []
        for chunk_data in chunks_data:
            doc = Document(
                page_content=chunk_data['content'],
                metadata=chunk_data['metadata']
            )
            documents.append(doc)
        
        logger.info(f"Converted {len(documents)} chunks for analysis")
        
        # Perform analysis with new engine using the provided system prompt directly
        analysis_result = rag_engine.analyze(
            question=question,
            chunks=documents,
            model=model_selection,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # Format results
        answer_text = analysis_result.answer
        chunks_text = format_chunks(chunks_data, selected_chunk_ids=selected_chunk_ids)
        
        # Format metadata with chunk selection info
        chunk_selection_info = ""
        if selected_chunk_ids is not None:
            chunk_selection_info = f"""
        ## Quellenauswahl
        - **Verwendete Chunks**: {len(chunks_data)} von {retrieved_chunks.get('metadata', {}).get('total_chunks', len(retrieved_chunks.get('chunks', [])))}
        - **Ausgewählte IDs**: {', '.join(map(str, selected_chunk_ids))}
        """
        
        metadata_text = f"""
        ## Analyseparameter
        - **Model**: {analysis_result.model}
        - **Frage**: {question}
        - **Analysezeit**: {analysis_time:.2f} Sekunden
        - **Temperatur**: {temperature}
        - **Max Tokens**: {max_tokens or "Standardwert"}
        {chunk_selection_info}

        ## System Prompt (verwendet)
        ```
        {system_prompt[:500]}{'...' if len(system_prompt) > 500 else ''}
        ```

        ## Quellen-Metadaten
        - **Anzahl Quellen**: {len(documents)}
        - **Retrieval-Methode**: {retrieved_chunks.get('metadata', {}).get('retrieval_method', 'Unbekannt')}
        - **Original-Suchzeit**: {retrieved_chunks.get('metadata', {}).get('search_time', 0):.2f} Sekunden
        """
        
        return answer_text, chunks_text, metadata_text
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        return f"Error: {str(e)}", "", "Analysis failed, check logs for details."

def perform_retrieval_and_update_ui(
    content_description: str,
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
    top_k: int,
    chunks_per_window: int = 5  # ENHANCED: New parameter
) -> Tuple[str, Dict[str, Any], str, gr.Accordion, gr.Accordion, gr.Accordion]:
    """
    Perform retrieval and update UI accordions with enhanced chunks per window support.
    """
    # Perform the retrieval
    info_text, retrieved_chunks = perform_retrieval(
        content_description, chunk_size, year_start, year_end,
        keywords, search_in, use_semantic_expansion,
        semantic_expansion_factor, expanded_words_json,
        enforce_keywords, use_time_windows, time_window_size, 
        top_k, chunks_per_window  # ENHANCED: Pass chunks per window
    )
    
    # Format the retrieved chunks for display
    if retrieved_chunks and retrieved_chunks.get('chunks'):
        formatted_chunks = format_chunks(
            retrieved_chunks.get('chunks'),
            keywords_to_use=keywords,
            use_time_windows=use_time_windows,
            time_window_size=time_window_size,
            year_start=year_start,
            year_end=year_end,
            chunks_per_window=chunks_per_window  # ENHANCED: Pass for display info
        )
        num_chunks = len(retrieved_chunks.get('chunks'))
        
        # ENHANCED: Add chunk count info to metadata for chunk selection
        retrieved_chunks['metadata']['total_chunks'] = num_chunks
    else:
        formatted_chunks = "Keine Texte gefunden."
        num_chunks = 0
    
    # Update UI accordions based on retrieval success
    if num_chunks > 0:
        retrieval_state = gr.update(open=False)
        retrieved_texts_state = gr.update(open=True)
        question_state = gr.update(open=True)
    else:
        retrieval_state = gr.update(open=True)
        retrieved_texts_state = gr.update(open=False)
        question_state = gr.update(open=False)
    
    return info_text, retrieved_chunks, formatted_chunks, retrieval_state, retrieved_texts_state, question_state

def perform_analysis_and_update_ui(
    question: str,
    retrieved_chunks: Dict[str, Any],
    model_selection: str,
    system_prompt_template: str,
    system_prompt_text: str,
    temperature: float,
    max_tokens: int,
    chunk_selection_mode: str = "all",  # ENHANCED: New parameter
    selected_chunks_state: Optional[List[int]] = None  # ENHANCED: Selected chunk IDs
) -> Tuple[str, str, gr.Accordion, gr.Accordion]:
    """
    Perform analysis and update UI accordions with chunk selection support.
    ENHANCED: Now supports filtering by selected chunk IDs.
    """
    # Use the edited system prompt text directly
    system_prompt = system_prompt_text.strip()
    
    # Fallback to default if somehow empty
    if not system_prompt:
        logger.warning("System prompt text is empty, falling back to default")
        system_prompt = settings.SYSTEM_PROMPTS["default"]
    
    logger.info(f"Using system prompt text directly (length: {len(system_prompt)} chars)")
    
    # ENHANCED: Determine which chunks to use based on selection mode
    selected_chunk_ids = None
    if chunk_selection_mode == "upload" or chunk_selection_mode == "manual":
        selected_chunk_ids = selected_chunks_state
        if selected_chunk_ids:
            logger.info(f"Using chunk selection mode '{chunk_selection_mode}' with {len(selected_chunk_ids)} selected chunks")
        else:
            logger.warning(f"Chunk selection mode '{chunk_selection_mode}' specified but no chunks selected, using all")
    
    # Perform the analysis
    answer_text, chunks_text, metadata_text = perform_analysis(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model_selection=model_selection,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        selected_chunk_ids=selected_chunk_ids  # ENHANCED: Pass selected chunk IDs
    )
    
    # Update UI accordions
    question_state = gr.update(open=False)
    results_state = gr.update(open=True)
    
    return answer_text, metadata_text, question_state, results_state

def format_chunks(
    chunks: List[Dict],
    keywords_to_use: Optional[str] = None, 
    expanded_words: Optional[Dict] = None,
    use_time_windows: bool = False, 
    time_window_size: int = 5, 
    year_start: int = 1948, 
    year_end: int = 1979,
    chunks_per_window: Optional[int] = None,  # ENHANCED: New parameter
    selected_chunk_ids: Optional[List[int]] = None  # ENHANCED: For showing selection
) -> str:
    """
    Format retrieved chunks for display with enhanced time window and selection support.
    """
    if not chunks:
        return "Keine passenden Texte gefunden."
    
    # ENHANCED: Add selection info if applicable
    selection_info = ""
    if selected_chunk_ids is not None:
        selection_info = f"**Hinweis**: Nur {len(chunks)} von ursprünglich gefundenen Texten werden angezeigt (ausgewählte IDs: {', '.join(map(str, selected_chunk_ids))})\n\n"
    
    # Group chunks by year for better readability
    chunks_by_year = {}
    for i, chunk in enumerate(chunks):
        year = chunk["metadata"].get("Jahrgang", "Unknown")
        if year not in chunks_by_year:
            chunks_by_year[year] = []
        # ENHANCED: Add original index for chunk ID tracking
        chunk_with_id = chunk.copy()
        chunk_with_id['display_id'] = i + 1
        chunks_by_year[year].append(chunk_with_id)
    
    chunks_text = selection_info
    
    # Display chunks grouped by time window if using time windows
    if use_time_windows:
        chunks_text += "# Ergebnisse nach Zeitfenstern\n\n"
        
        # ENHANCED: Show chunks per window info
        if chunks_per_window:
            chunks_text += f"**Konfiguration**: {chunks_per_window} Chunks pro Zeitfenster\n\n"
        
        # Create time windows
        time_windows = []
        for window_start in range(year_start, year_end + 1, time_window_size):
            window_end = min(window_start + time_window_size - 1, year_end)
            time_windows.append((window_start, window_end))
        
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
                
                # Count chunks per year in this window
                window_year_counts = {}
                for y, _, _ in window_chunks:
                    window_year_counts[y] = window_year_counts.get(y, 0) + 1
                
                # Show year distribution within window
                chunks_text += "**Verteilung:** "
                chunks_text += ", ".join([f"{y}: {count} Texte" for y, count in sorted(window_year_counts.items())])
                chunks_text += "\n\n"
                
                # Add each chunk
                current_year = None
                chunk_in_year = 1
                
                for year, _, chunk in sorted(window_chunks):
                    if year != current_year:
                        chunks_text += f"### {year}\n\n"
                        current_year = year
                        chunk_in_year = 1
                    
                    metadata = chunk["metadata"]
                    display_id = chunk.get('display_id', '?')
                    chunks_text += f"#### {chunk_in_year}. {metadata.get('Artikeltitel', 'Kein Titel')} (ID: {display_id})\n\n"
                    chunks_text += f"**Datum**: {metadata.get('Datum', 'Unbekannt')} | "
                    chunks_text += f"**Relevanz**: {chunk['relevance_score']:.3f}"
                    
                    url = metadata.get('URL')
                    if url and url != 'Keine URL':
                        chunks_text += f" | [**Link zum Artikel**]({url})"
                    
                    chunks_text += "\n\n"
                    chunks_text += f"**Text**:\n{chunk['content']}\n\n"
                    chunks_text += "---\n\n"
                    chunk_in_year += 1
                
                chunks_text += "\n"
            else:
                chunks_text += window_label
                chunks_text += "Keine Texte gefunden in diesem Zeitfenster.\n\n"
    else:
        # Regular display by year when not using time windows
        chunks_text += f"# Gefundene Texte ({len(chunks)})\n\n"
        
        for year in sorted(chunks_by_year.keys()):
            chunks_text += f"## {year}\n\n"
            for i, chunk in enumerate(chunks_by_year[year], 1):
                metadata = chunk["metadata"]
                display_id = chunk.get('display_id', '?')
                chunks_text += f"### {i}. {metadata.get('Artikeltitel', 'Kein Titel')} (ID: {display_id})\n\n"
                chunks_text += f"**Datum**: {metadata.get('Datum', 'Unbekannt')} | "
                chunks_text += f"**Relevanz**: {chunk['relevance_score']:.3f}"
                
                url = metadata.get('URL')
                if url and url != 'Keine URL':
                    chunks_text += f" | [**Link zum Artikel**]({url})"
                
                chunks_text += "\n\n"
                chunks_text += f"**Text**:\n{chunk['content']}\n\n"
                chunks_text += "---\n\n"
    
    return chunks_text


# ENHANCED: Custom time window strategy with chunks per window control
class EnhancedTimeWindowSearchStrategy:
    """Enhanced time window search strategy with chunks per window control."""
    
    def __init__(self, window_size: int = 5, chunks_per_window: int = 5):
        self.window_size = window_size
        self.chunks_per_window = chunks_per_window
    
    def search(self, config, vector_store, progress_callback=None):
        """Execute time-windowed search with controlled chunks per window."""
        from src.core.search.strategies import SearchResult
        
        start_time = time.time()
        start_year, end_year = config.year_range
        
        # Create time windows
        windows = []
        for window_start in range(start_year, end_year + 1, self.window_size):
            window_end = min(window_start + self.window_size - 1, end_year)
            windows.append((window_start, window_end))
        
        logger.info(f"Enhanced time window search: {len(windows)} windows, {self.chunks_per_window} chunks per window")
        
        all_chunks = []
        window_counts = {}
        
        # Search each window
        for i, (window_start, window_end) in enumerate(windows):
            if progress_callback:
                progress = (i / len(windows))
                progress_callback(f"Searching {window_start}-{window_end}...", progress)
            
            # Create window-specific filter
            window_filter = vector_store.build_metadata_filter(
                year_range=[window_start, window_end],
                keywords=None,
                search_in=None
            )
            
            try:
                # ENHANCED: Use chunks_per_window instead of total k
                window_chunks = vector_store.similarity_search(
                    query=config.content_description,
                    chunk_size=config.chunk_size,
                    k=self.chunks_per_window,  # Use per-window limit
                    filter_dict=window_filter,
                    min_relevance_score=0.3,
                    keywords=config.keywords,
                    search_in=config.search_fields,
                    enforce_keywords=config.enforce_keywords
                )
                
                window_key = f"{window_start}-{window_end}"
                window_counts[window_key] = len(window_chunks)
                
                # Add window metadata to each chunk
                for doc, score in window_chunks:
                    doc.metadata['time_window'] = window_key
                    doc.metadata['window_start'] = window_start
                    doc.metadata['window_end'] = window_end
                
                all_chunks.extend(window_chunks)
                
            except Exception as e:
                logger.error(f"Error searching window {window_start}-{window_end}: {e}")
                window_counts[f"{window_start}-{window_end}"] = 0
        
        # Sort by relevance score
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            chunks=all_chunks,
            metadata={
                "strategy": "enhanced_time_window",
                "search_time": search_time,
                "window_size": self.window_size,
                "chunks_per_window": self.chunks_per_window,
                "windows": windows,
                "window_counts": window_counts,
                "total_chunks_found": len(all_chunks)
            }
        )