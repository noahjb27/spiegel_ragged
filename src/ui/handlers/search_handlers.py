# src/ui/handlers/search_handlers.py - Updated with new terminology and chunk selection
"""
Updated handler functions for search operations with new terminology:
- Retrieval-Query (instead of content_description)
- Zeitintervall-Suche (instead of Zeitfenster-Suche)
- LLM-Unterstützte Auswahl (instead of Agent search)
- Enhanced chunk selection functionality
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
    retrieval_query: str,  # UPDATED: from content_description
    chunk_size: int,
    year_start: int,
    year_end: int,
    keywords: str,
    search_in: List[str],
    use_semantic_expansion: bool,
    semantic_expansion_factor: int,
    expanded_words_json: str,
    use_time_intervals: bool,  # UPDATED: from use_time_windows
    time_interval_size: int,   # UPDATED: from time_window_size
    top_k: int,
    chunks_per_interval: int = 5  # UPDATED: from chunks_per_window
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform source retrieval using the updated strategy-based approach with Zeitintervall-Suche.
    """
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", None
        
        start_time = time.time()
        
        # Clean and validate parameters
        retrieval_query = retrieval_query.strip() if retrieval_query else ""
        if not retrieval_query:
            return "Error: Retrieval-Query ist erforderlich", None
        
        keywords_cleaned = keywords.strip() if keywords else None
        search_fields = search_in if search_in else ["Text"]
        
        logger.info(f"Starting retrieval with Zeitintervall-Suche: {use_time_intervals}")
        logger.info(f"Retrieval-Query: '{retrieval_query}', Keywords: '{keywords_cleaned}'")
        
        # Determine effective top_k based on time interval usage
        if use_time_intervals:
            # Calculate expected number of intervals
            year_span = year_end - year_start + 1
            num_intervals = max(1, (year_span + time_interval_size - 1) // time_interval_size)
            effective_top_k = chunks_per_interval * num_intervals
            logger.info(f"Zeit-Intervalle: {num_intervals} Intervalle × {chunks_per_interval} chunks = {effective_top_k} total target")
        else:
            effective_top_k = top_k
            logger.info(f"Standard search: {effective_top_k} chunks total")
        
        # Create search configuration
        config = SearchConfig(
            content_description=retrieval_query,  # Map new terminology to existing field
            year_range=(year_start, year_end),
            chunk_size=chunk_size,
            keywords=keywords_cleaned,
            search_fields=search_fields,
            enforce_keywords=True,  # Always enforce keywords now
            top_k=effective_top_k
        )
        
        # Choose strategy based on user selection
        if use_time_intervals:
            logger.info(f"Using Zeitintervall-SearchStrategy with interval size: {time_interval_size}")
            # Create custom strategy with chunks per interval logic
            strategy = EnhancedTimeIntervalSearchStrategy(
                interval_size=time_interval_size,
                chunks_per_interval=chunks_per_interval
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
                'retrieval_method': 'zeit_interval' if use_time_intervals else 'standard',  # UPDATED
                'chunks_per_interval': chunks_per_interval if use_time_intervals else None,  # UPDATED
                'effective_top_k': effective_top_k
            }
        }
        
        # Add strategy-specific metadata
        results['metadata'].update(search_result.metadata)
        
        num_chunks = len(chunks_for_ui)
        logger.info(f"Retrieval completed: {num_chunks} chunks in {retrieval_time:.2f}s")
        
        # Create info message with updated terminology
        method_name = "Zeitintervall-Suche" if use_time_intervals else "Standard-Suche"
        info_text = f"""
        ### Quellen erfolgreich abgerufen ({method_name})
        
        **Retrieval-Query**: {retrieval_query}  
        **Zeitraum**: {year_start} - {year_end}  
        **Anzahl gefundener Quellen**: {num_chunks}  
        **Abrufzeit**: {retrieval_time:.2f} Sekunden
        """
        
        if use_time_intervals:
            intervals = search_result.metadata.get('intervals', [])  # UPDATED from windows
            info_text += f"""
            **Zeit-Intervalle**: {len(intervals)} Intervalle à {time_interval_size} Jahre
            **Chunks pro Intervall**: {chunks_per_interval}
            **Ziel-Gesamtzahl**: {effective_top_k}
            **Intervalle**: {', '.join([f"{w[0]}-{w[1]}" for w in intervals[:5]])}{'...' if len(intervals) > 5 else ''}
            """
        
        if keywords_cleaned:
            info_text += f"""
            **Schlagwörter**: {keywords_cleaned}
            **Semantische Erweiterung**: {'Aktiviert' if use_semantic_expansion else 'Deaktiviert'}
            """
        
        if num_chunks == 0:
            info_text = f"""
            ### Keine passenden Quellen gefunden
            
            Versuchen Sie es mit einer anderen Retrieval-Query oder erweitern Sie die Filter.
            - Zeitraum: {year_start} - {year_end}
            - Methode: {method_name}
            """
        
        return info_text, results
        
    except Exception as e:
        logger.error(f"Error in retrieval: {e}", exc_info=True)
        return f"Error: {str(e)}", None

def perform_analysis(
    user_prompt: str,  # UPDATED: from question
    retrieved_chunks: Dict[str, Any],
    model_selection: str,
    system_prompt: str,
    temperature: float = 0.3,
    selected_chunk_ids: Optional[List[int]] = None
) -> Tuple[str, str, str]:
    """
    Perform analysis on retrieved chunks with optional chunk selection.
    UPDATED: Uses new terminology and enhanced chunk selection.
    """
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", "", "Check the logs for details"
        
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            return "Bitte führen Sie zuerst eine Heuristik durch", "", "Keine Quellen abgerufen"
        
        start_time = time.time()
        logger.info(f"Starting analysis with user-prompt: '{user_prompt}'")
        logger.info(f"Using system prompt: {system_prompt[:100]}...")
        
        # Convert UI chunks back to Document format
        from langchain.docstore.document import Document
        
        chunks_data = retrieved_chunks.get('chunks', [])
        
        # Filter chunks if specific IDs are selected
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
            question=user_prompt,
            chunks=documents,
            model=model_selection,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # Format results with updated terminology
        answer_text = analysis_result.answer
        chunks_text = format_chunks(chunks_data, selected_chunk_ids=selected_chunk_ids)
        
        # Format metadata with chunk selection info and updated terminology
        chunk_selection_info = ""
        if selected_chunk_ids is not None:
            chunk_selection_info = f"""
        ## Quellenauswahl
        - **Verwendete Chunks**: {len(chunks_data)} von {retrieved_chunks.get('metadata', {}).get('total_chunks', len(retrieved_chunks.get('chunks', [])))}
        - **Ausgewählte IDs**: {', '.join(map(str, selected_chunk_ids))}
        """
        
        metadata_text = f"""
        ## Analyse-Parameter
        - **Modell**: {analysis_result.model}
        - **User-Prompt**: {user_prompt}
        - **Analysezeit**: {analysis_time:.2f} Sekunden
        - **Temperatur**: {temperature} (Determinismus-Grad)
        {chunk_selection_info}

        ## System-Prompt (verwendet)
        ```
        {system_prompt[:500]}{'...' if len(system_prompt) > 500 else ''}
        ```

        ## Quellen-Metadaten
        - **Anzahl Quellen**: {len(documents)}
        - **Retrieval-Methode**: {retrieved_chunks.get('metadata', {}).get('retrieval_method', 'Unbekannt')}
        - **Original-Suchzeit**: {retrieved_chunks.get('metadata', {}).get('search_time', 0):.2f} Sekunden
        - **Chunking-Größe**: {retrieved_chunks.get('metadata', {}).get('chunk_size', 'Unbekannt')}
        """
        
        return answer_text, chunks_text, metadata_text
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        return f"Error: {str(e)}", "", "Analysis failed, check logs for details."

def perform_retrieval_and_update_ui(
    retrieval_query: str,  
    chunk_size: int,
    year_start: int,
    year_end: int,
    keywords: str,
    search_in: List[str],
    use_semantic_expansion: bool,
    semantic_expansion_factor: int,
    expanded_words_json: str,
    use_time_intervals: bool, 
    time_interval_size: int,  
    top_k: int,
    chunks_per_interval: int = 5 
) -> Tuple[str, Dict[str, Any], gr.update, gr.update, gr.update]:
    """
    Perform retrieval and update UI accordions with updated terminology.
    FIXED: Returns proper gr.update objects instead of raw accordions.
    """
    # Perform the retrieval
    info_text, retrieved_chunks = perform_retrieval(
        retrieval_query, chunk_size, year_start, year_end,
        keywords, search_in, use_semantic_expansion,
        semantic_expansion_factor, expanded_words_json,
        use_time_intervals, time_interval_size, 
        top_k, chunks_per_interval
    )
    
    # Format the retrieved chunks for display (if needed for backward compatibility)
    if retrieved_chunks and retrieved_chunks.get('chunks'):
        num_chunks = len(retrieved_chunks.get('chunks'))
        
        # Add chunk count info to metadata for chunk selection
        retrieved_chunks['metadata']['total_chunks'] = num_chunks
    else:
        num_chunks = 0
    
    # Update UI accordions based on retrieval success
    if num_chunks > 0:
        retrieval_state = gr.update(open=False)
        retrieved_texts_state = gr.update(open=True)
        analysis_state = gr.update(open=True)
    else:
        retrieval_state = gr.update(open=True)
        retrieved_texts_state = gr.update(open=False)
        analysis_state = gr.update(open=False)
    
    return info_text, retrieved_chunks, retrieval_state, retrieved_texts_state, analysis_state

def update_chunks_display_handler(retrieved_chunks: Dict[str, Any]) -> tuple:
    """
    Handle updating the chunks display after retrieval.
    
    Args:
        retrieved_chunks: Retrieved chunks data
        
    Returns:
        Tuple for updating chunks display components
    """
    # Import here to avoid circular imports
    from src.ui.components.retrieved_chunks_display import update_chunks_display
    
    return update_chunks_display(retrieved_chunks)

def perform_analysis_and_update_ui_with_transferred_chunks(
    user_prompt: str,
    transferred_chunks: list,
    model_selection: str,
    system_prompt_template: str,
    system_prompt_text: str,
    temperature: float
) -> Tuple[str, str, gr.update, gr.update]:
    """
    Perform analysis using transferred chunks.
    FIXED: Returns proper gr.update objects.
    
    Args:
        user_prompt: User's research question
        transferred_chunks: Chunks transferred from heuristic phase
        model_selection: Selected LLM model
        system_prompt_template: Template name
        system_prompt_text: Actual system prompt text
        temperature: Generation temperature
        
    Returns:
        Tuple of analysis results for UI update
    """
    if not transferred_chunks:
        return (
            "❌ Keine Quellen für die Analyse verfügbar. Bitte übertragen Sie zuerst Quellen aus der Heuristik.",
            "**Fehler**: Keine übertragenen Quellen",
            gr.update(open=True),   # Keep analysis accordion open
            gr.update(open=False)   # Keep results closed
        )
    
    # Convert transferred chunks to the format expected by existing analysis function
    retrieved_chunks_format = {
        'chunks': transferred_chunks,
        'metadata': {
            'total_chunks': len(transferred_chunks),
            'source': 'transferred_from_heuristic'
        }
    }
    
    # Use existing analysis function with adapted format
    answer_text, metadata_text, analysis_config_state, results_state = perform_analysis_and_update_ui(
        user_prompt=user_prompt,
        retrieved_chunks=retrieved_chunks_format,
        model_selection=model_selection,
        system_prompt_template=system_prompt_template,
        system_prompt_text=system_prompt_text,
        temperature=temperature,
        chunk_selection_mode="all",  # Use all transferred chunks
        selected_chunks_state=None   # Not needed since we use all transferred
    )
    
    return answer_text, metadata_text, analysis_config_state, results_state

def perform_analysis_and_update_ui(
    user_prompt: str,  # UPDATED: from question
    retrieved_chunks: Dict[str, Any],
    model_selection: str,
    system_prompt_template: str,
    system_prompt_text: str,
    temperature: float,
    chunk_selection_mode: str = "all",
    selected_chunks_state: Optional[List[int]] = None
) -> Tuple[str, str, gr.Accordion, gr.Accordion]:
    """
    Perform analysis and update UI accordions with updated terminology.
    """
    # Use the edited system prompt text directly
    system_prompt = system_prompt_text.strip()
    
    # Fallback to default if somehow empty
    if not system_prompt:
        logger.warning("System prompt text is empty, falling back to default")
        system_prompt = settings.SYSTEM_PROMPTS["default"]
    
    logger.info(f"Using system prompt text directly (length: {len(system_prompt)} chars)")
    
    # Determine which chunks to use based on selection mode
    selected_chunk_ids = None
    if chunk_selection_mode == "selected" and selected_chunks_state:
        selected_chunk_ids = selected_chunks_state
        logger.info(f"Using chunk selection with {len(selected_chunk_ids)} selected chunks")
    
    # Perform the analysis
    answer_text, chunks_text, metadata_text = perform_analysis(
        user_prompt=user_prompt,
        retrieved_chunks=retrieved_chunks,
        model_selection=model_selection,
        system_prompt=system_prompt,
        temperature=temperature,
        selected_chunk_ids=selected_chunk_ids
    )
    
    # Update UI accordions
    analysis_config_state = gr.update(open=False)
    results_state = gr.update(open=True)
    
    return answer_text, metadata_text, analysis_config_state, results_state

def format_chunks(
    chunks: List[Dict],
    keywords_to_use: Optional[str] = None, 
    expanded_words: Optional[Dict] = None,
    use_time_intervals: bool = False,  # UPDATED: from use_time_windows
    time_interval_size: int = 5,       # UPDATED: from time_window_size
    year_start: int = 1948, 
    year_end: int = 1979,
    chunks_per_interval: Optional[int] = None,  # UPDATED: from chunks_per_window
    selected_chunk_ids: Optional[List[int]] = None
) -> str:
    """
    Format retrieved chunks for display with updated terminology.
    """
    if not chunks:
        return "Keine passenden Texte gefunden."
    
    # Add selection info if applicable
    selection_info = ""
    if selected_chunk_ids is not None:
        selection_info = f"**Hinweis**: Nur {len(chunks)} von ursprünglich gefundenen Texten werden angezeigt (ausgewählte IDs: {', '.join(map(str, selected_chunk_ids))})\n\n"
    
    # Group chunks by year for better readability
    chunks_by_year = {}
    for i, chunk in enumerate(chunks):
        year = chunk["metadata"].get("Jahrgang", "Unknown")
        if year not in chunks_by_year:
            chunks_by_year[year] = []
        # Add original index for chunk ID tracking
        chunk_with_id = chunk.copy()
        chunk_with_id['display_id'] = i + 1
        chunks_by_year[year].append(chunk_with_id)
    
    chunks_text = selection_info
    
    # Display chunks grouped by time interval if using Zeitintervall-Suche
    if use_time_intervals:
        chunks_text += "# Ergebnisse nach Zeit-Intervallen\n\n"
        
        # Show chunks per interval info
        if chunks_per_interval:
            chunks_text += f"**Konfiguration**: {chunks_per_interval} Chunks pro Zeitintervall\n\n"
        
        # Create time intervals
        time_intervals = []
        for interval_start in range(year_start, year_end + 1, time_interval_size):
            interval_end = min(interval_start + time_interval_size - 1, year_end)
            time_intervals.append((interval_start, interval_end))
        
        # Group years into their respective time intervals
        for interval_start, interval_end in time_intervals:
            interval_label = f"## Zeitintervall {interval_start}-{interval_end}\n\n"
            interval_chunks = []
            
            # Collect chunks from years in this interval
            for year in sorted(chunks_by_year.keys()):
                if isinstance(year, int) and interval_start <= year <= interval_end:
                    interval_chunks.extend([(year, i, chunk) for i, chunk in enumerate(chunks_by_year[year])])
            
            # Only add interval if it has chunks
            if interval_chunks:
                chunks_text += interval_label
                
                # Count chunks per year in this interval
                interval_year_counts = {}
                for y, _, _ in interval_chunks:
                    interval_year_counts[y] = interval_year_counts.get(y, 0) + 1
                
                # Show year distribution within interval
                chunks_text += "**Verteilung:** "
                chunks_text += ", ".join([f"{y}: {count} Texte" for y, count in sorted(interval_year_counts.items())])
                chunks_text += "\n\n"
                
                # Add each chunk
                current_year = None
                chunk_in_year = 1
                
                for year, _, chunk in sorted(interval_chunks):
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
                chunks_text += interval_label
                chunks_text += "Keine Texte gefunden in diesem Zeitintervall.\n\n"
    else:
        # Regular display by year when not using Zeitintervall-Suche
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


# UPDATED: Custom time interval strategy with new terminology
class EnhancedTimeIntervalSearchStrategy:
    """Enhanced Zeitintervall search strategy with chunks per interval control."""
    
    def __init__(self, interval_size: int = 5, chunks_per_interval: int = 5):
        self.interval_size = interval_size
        self.chunks_per_interval = chunks_per_interval
    
    def search(self, config, vector_store, progress_callback=None):
        """Execute Zeitintervall search with controlled chunks per interval."""
        from src.core.search.strategies import SearchResult
        
        start_time = time.time()
        start_year, end_year = config.year_range
        
        # Create time intervals
        intervals = []
        for interval_start in range(start_year, end_year + 1, self.interval_size):
            interval_end = min(interval_start + self.interval_size - 1, end_year)
            intervals.append((interval_start, interval_end))
        
        logger.info(f"Enhanced Zeitintervall search: {len(intervals)} intervals, {self.chunks_per_interval} chunks per interval")
        
        all_chunks = []
        interval_counts = {}
        
        # Search each interval
        for i, (interval_start, interval_end) in enumerate(intervals):
            if progress_callback:
                progress = (i / len(intervals))
                progress_callback(f"Searching {interval_start}-{interval_end}...", progress)
            
            # Create interval-specific filter
            interval_filter = vector_store.build_metadata_filter(
                year_range=[interval_start, interval_end],
                keywords=None,
                search_in=None
            )
            
            try:
                # Use chunks_per_interval instead of total k
                interval_chunks = vector_store.similarity_search(
                    query=config.content_description,
                    chunk_size=config.chunk_size,
                    k=self.chunks_per_interval,  # Use per-interval limit
                    filter_dict=interval_filter,
                    min_relevance_score=0.3,
                    keywords=config.keywords,
                    search_in=config.search_fields,
                    enforce_keywords=config.enforce_keywords
                )
                
                interval_key = f"{interval_start}-{interval_end}"
                interval_counts[interval_key] = len(interval_chunks)
                
                # Add interval metadata to each chunk
                for doc, score in interval_chunks:
                    doc.metadata['time_interval'] = interval_key  # UPDATED: from time_window
                    doc.metadata['interval_start'] = interval_start
                    doc.metadata['interval_end'] = interval_end
                
                all_chunks.extend(interval_chunks)
                
            except Exception as e:
                logger.error(f"Error searching interval {interval_start}-{interval_end}: {e}")
                interval_counts[f"{interval_start}-{interval_end}"] = 0
        
        # Sort by relevance score
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            chunks=all_chunks,
            metadata={
                "strategy": "enhanced_time_interval",  # UPDATED
                "search_time": search_time,
                "interval_size": self.interval_size,
                "chunks_per_interval": self.chunks_per_interval,
                "intervals": intervals,  # UPDATED: from windows
                "interval_counts": interval_counts,  # UPDATED: from window_counts
                "total_chunks_found": len(all_chunks)
            }
        )