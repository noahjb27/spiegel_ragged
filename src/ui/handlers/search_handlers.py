# src/ui/handlers/search_handlers.py
"""
Handler functions for search operations.
These functions are connected to UI events in the search panel.
"""
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Global reference to the RAG engine
# This will be initialized in the main app
rag_engine = None

def set_rag_engine(engine: Any) -> None:
    """
    Set the global RAG engine reference.
    
    Args:
        engine: The SpiegelRAGEngine instance
    """
    global rag_engine
    rag_engine = engine

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
    model_selection: str,
    openai_api_key: str
) -> Tuple[str, str, str]:
    """
    Perform search with keyword filtering and semantic expansion.
    
    Args:
        query: Content description to search for
        question: Question to answer based on search results
        chunk_size: Size of text chunks to retrieve
        year_start: Start year for search range
        year_end: End year for search range
        keywords: Boolean expression of keywords to filter by
        search_in: List of fields to search in
        use_semantic_expansion: Whether to use semantic expansion
        semantic_expansion_factor: Number of similar words to find
        expanded_words_json: JSON string of expanded words
        enforce_keywords: Whether to strictly enforce keyword filtering
        use_time_windows: Whether to use time window search
        time_window_size: Size of time windows in years
        model_selection: Selected LLM model
        openai_api_key: OpenAI API key (if applicable)
        
    Returns:
        Tuple of (answer text, chunks text, metadata text)
    """
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
            model=model_to_use,
            openai_api_key=openai_api_key,
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
        chunks_text = format_chunks(
            results, 
            keywords_to_use, 
            expanded_words, 
            use_time_windows, 
            time_window_size, 
            year_start, 
            year_end
        )
        
        # Create time windows for metadata
        time_windows = []
        window_counts = {}
        if use_time_windows:
            for window_start in range(year_start, year_end + 1, time_window_size):
                window_end = min(window_start + time_window_size - 1, year_end)
                time_windows.append((window_start, window_end))
            
            # Count chunks per window
            for chunk in results.get('chunks', []):
                year = chunk["metadata"].get("Jahrgang")
                if isinstance(year, int):
                    for i, (window_start, window_end) in enumerate(time_windows):
                        if window_start <= year <= window_end:
                            window_key = f"{window_start}-{window_end}"
                            window_counts[window_key] = window_counts.get(window_key, 0) + 1
                            break
        
        # Format metadata using the helper function from ui_helpers.py
        from src.ui.utils.ui_helpers import format_search_metadata
        metadata_text = format_search_metadata(
            model=model_to_use,
            query=query,
            question=question,
            chunk_size=chunk_size,
            year_start=year_start,
            year_end=year_end,
            search_time=search_time,
            num_chunks=num_chunks,
            keywords=keywords_to_use,
            search_fields=search_fields,
            enforce_keywords=enforce_keywords,
            use_semantic_expansion=use_semantic_expansion and keywords_to_use is not None,
            use_time_windows=use_time_windows,
            time_window_size=time_window_size,
            expanded_words=expanded_words,
            time_windows=time_windows,
            window_counts=window_counts
        )
        
        return answer_text, chunks_text, metadata_text
    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return f"Error: {str(e)}", "", "Search failed, check logs for details."

def format_chunks(
    results: Dict, 
    keywords_to_use: Optional[str], 
    expanded_words: Optional[Dict],
    use_time_windows: bool, 
    time_window_size: int, 
    year_start: int, 
    year_end: int
) -> str:
    """
    Format retrieved chunks for display.
    
    Args:
        results: Search results dictionary
        keywords_to_use: Keywords used in search
        expanded_words: Dictionary of expanded words
        use_time_windows: Whether time windows were used
        time_window_size: Size of time windows
        year_start: Start year of search range
        year_end: End year of search range
        
    Returns:
        Formatted markdown string with chunks information
    """
    chunks_text = ""
    if not results.get("chunks"):
        return "Keine passenden Texte gefunden."
    
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
    
    return chunks_text