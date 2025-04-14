# src/ui/handlers/search_handlers.py
"""
Handler functions for search operations.
These functions are connected to UI events in the search panel.
Refactored to support separate retrieval and analysis steps.
"""
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any
import gradio as gr


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import settings

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
    # parameters for agent-based retrieval
    use_agent_retrieval: bool = False,
    initial_chunk_count: int = 100,
    enable_interactive: bool = False,
    relevance_weight: float = 0.4,
    diversity_weight: float = 0.3,
    quality_weight: float = 0.3,
    model_selection: str = "hu-llm",
    openai_api_key: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform source retrieval based on content description and filters.
    Now supports both standard and agent-based cascading retrieval.
    """
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", None
        
        start_time = time.time()
        logger.info(f"Starting retrieval: content_description='{content_description}', keywords='{keywords}', agent_based={use_agent_retrieval}")
        
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
        
        # Determine model to use for agent
        model_to_use = "hu-llm"  # Default
        if model_selection == "openai-gpt4o":
            model_to_use = "gpt-4o"
        elif model_selection == "openai-gpt35":
            model_to_use = "gpt-3.5-turbo"
        
        # Setup progress tracking (would be integrated with UI in real implementation)
        progress_updates = []
        def progress_callback(message, progress):
            progress_updates.append({"message": message, "progress": progress})
            logger.info(f"Progress: {message} ({progress:.2f})")
        
        # Choose retrieval method based on user selection
        if use_agent_retrieval:
            # Configure filtering stages based on weights
            filtering_stages = [
                {
                    "name": "Basic Relevance",
                    "keep_ratio": 0.5,
                    "criteria": ["relevance"]
                },
                {
                    "name": "Contextual Quality",
                    "keep_ratio": 0.4,
                    "criteria": ["context", "quality"]
                },
                {
                    "name": "Diversity Selection",
                    "keep_ratio": 0.5,
                    "criteria": ["diversity", "representativeness"]
                }
            ]
            
            # Perform cascading agent-based retrieval
            results = rag_engine.cascading_retrieve(
                content_description=content_description,
                question=None,  # Initially no question available during retrieval
                year_range=[year_start, year_end],
                chunk_size=chunk_size,
                keywords=keywords_to_use,
                search_in=search_fields,
                initial_chunk_count=initial_chunk_count,
                filtering_stages=filtering_stages,
                model=model_to_use,
                openai_api_key=openai_api_key,
                interactive=enable_interactive,
                progress_callback=progress_callback
            )
        else:
            # Standard retrieval
            results = rag_engine.retrieve(
                content_description=content_description,
                year_range=[year_start, year_end],
                chunk_size=chunk_size,
                keywords=keywords_to_use,
                search_in=search_fields,
                use_iterative_search=use_time_windows,
                time_window_size=time_window_size,
                use_semantic_expansion=use_semantic_expansion and keywords_to_use is not None,
                semantic_expansion_factor=semantic_expansion_factor,
                enforce_keywords=enforce_keywords,
                top_k=top_k
            )
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieval completed in {retrieval_time:.2f} seconds")
        
        # Get actual number of chunks
        num_chunks = len(results.get('chunks', []))
        logger.info(f"Found {num_chunks} chunks")
        
        # Add progress information to results metadata
        if use_agent_retrieval:
            results["metadata"]["progress"] = progress_updates
            results["metadata"]["retrieval_method"] = "agent"
        else:
            results["metadata"]["retrieval_method"] = "standard"
        
        # Create summary info message
        if num_chunks > 0:
            info_text = f"""
            ### Quellen erfolgreich abgerufen
            
            **Inhaltsbeschreibung**: {content_description}  
            **Zeitraum**: {year_start} - {year_end}  
            **Anzahl gefundener Quellen**: {num_chunks}  
            **Abrufzeit**: {retrieval_time:.2f} Sekunden
            **Methode**: {"Agent-basiertes Retrieval" if use_agent_retrieval else "Standard-Retrieval"}
            
            Sie können jetzt im Tab "2. Quellen analysieren" Fragen zu diesen Quellen stellen.
            """
            
            # Add agent-specific info
            if use_agent_retrieval:
                info_text += f"""
                
                **Agent-Prozess**:
                - Initiale Dokumentanzahl: {initial_chunk_count}
                - Finale Dokumentanzahl: {num_chunks}
                - Filterungsphasen: {len(filtering_stages)}
                """
                
                # Add progress information
                if progress_updates:
                    info_text += "\n**Fortschritt**:\n"
                    for update in progress_updates:
                        info_text += f"- {update['message']}\n"
        else:
            info_text = f"""
            ### Keine passenden Quellen gefunden
            
            Versuchen Sie es mit einer anderen Inhaltsbeschreibung oder erweitern Sie die Filter.
            """
        
        return info_text, results
        
    except Exception as e:
        logger.error(f"Error in retrieval: {e}", exc_info=True)
        return f"Error: {str(e)}", None
    

def perform_analysis(
    question: str,
    retrieved_chunks: Dict[str, Any],
    model_selection: str,
    openai_api_key: str,
    system_prompt: Optional[str] = None,  # Add new parameters
    temperature: float = 0.3,
    max_tokens: Optional[int] = None
) -> Tuple[str, str, str]:
    """
    Perform analysis on previously retrieved chunks.
    
    Args:
        question: Question to answer based on retrieved chunks
        retrieved_chunks: Dict with previously retrieved chunks
        model_selection: Selected LLM model
        openai_api_key: OpenAI API key (if applicable)
        system_prompt: Custom system prompt (if provided)
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (answer text, chunks text, metadata text)
    """
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", "", "Check the logs for details"
        
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            return "Bitte rufen Sie zuerst Quellen ab", "", "Keine Quellen abgerufen"
        
        start_time = time.time()
        logger.info(f"Starting analysis with question: '{question}'")
        
        # Handle model selection
        model_to_use = "hu-llm"  # Default
        if model_selection == "openai-gpt4o":
            model_to_use = "gpt-4o"
        elif model_selection == "openai-gpt35":
            model_to_use = "gpt-3.5-turbo"
        
        # Perform analysis with new parameters
        results = rag_engine.analyze(
            question=question,
            model=model_to_use,
            openai_api_key=openai_api_key,
            with_citations=False,
            system_prompt=system_prompt,    # Pass new parameters
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # Get chunks from the retrieval results
        chunks = retrieved_chunks.get('chunks', [])
        
        # Format results
        answer_text = results.get('answer', 'No answer generated')
        chunks_text = format_chunks(chunks)
        
        # Format metadata
        from src.ui.utils.ui_helpers import format_analysis_metadata
        metadata_text = format_analysis_metadata(
            question=question,
            model=model_to_use,
            analysis_time=analysis_time,
            retrieved_info=retrieved_chunks.get('metadata', {}),
            temperature=temperature,     # Include new parameters in metadata
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        
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
    top_k: int
) -> Tuple[str, Dict[str, Any], str, gr.Accordion, gr.Accordion, gr.Accordion]:
    """
    Perform retrieval and update UI accordions.
    
    Returns:
        Tuple containing:
        - Retrieved info text
        - Retrieved chunks state
        - Formatted chunks for display
        - Retrieval accordion state (collapsed)
        - Retrieved texts accordion state (expanded)
        - Question accordion state (expanded)
    """
    # First, perform the retrieval
    info_text, retrieved_chunks = perform_retrieval(
        content_description, chunk_size, year_start, year_end,
        keywords, search_in, use_semantic_expansion,
        semantic_expansion_factor, expanded_words_json,
        enforce_keywords, use_time_windows, time_window_size, top_k
    )
    
    # Format the retrieved chunks for display
    if retrieved_chunks and retrieved_chunks.get('chunks'):
        formatted_chunks = format_chunks(retrieved_chunks.get('chunks'))
        num_chunks = len(retrieved_chunks.get('chunks'))
    else:
        formatted_chunks = "Keine Texte gefunden."
        num_chunks = 0
    
    # Update UI accordions based on retrieval success
    if num_chunks > 0:
        # Collapse retrieval section
        retrieval_state = gr.update(open=False)
        # Expand retrieved texts section
        retrieved_texts_state = gr.update(open=True)
        # Expand question section
        question_state = gr.update(open=True)
    else:
        # Keep retrieval section expanded if no results
        retrieval_state = gr.update(open=True)
        # Collapse other sections
        retrieved_texts_state = gr.update(open=False)
        question_state = gr.update(open=False)
    
    return info_text, retrieved_chunks, formatted_chunks, retrieval_state, retrieved_texts_state, question_state

def perform_analysis_and_update_ui(
    question: str,
    retrieved_chunks: Dict[str, Any],
    model_selection: str,
    openai_api_key: str,
    system_prompt_template: str,
    custom_system_prompt: str,
    temperature: float,
    max_tokens: int
) -> Tuple[str, str, gr.Accordion, gr.Accordion]:
    """
    Perform analysis and update UI accordions.
    
    Returns:
        Tuple containing:
        - Answer text
        - Metadata text
        - Question accordion state (collapsed)
        - Results accordion state (expanded)
    """
    # Determine which system prompt to use
    if custom_system_prompt.strip():
        system_prompt = custom_system_prompt
    else:
        system_prompt = settings.SYSTEM_PROMPTS.get(system_prompt_template, settings.SYSTEM_PROMPTS["default"])
    
    # Perform the analysis - note the proper unpacking of return values
    answer_text, chunks_text, metadata_text = perform_analysis(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model_selection=model_selection,
        openai_api_key=openai_api_key,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Update UI accordions
    # Collapse question section
    question_state = gr.update(open=False)
    # Expand results section
    results_state = gr.update(open=True)
    
    return answer_text, metadata_text, question_state, results_state

def format_chunks(
    chunks: List[Dict],
    keywords_to_use: Optional[str] = None, 
    expanded_words: Optional[Dict] = None,
    use_time_windows: bool = False, 
    time_window_size: int = 5, 
    year_start: int = 1948, 
    year_end: int = 1979
) -> str:
    """
    Format retrieved chunks for display.
    
    Args:
        chunks: List of chunk dictionaries
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
    if not chunks:
        return "Keine passenden Texte gefunden."
    
    # Group chunks by year for better readability
    chunks_by_year = {}
    for chunk in chunks:
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
                    chunks_text += f"**Relevanz**: {chunk['relevance_score']:.3f}"
                    url = metadata.get('URL')
                    if url and url != 'Keine URL':
                        chunks_text += f" | [**Link zum Artikel**]({url})"

                    chunks_text += "\n\n"
                      
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
                chunks_text += f"**Relevanz**: {chunk['relevance_score']:.3f}"
                url = metadata.get('URL')
                if url and url != 'Keine URL':
                    chunks_text += f" | [**Link zum Artikel**]({url})"
                chunks_text += "\n\n"
                
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