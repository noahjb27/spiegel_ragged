# src/ui/handlers/search_handlers.py - Fixed version
"""
Handler functions for search operations - Fixed for new strategy-based architecture
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
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform source retrieval using the new strategy-based approach.
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
        
        # Create search configuration
        config = SearchConfig(
            content_description=content_description,
            year_range=(year_start, year_end),
            chunk_size=chunk_size,
            keywords=keywords_cleaned,
            search_fields=search_fields,
            enforce_keywords=enforce_keywords,
            top_k=top_k
        )
        
        # Choose strategy based on user selection
        if use_time_windows:
            logger.info(f"Using TimeWindowSearchStrategy with window size: {time_window_size}")
            strategy = TimeWindowSearchStrategy(window_size=time_window_size)
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
                'retrieval_method': 'time_window' if use_time_windows else 'standard'
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
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None
) -> Tuple[str, str, str]:
    """
    Perform analysis on previously retrieved chunks using the new engine.
    """
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", "", "Check the logs for details"
        
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            return "Bitte rufen Sie zuerst Quellen ab", "", "Keine Quellen abgerufen"
        
        start_time = time.time()
        logger.info(f"Starting analysis with question: '{question}'")
        
        # Convert UI chunks back to Document format
        from langchain.docstore.document import Document
        
        chunks_data = retrieved_chunks.get('chunks', [])
        documents = []
        for chunk_data in chunks_data:
            doc = Document(
                page_content=chunk_data['content'],
                metadata=chunk_data['metadata']
            )
            documents.append(doc)
        
        logger.info(f"Converted {len(documents)} chunks for analysis")
        
        # Handle model selection - FIXED: Use model name directly
        model_to_use = model_selection
        
        # Perform analysis with new engine - FIXED: Removed openai_api_key parameter
        analysis_result = rag_engine.analyze(
            question=question,
            chunks=documents,
            model=model_to_use,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # Format results
        answer_text = analysis_result.answer
        chunks_text = format_chunks(chunks_data)
        
        # Format metadata
        metadata_text = f"""
        ## Analyseparameter
        - **Model**: {analysis_result.model}
        - **Frage**: {question}
        - **Analysezeit**: {analysis_time:.2f} Sekunden
        - **Temperatur**: {temperature}
        - **Max Tokens**: {max_tokens or "Standardwert"}

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
    top_k: int
) -> Tuple[str, Dict[str, Any], str, gr.Accordion, gr.Accordion, gr.Accordion]:
    """
    Perform retrieval and update UI accordions.
    """
    # Perform the retrieval
    info_text, retrieved_chunks = perform_retrieval(
        content_description, chunk_size, year_start, year_end,
        keywords, search_in, use_semantic_expansion,
        semantic_expansion_factor, expanded_words_json,
        enforce_keywords, use_time_windows, time_window_size, top_k
    )
    
    # Format the retrieved chunks for display
    if retrieved_chunks and retrieved_chunks.get('chunks'):
        formatted_chunks = format_chunks(
            retrieved_chunks.get('chunks'),
            keywords_to_use=keywords,
            use_time_windows=use_time_windows,
            time_window_size=time_window_size,
            year_start=year_start,
            year_end=year_end
        )
        num_chunks = len(retrieved_chunks.get('chunks'))
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
    custom_system_prompt: str,
    temperature: float,
    max_tokens: int
) -> Tuple[str, str, gr.Accordion, gr.Accordion]:
    """
    Perform analysis and update UI accordions.
    """
    # Determine which system prompt to use
    if custom_system_prompt.strip():
        system_prompt = custom_system_prompt
    else:
        system_prompt = settings.SYSTEM_PROMPTS.get(system_prompt_template, settings.SYSTEM_PROMPTS["default"])
    
    # Perform the analysis - FIXED: Removed openai_api_key parameter
    answer_text, chunks_text, metadata_text = perform_analysis(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model_selection=model_selection,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
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
    year_end: int = 1979
) -> str:
    """
    Format retrieved chunks for display with enhanced time window support.
    """
    if not chunks:
        return "Keine passenden Texte gefunden."
    
    # Group chunks by year for better readability
    chunks_by_year = {}
    for chunk in chunks:
        year = chunk["metadata"].get("Jahrgang", "Unknown")
        if year not in chunks_by_year:
            chunks_by_year[year] = []
        chunks_by_year[year].append(chunk)
    
    chunks_text = ""
    
    # Display chunks grouped by time window if using time windows
    if use_time_windows:
        chunks_text += "# Ergebnisse nach Zeitfenstern\n\n"
        
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
                    chunks_text += f"#### {chunk_in_year}. {metadata.get('Artikeltitel', 'Kein Titel')}\n\n"
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
                chunks_text += f"### {i}. {metadata.get('Artikeltitel', 'Kein Titel')}\n\n"
                chunks_text += f"**Datum**: {metadata.get('Datum', 'Unbekannt')} | "
                chunks_text += f"**Relevanz**: {chunk['relevance_score']:.3f}"
                
                url = metadata.get('URL')
                if url and url != 'Keine URL':
                    chunks_text += f" | [**Link zum Artikel**]({url})"
                
                chunks_text += "\n\n"
                chunks_text += f"**Text**:\n{chunk['content']}\n\n"
                chunks_text += "---\n\n"
    
    return chunks_text