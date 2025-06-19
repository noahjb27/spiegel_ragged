# src/ui/handlers/agent_handlers.py - Updated to handle both scores
"""
Updated handlers for agent search functionality with dual score support.
"""
import json
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import gradio as gr

from src.core.search.agent_strategy import TimeWindowedAgentStrategy, AgentSearchConfig
from src.core.search.strategies import SearchConfig
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global references
rag_engine = None
current_agent_search = None
search_thread = None

def set_rag_engine(engine: Any) -> None:
    """Set the global RAG engine reference."""
    global rag_engine
    rag_engine = engine

def perform_agent_search(
    content_description: str,
    chunk_size: int,
    year_start: int,
    year_end: int,
    agent_use_time_windows: bool,
    agent_time_window_size: int,
    chunks_per_window_initial: int,
    chunks_per_window_final: int,
    agent_min_retrieval_score: float,  # NEW parameter
    agent_keywords: str,
    agent_search_in: List[str],
    agent_enforce_keywords: bool,
    agent_model: str,
    agent_system_prompt_template: str,
    agent_custom_system_prompt: str,
    progress_callback: Optional[Any] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform agent search with time windowing and dual score preservation.
    
    Returns:
        Tuple of (status_message, search_results)
    """
    global current_agent_search
    
    try:
        if not rag_engine:
            return "Error: RAG Engine nicht initialisiert", None
        
        if not content_description.strip():
            return "Error: Inhaltsbeschreibung ist erforderlich", None
        
        start_time = time.time()
        logger.info(f"Starting agent search: '{content_description}'")
        
        # Clean parameters
        content_description = content_description.strip()
        keywords_cleaned = agent_keywords.strip() if agent_keywords else None
        search_fields = agent_search_in if agent_search_in else ["Text"]
        
        # Determine system prompt
        if agent_custom_system_prompt.strip():
            system_prompt = agent_custom_system_prompt.strip()
        else:
            system_prompt = settings.ALL_SYSTEM_PROMPTS.get(
                agent_system_prompt_template, 
                settings.AGENT_SYSTEM_PROMPTS["agent_default"]
            )
        
        # Create search configurations
        search_config = SearchConfig(
            content_description=content_description,
            year_range=(year_start, year_end),
            chunk_size=chunk_size,
            keywords=keywords_cleaned,
            search_fields=search_fields,
            enforce_keywords=agent_enforce_keywords,
            top_k=100  # Not used in agent search, but required
        )
        
        agent_config = AgentSearchConfig(
            use_time_windows=agent_use_time_windows,
            time_window_size=agent_time_window_size,
            chunks_per_window_initial=chunks_per_window_initial,
            chunks_per_window_final=chunks_per_window_final,
            agent_model=agent_model,
            agent_system_prompt=system_prompt,
            min_retrieval_relevance_score=agent_min_retrieval_score  # NEW: Pass minimum score
        )
        
        # Create and execute agent strategy
        agent_strategy = TimeWindowedAgentStrategy(
            llm_service=rag_engine.llm_service,
            agent_config=agent_config
        )
        
        # Store reference for potential cancellation
        current_agent_search = agent_strategy
        
        logger.info("Executing agent search with time windowing...")
        
        # Progress callback wrapper
        def progress_wrapper(message: str, progress: float):
            if progress_callback:
                progress_callback(message, progress)
        
        # Execute search
        search_result = agent_strategy.search(
            config=search_config,
            vector_store=rag_engine.vector_store,
            progress_callback=progress_wrapper
        )
        
        search_time = time.time() - start_time
        
        # Clear reference
        current_agent_search = None
        
        # Check for errors
        if "error" in search_result.metadata:
            error_msg = search_result.metadata["error"]
            logger.error(f"Agent search error: {error_msg}")
            return f"Error: {error_msg}", None
        
        # Convert search result to UI format WITH BOTH SCORES
        chunks_for_ui = []
        for doc, llm_score in search_result.chunks:
            # Extract both scores from metadata
            vector_score = doc.metadata.get('vector_similarity_score', 0.0)
            llm_eval_score = doc.metadata.get('llm_evaluation_score', llm_score)
            
            chunks_for_ui.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': llm_score,  # Primary score for UI display (LLM score)
                'vector_similarity_score': vector_score,  # NEW: Vector similarity score
                'llm_evaluation_score': llm_eval_score  # NEW: Explicit LLM score
            })
        
        # Build comprehensive results dictionary
        results = {
            'chunks': chunks_for_ui,
            'metadata': {
                'search_time': search_time,
                'strategy': 'time_windowed_agent',
                'chunk_size': chunk_size,
                'year_range': [year_start, year_end],
                'keywords': keywords_cleaned,
                'retrieval_method': 'agent_time_windowed' if agent_use_time_windows else 'agent_global',
                'min_retrieval_relevance_score': agent_min_retrieval_score,  # NEW: Include in metadata
                **search_result.metadata  # Include all agent metadata
            }
        }
        
        num_chunks = len(chunks_for_ui)
        logger.info(f"Agent search completed: {num_chunks} chunks in {search_time:.2f}s")
        
        # Create success message
        method_name = "Agenten-Suche mit Zeitfenstern" if agent_use_time_windows else "Agenten-Suche (global)"
        
        info_text = f"""
        ### Quellen erfolgreich durch KI-Bewertung ausgewählt ({method_name})
        
        **Inhaltsbeschreibung**: {content_description}  
        **Zeitraum**: {year_start} - {year_end}  
        **Anzahl ausgewählter Quellen**: {num_chunks}  
        **Suchzeit**: {search_time:.2f} Sekunden  
        **Bewertungsmodell**: {agent_model}
        **Mindest-Retrieval-Score**: {agent_min_retrieval_score}
        """
        
        if agent_use_time_windows:
            windows = search_result.metadata.get('time_windows', [])
            total_initial = search_result.metadata.get('total_initial_chunks', 0)
            info_text += f"""
            **Zeitfenster**: {len(windows)} Fenster à {agent_time_window_size} Jahre
            **Initial abgerufen**: {total_initial} Texte insgesamt
            **Pro Fenster**: {chunks_per_window_initial} → {chunks_per_window_final} Texte
            """
        
        if keywords_cleaned:
            info_text += f"""
            **Schlagwörter**: {keywords_cleaned}
            **Strikte Filterung**: {'Ja' if agent_enforce_keywords else 'Nein'}
            """
        
        return info_text, results
        
    except Exception as e:
        logger.error(f"Error in agent search: {e}", exc_info=True)
        current_agent_search = None
        return f"Error: {str(e)}", None

def cancel_agent_search() -> str:
    """Cancel the currently running agent search."""
    global current_agent_search
    
    if current_agent_search:
        logger.info("Cancelling agent search...")
        current_agent_search.cancel_search()
        return "Agenten-Suche wird abgebrochen..."
    else:
        return "Keine aktive Suche zum Abbrechen."

def perform_agent_search_threaded(
    content_description: str,
    chunk_size: int,
    year_start: int,
    year_end: int,
    agent_use_time_windows: bool,
    agent_time_window_size: int,
    chunks_per_window_initial: int,
    chunks_per_window_final: int,
    agent_min_retrieval_score: float,  # NEW parameter
    agent_keywords: str,
    agent_search_in: List[str],
    agent_enforce_keywords: bool,
    agent_model: str,
    agent_system_prompt_template: str,
    agent_custom_system_prompt: str
) -> Tuple[str, Dict[str, Any], str, gr.update, gr.update, gr.update, gr.update, gr.update, gr.update]:
    """
    Perform agent search in a thread with UI updates.
    
    Returns:
        Tuple for UI updates: (search_status, retrieved_chunks_state, formatted_chunks, 
                              search_mode_update, search_btn_update, cancel_btn_update,
                              progress_update, retrieved_texts_accordion, question_accordion)
    """
    global search_thread
    
    try:
        # Initialize UI state
        progress_state = gr.update(value="Agenten-Suche gestartet...", visible=True)
        cancel_btn_state = gr.update(visible=True)
        search_btn_state = gr.update(interactive=False)
        
        # Progress tracking
        progress_messages = []
        
        def progress_callback(message: str, progress: float):
            progress_messages.append(f"{progress*100:.0f}% - {message}")
            # Update progress display
            return gr.update(value=f"**Fortschritt**: {message} ({progress*100:.0f}%)")
        
        # Run search with NEW parameter
        search_status, retrieved_chunks = perform_agent_search(
            content_description, chunk_size, year_start, year_end,
            agent_use_time_windows, agent_time_window_size,
            chunks_per_window_initial, chunks_per_window_final,
            agent_min_retrieval_score,  # NEW: Pass minimum retrieval score
            agent_keywords, agent_search_in, agent_enforce_keywords,
            agent_model, agent_system_prompt_template, agent_custom_system_prompt,
            progress_callback
        )
        
        # Format chunks for display
        if retrieved_chunks and retrieved_chunks.get('chunks'):
            formatted_chunks = format_agent_chunks_with_dual_scores(retrieved_chunks)  # Updated function
            num_chunks = len(retrieved_chunks.get('chunks'))
            
            # Update accordion states for successful search
            retrieved_texts_state = gr.update(open=True)
            question_state = gr.update(open=True)
        else:
            formatted_chunks = "Keine Texte durch KI-Bewertung ausgewählt."
            num_chunks = 0
            
            # Keep search accordion open for retry
            retrieved_texts_state = gr.update(open=False)
            question_state = gr.update(open=False)
        
        # Final UI updates
        final_progress = gr.update(visible=False)
        final_cancel_btn = gr.update(visible=False)
        final_search_btn = gr.update(interactive=True)
        
        # Clear thread reference
        search_thread = None
        
        return (
            search_status, retrieved_chunks, formatted_chunks,
            gr.update(),  # search_mode (no change)
            final_search_btn, final_cancel_btn, final_progress,
            retrieved_texts_state, question_state
        )
        
    except Exception as e:
        logger.error(f"Threaded agent search failed: {e}", exc_info=True)
        
        # Reset UI on error
        error_status = f"Error: {str(e)}"
        final_progress = gr.update(visible=False)
        final_cancel_btn = gr.update(visible=False)
        final_search_btn = gr.update(interactive=True)
        
        search_thread = None
        
        return (
            error_status, None, "Fehler bei der Agenten-Suche.",
            gr.update(), final_search_btn, final_cancel_btn, final_progress,
            gr.update(open=False), gr.update(open=False)
        )

def format_agent_chunks_with_dual_scores(retrieved_chunks: Dict[str, Any]) -> str:
    """
    Format agent search results for display, highlighting both vector and LLM scores.
    """
    chunks = retrieved_chunks.get('chunks', [])
    metadata = retrieved_chunks.get('metadata', {})
    
    if not chunks:
        return "Keine Texte durch KI-Bewertung ausgewählt."
    
    # Group by time window if available
    chunks_by_window = {}
    evaluations = metadata.get('evaluations', [])
    
    for chunk in chunks:
        window = chunk['metadata'].get('time_window', 'Global')
        if window not in chunks_by_window:
            chunks_by_window[window] = []
        chunks_by_window[window].append(chunk)
    
    # Format output
    use_time_windows = len(chunks_by_window) > 1 or 'Global' not in chunks_by_window
    
    formatted_text = f"# KI-bewertete Quellen ({len(chunks)} ausgewählt)\n\n"
    
    # Add summary information with dual scores
    if use_time_windows:
        formatted_text += "## Übersicht nach Zeitfenstern\n\n"
        for window, window_chunks in sorted(chunks_by_window.items()):
            avg_llm_score = sum(c['llm_evaluation_score'] for c in window_chunks) / len(window_chunks)
            avg_vector_score = sum(c['vector_similarity_score'] for c in window_chunks) / len(window_chunks)
            formatted_text += f"- **{window}**: {len(window_chunks)} Texte (Ø LLM: {avg_llm_score:.3f}, Ø Vector: {avg_vector_score:.3f})\n"
        formatted_text += "\n"
    
    # Display chunks with both scores
    for window in sorted(chunks_by_window.keys()):
        if use_time_windows:
            formatted_text += f"## Zeitfenster {window}\n\n"
        
        window_chunks = chunks_by_window[window]
        
        for i, chunk in enumerate(window_chunks, 1):
            metadata_chunk = chunk['metadata']
            formatted_text += f"### {i}. {metadata_chunk.get('Artikeltitel', 'Kein Titel')}\n\n"
            
            # Show both evaluation scores
            llm_score = chunk.get('llm_evaluation_score', 0.0)
            vector_score = chunk.get('vector_similarity_score', 0.0)
            
            formatted_text += f"**Datum**: {metadata_chunk.get('Datum', 'Unbekannt')} | "
            formatted_text += f"**LLM-Score**: {llm_score:.3f} | "
            formatted_text += f"**Vector-Score**: {vector_score:.3f}"
            
            # Add evaluation text if available
            eval_text = metadata_chunk.get('evaluation_text', '')
            if eval_text and 'Score:' in eval_text:
                original_score = eval_text.split('Score:')[1].split('-')[0].strip()
                formatted_text += f" ({original_score})"
            
            url = metadata_chunk.get('URL')
            if url and url != 'Keine URL':
                formatted_text += f" | [**Link zum Artikel**]({url})"
            
            formatted_text += "\n\n"
            
            # Show evaluation reasoning if available
            if eval_text and '-' in eval_text:
                reasoning = eval_text.split('-', 1)[1].strip()
                formatted_text += f"**KI-Begründung**: {reasoning}\n\n"
            
            formatted_text += f"**Text**:\n{chunk['content']}\n\n"
            formatted_text += "---\n\n"
    
    return formatted_text

def create_agent_download_comprehensive(retrieved_chunks: Optional[Dict[str, Any]]) -> str:
    """
    Create comprehensive download with all retrieved chunks, evaluations, and dual scores.
    
    Returns:
        Path to the created file, or None if no data
    """
    if not retrieved_chunks:
        return None
    
    try:
        import tempfile
        import json
        from datetime import datetime
        
        # Prepare comprehensive data
        metadata = retrieved_chunks.get('metadata', {})
        evaluations = metadata.get('evaluations', [])
        
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "source": "Der Spiegel RAG System - Agent Search",
                "search_type": "time_windowed_agent",
                "includes_dual_scores": True  # NEW: Indicate dual score support
            },
            "search_configuration": {
                "strategy": metadata.get('strategy', 'unknown'),
                "year_range": metadata.get('year_range', []),
                "chunk_size": metadata.get('chunk_size', 0),
                "keywords": metadata.get('keywords', ''),
                "time_windows": metadata.get('time_windows', []),
                "agent_config": metadata.get('agent_config', {}),
                "search_time": metadata.get('search_time', 0),
                "min_retrieval_relevance_score": metadata.get('min_retrieval_relevance_score', 0.25)  # NEW
            },
            "retrieval_summary": {
                "total_initial_chunks": metadata.get('total_initial_chunks', 0),
                "total_final_chunks": metadata.get('total_final_chunks', 0),
                "window_chunks_map": metadata.get('window_chunks_map', {}),
                "evaluation_count": len(evaluations)
            },
            "selected_chunks": [],
            "all_evaluations": evaluations
        }
        
        # Add selected chunks with dual scores
        for chunk in retrieved_chunks.get('chunks', []):
            chunk_data = {
                "content": chunk.get('content', ''),
                "relevance_score": chunk.get('relevance_score', 0.0),  # Primary UI score (LLM)
                "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),  # NEW: Vector score
                "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),  # NEW: LLM score
                "metadata": chunk.get('metadata', {}),
                "time_window": chunk.get('metadata', {}).get('time_window', 'Unknown'),
                "evaluation_text": chunk.get('metadata', {}).get('evaluation_text', ''),
                "score_details": {  # NEW: Detailed score breakdown
                    "vector_similarity": chunk.get('vector_similarity_score', 0.0),
                    "llm_evaluation": chunk.get('llm_evaluation_score', 0.0),
                    "primary_display_score": chunk.get('relevance_score', 0.0),
                    "evaluation_reasoning": chunk.get('metadata', {}).get('evaluation_text', '')
                }
            }
            export_data["selected_chunks"].append(chunk_data)
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            prefix='spiegel_agent_comprehensive_dual_scores_',
            delete=False,
            encoding='utf-8'
        )
        
        json.dump(export_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        
        logger.info(f"Created comprehensive agent download with dual scores at {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating comprehensive download: {e}")
        return None