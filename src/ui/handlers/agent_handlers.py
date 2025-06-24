# src/ui/handlers/llm_assisted_handlers.py - Updated with new terminology
"""
Updated handlers for LLM-assisted functionality (formerly agent search).
UPDATED: All terminology changed from "agent" to "LLM-Unterstützte Auswahl"
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
current_llm_assisted_search = None  # UPDATED: from current_agent_search
search_thread = None

def set_rag_engine(engine: Any) -> None:
    """Set the global RAG engine reference."""
    global rag_engine
    rag_engine = engine

def perform_llm_assisted_search(  # UPDATED: from perform_agent_search
    retrieval_query: str,  # UPDATED: from content_description
    chunk_size: int,
    year_start: int,
    year_end: int,
    llm_assisted_use_time_intervals: bool,  # UPDATED: from agent_use_time_windows
    llm_assisted_time_interval_size: int,  # UPDATED: from agent_time_window_size
    chunks_per_interval_initial: int,  # UPDATED: from chunks_per_window_initial
    chunks_per_interval_final: int,  # UPDATED: from chunks_per_window_final
    llm_assisted_min_retrieval_score: float,  # UPDATED: from agent_min_retrieval_score
    llm_assisted_keywords: str,  # UPDATED: from agent_keywords
    llm_assisted_search_in: List[str],  # UPDATED: from agent_search_in
    llm_assisted_model: str,  # UPDATED: from agent_model
    llm_assisted_temperature: float,  # NEW: temperature parameter
    llm_assisted_system_prompt_template: str,  # UPDATED: from agent_system_prompt_template
    llm_assisted_system_prompt_text: str,  # UPDATED: from agent_system_prompt_text
    progress_callback: Optional[Any] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform LLM-assisted search with enhanced system prompt support.
    UPDATED: Renamed from perform_agent_search with new terminology and temperature support.
    """
    global current_llm_assisted_search  # UPDATED: from current_agent_search
    
    try:
        if not rag_engine:
            return "Error: RAG Engine nicht initialisiert", None
        
        if not retrieval_query.strip():  # UPDATED: from content_description
            return "Error: Retrieval-Query ist erforderlich", None  # UPDATED terminology
        
        start_time = time.time()
        logger.info(f"Starting LLM-assisted search: '{retrieval_query}'")  # UPDATED terminology
        logger.info(f"Initial retrieval count: {chunks_per_interval_initial}, Final: {chunks_per_interval_final}")
        logger.info(f"LLM temperature: {llm_assisted_temperature}")  # NEW: log temperature
        
        # Clean parameters
        retrieval_query = retrieval_query.strip()  # UPDATED: from content_description
        keywords_cleaned = llm_assisted_keywords.strip() if llm_assisted_keywords else None  # UPDATED
        search_fields = llm_assisted_search_in if llm_assisted_search_in else ["Text"]  # UPDATED
        
        # Use the editable system prompt text directly
        system_prompt = llm_assisted_system_prompt_text.strip()  # UPDATED
        
        # Fallback to template if somehow empty
        if not system_prompt:
            logger.warning("System prompt text is empty, falling back to template")
            system_prompt = settings.LLM_ASSISTED_SYSTEM_PROMPTS.get(  # UPDATED: from AGENT_SYSTEM_PROMPTS
                llm_assisted_system_prompt_template,  # UPDATED
                settings.LLM_ASSISTED_SYSTEM_PROMPTS["standard_evaluation"]  # UPDATED
            )
        
        logger.info(f"Using editable system prompt (length: {len(system_prompt)} chars)")
        logger.info(f"System prompt preview: {system_prompt[:150]}...")
        
        # Create search configurations with updated field names
        search_config = SearchConfig(
            content_description=retrieval_query,  # UPDATED: from content_description
            year_range=(year_start, year_end),
            chunk_size=chunk_size,
            keywords=keywords_cleaned,
            search_fields=search_fields,
            enforce_keywords=True,  # Always true now
            top_k=100  # Not used in LLM-assisted search, but required
        )
        
        # UPDATED: Create agent config with new field names and temperature
        agent_config = AgentSearchConfig(
            use_time_windows=llm_assisted_use_time_intervals,  # UPDATED
            time_window_size=llm_assisted_time_interval_size,  # UPDATED
            chunks_per_window_initial=chunks_per_interval_initial,  # UPDATED
            chunks_per_window_final=chunks_per_interval_final,  # UPDATED
            agent_model=llm_assisted_model,  # UPDATED
            agent_system_prompt=system_prompt,
            min_retrieval_relevance_score=llm_assisted_min_retrieval_score,  # UPDATED
            evaluation_temperature=llm_assisted_temperature  # NEW: add temperature
        )
        
        # Create and execute LLM-assisted strategy (still using internal agent strategy)
        llm_assisted_strategy = TimeWindowedAgentStrategy(  # Keep using existing strategy internally
            llm_service=rag_engine.llm_service,
            agent_config=agent_config
        )
        
        # Store reference for potential cancellation
        current_llm_assisted_search = llm_assisted_strategy  # UPDATED
        
        logger.info("Executing LLM-assisted search with enhanced time interval processing...")  # UPDATED
        
        # Progress callback wrapper
        def progress_wrapper(message: str, progress: float):
            if progress_callback:
                progress_callback(message, progress)
        
        # Execute search
        search_result = llm_assisted_strategy.search(
            config=search_config,
            vector_store=rag_engine.vector_store,
            progress_callback=progress_wrapper
        )
        
        search_time = time.time() - start_time
        
        # Clear reference
        current_llm_assisted_search = None  # UPDATED
        
        # Check for errors
        if "error" in search_result.metadata:
            error_msg = search_result.metadata["error"]
            logger.error(f"LLM-assisted search error: {error_msg}")  # UPDATED
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
                'vector_similarity_score': vector_score,  # Vector similarity score
                'llm_evaluation_score': llm_eval_score  # Explicit LLM score
            })
        
        # Build comprehensive results dictionary with updated terminology
        results = {
            'chunks': chunks_for_ui,
            'metadata': {
                'search_time': search_time,
                'strategy': 'llm_assisted_time_intervals',  # UPDATED: from time_windowed_agent
                'chunk_size': chunk_size,
                'year_range': [year_start, year_end],
                'keywords': keywords_cleaned,
                'retrieval_method': 'llm_assisted_time_intervals' if llm_assisted_use_time_intervals else 'llm_assisted_global',  # UPDATED
                'min_retrieval_relevance_score': llm_assisted_min_retrieval_score,  # UPDATED
                'evaluation_temperature': llm_assisted_temperature,  # NEW: store temperature
                'system_prompt_used': system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt,
                'system_prompt_template': llm_assisted_system_prompt_template,  # UPDATED
                **search_result.metadata  # Include all strategy metadata
            }
        }
        
        num_chunks = len(chunks_for_ui)
        logger.info(f"LLM-assisted search completed: {num_chunks} chunks in {search_time:.2f}s")  # UPDATED
        
        # Create success message with updated terminology
        method_name = "LLM-Unterstützte Auswahl mit Zeit-Intervallen" if llm_assisted_use_time_intervals else "LLM-Unterstützte Auswahl (global)"  # UPDATED
        
        info_text = f"""
        ### Quellen erfolgreich durch KI-Bewertung ausgewählt ({method_name})
        
        **Retrieval-Query**: {retrieval_query}  
        **Zeitraum**: {year_start} - {year_end}  
        **Anzahl ausgewählter Quellen**: {num_chunks}  
        **Suchzeit**: {search_time:.2f} Sekunden  
        **Bewertungsmodell**: {llm_assisted_model}  
        **Bewertungstemperatur**: {llm_assisted_temperature}
        **Mindest-Retrieval-Score**: {llm_assisted_min_retrieval_score}
        **System-Prompt**: {llm_assisted_system_prompt_template} (angepasst)
        """
        
        if llm_assisted_use_time_intervals:  # UPDATED
            intervals = search_result.metadata.get('time_windows', [])  # Keep internal naming
            total_initial = search_result.metadata.get('total_initial_chunks', 0)
            info_text += f"""
            **Zeit-Intervalle**: {len(intervals)} Intervalle à {llm_assisted_time_interval_size} Jahre
            **Initial abgerufen**: {total_initial} Texte insgesamt
            **Pro Intervall**: {chunks_per_interval_initial} → {chunks_per_interval_final} Texte
            """
        
        if keywords_cleaned:
            info_text += f"""
            **Schlagwörter**: {keywords_cleaned}
            **Filterung**: Immer aktiviert (strikte Filterung)
            """
        
        return info_text, results
        
    except Exception as e:
        logger.error(f"Error in LLM-assisted search: {e}", exc_info=True)  # UPDATED
        current_llm_assisted_search = None  # UPDATED
        return f"Error: {str(e)}", None

def cancel_llm_assisted_search() -> str: 
    """Cancel the currently running LLM-assisted search."""
    global current_llm_assisted_search 
    
    if current_llm_assisted_search: 
        logger.info("Cancelling LLM-assisted search...")  
        current_llm_assisted_search.cancel_search()
        return "LLM-Unterstützte Auswahl wird abgebrochen..." 
    else:
        return "Keine aktive Suche zum Abbrechen."

def perform_llm_assisted_search_threaded( 
    retrieval_query: str, 
    chunk_size: int,
    year_start: int,
    year_end: int,
    llm_assisted_use_time_intervals: bool, 
    llm_assisted_time_interval_size: int, 
    chunks_per_interval_initial: int, 
    chunks_per_interval_final: int,  
    llm_assisted_min_retrieval_score: float, 
    llm_assisted_keywords: str, 
    llm_assisted_search_in: List[str], 
    llm_assisted_model: str,  
    llm_assisted_temperature: float,  
    llm_assisted_system_prompt_template: str, 
    llm_assisted_system_prompt_text: str  
) -> Tuple[str, Dict[str, Any], str, gr.update, gr.update, gr.update, gr.update, gr.update]:
    """
    Perform LLM-assisted search in a thread with UI updates and enhanced system prompt support.
    """
    global search_thread
    
    try:
        # Initialize UI state
        progress_state = gr.update(value="LLM-Unterstützte Auswahl gestartet...", visible=True) 
        cancel_btn_state = gr.update(visible=True)
        search_btn_state = gr.update(interactive=False)
        
        # Progress tracking
        progress_messages = []
        
        def progress_callback(message: str, progress: float):
            progress_messages.append(f"{progress*100:.0f}% - {message}")
            # Update progress display
            return gr.update(value=f"**Fortschritt**: {message} ({progress*100:.0f}%)")
        
        # Log system prompt usage with new terminology
        logger.info(f"LLM-assisted search using editable system prompt (template: {llm_assisted_system_prompt_template})")  # UPDATED
        logger.info(f"System prompt length: {len(llm_assisted_system_prompt_text)} characters")
        logger.info(f"Evaluation temperature: {llm_assisted_temperature}")  # NEW
        
        # Run search with updated parameters
        search_status, retrieved_chunks = perform_llm_assisted_search(  # UPDATED
            retrieval_query, chunk_size, year_start, year_end,  # UPDATED
            llm_assisted_use_time_intervals, llm_assisted_time_interval_size,  # UPDATED
            chunks_per_interval_initial, chunks_per_interval_final,  # UPDATED
            llm_assisted_min_retrieval_score, llm_assisted_keywords, llm_assisted_search_in,  # UPDATED
            llm_assisted_model, llm_assisted_temperature, llm_assisted_system_prompt_template,  # UPDATED, NEW
            llm_assisted_system_prompt_text,  # UPDATED
            progress_callback
        )
        
        # Format chunks for display
        if retrieved_chunks and retrieved_chunks.get('chunks'):
            formatted_chunks = format_llm_assisted_chunks_with_dual_scores(retrieved_chunks)  # UPDATED
            num_chunks = len(retrieved_chunks.get('chunks'))
            
            # Update accordion states for successful search
            retrieved_texts_state = gr.update(open=True)
            question_state = gr.update(open=True)  # This should be analysis_state now
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
            search_status, retrieved_chunks, 
            'llm_assisted',  # search_mode radio value
            final_search_btn, final_cancel_btn, final_progress,
            retrieved_texts_state, question_state
        )
        
    except Exception as e:
        logger.error(f"Threaded LLM-assisted search failed: {e}", exc_info=True)  # UPDATED
        
        # Reset UI on error
        error_status = f"Error: {str(e)}"
        final_progress = gr.update(visible=False)
        final_cancel_btn = gr.update(visible=False)
        final_search_btn = gr.update(interactive=True)        
        search_thread = None
        
        return (
            error_status, None,  # search_status, retrieved_chunks_state
            'llm_assisted',  # search_mode radio value
            final_search_btn, final_cancel_btn, final_progress,
            gr.update(open=False), gr.update(open=False)
        )

def format_llm_assisted_chunks_with_dual_scores(retrieved_chunks: Dict[str, Any]) -> str: 
    """
    Format LLM-assisted search results for display, highlighting both vector and LLM scores.
    UPDATED: Enhanced with new terminology and temperature information.
    """
    chunks = retrieved_chunks.get('chunks', [])
    metadata = retrieved_chunks.get('metadata', {})
    
    if not chunks:
        return "Keine Texte durch KI-Bewertung ausgewählt."
    
    # Group by time interval if available
    chunks_by_interval = {}  
    evaluations = metadata.get('evaluations', [])
    
    for chunk in chunks:
        interval = chunk['metadata'].get('time_window', 'Global')  # Keep internal naming but display as interval
        if interval not in chunks_by_interval:
            chunks_by_interval[interval] = []
        chunks_by_interval[interval].append(chunk)
    
    # Format output with updated terminology
    use_time_intervals = len(chunks_by_interval) > 1 or 'Global' not in chunks_by_interval  
    
    formatted_text = f"# KI-bewertete Quellen ({len(chunks)} ausgewählt)\n\n"
    
    # Enhanced: Add system prompt and temperature information
    system_prompt_info = metadata.get('system_prompt_used', '')
    system_prompt_template = metadata.get('system_prompt_template', 'unknown')
    evaluation_temperature = metadata.get('evaluation_temperature', 'nicht angegeben') 
    
    if system_prompt_info:
        formatted_text += f"**Verwendeter System-Prompt**: {system_prompt_template} (angepasst)\n"
        formatted_text += f"**Bewertungstemperatur**: {evaluation_temperature}\n"
        formatted_text += f"**Prompt-Vorschau**: {system_prompt_info}\n\n"
    
    # Add summary information with dual scores
    if use_time_intervals:
        formatted_text += "## Übersicht nach Zeit-Intervallen\n\n"  
        for interval, interval_chunks in sorted(chunks_by_interval.items()): 
            avg_llm_score = sum(c['llm_evaluation_score'] for c in interval_chunks) / len(interval_chunks)
            avg_vector_score = sum(c['vector_similarity_score'] for c in interval_chunks) / len(interval_chunks)
            formatted_text += f"- **{interval}**: {len(interval_chunks)} Texte (Ø LLM: {avg_llm_score:.3f}, Ø Vector: {avg_vector_score:.3f})\n"
        formatted_text += "\n"
    
    # Display chunks with both scores
    for interval in sorted(chunks_by_interval.keys()):  
        if use_time_intervals:
            formatted_text += f"## Zeitintervall {interval}\n\n"  # UPDATED terminology
        
        interval_chunks = chunks_by_interval[interval]  # UPDATED variable names
        
        for i, chunk in enumerate(interval_chunks, 1):
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

def create_llm_assisted_download_comprehensive(retrieved_chunks: Optional[Dict[str, Any]]) -> str:  # UPDATED: from create_agent_download_comprehensive
    """
    Create comprehensive download with all retrieved chunks, evaluations, dual scores, and system prompt info.
    UPDATED: Enhanced with new terminology and temperature information.
    
    Returns:
        Path to the created file, or None if no data
    """
    if not retrieved_chunks:
        return None
    
    try:
        import tempfile
        import json
        from datetime import datetime
        
        # Prepare comprehensive data with updated terminology
        metadata = retrieved_chunks.get('metadata', {})
        evaluations = metadata.get('evaluations', [])
        
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "source": "SPIEGEL RAG System - LLM-Unterstützte Auswahl Umfassend",  # UPDATED
                "search_type": "llm_assisted_time_intervals",  # UPDATED
                "includes_dual_scores": True,
                "includes_system_prompt_info": True,
                "includes_temperature_info": True  # NEW
            },
            "search_configuration": {
                "strategy": metadata.get('strategy', 'unknown'),
                "year_range": metadata.get('year_range', []),
                "chunk_size": metadata.get('chunk_size', 0),
                "keywords": metadata.get('keywords', ''),
                "time_intervals": metadata.get('time_windows', []),  # Keep internal naming
                "llm_assisted_config": metadata.get('agent_config', {}),  # Map to new name
                "search_time": metadata.get('search_time', 0),
                "min_retrieval_relevance_score": metadata.get('min_retrieval_relevance_score', 0.25),
                "evaluation_temperature": metadata.get('evaluation_temperature', 'not specified'),  # NEW
                # System prompt information
                "system_prompt_template": metadata.get('system_prompt_template', 'unknown'),
                "system_prompt_preview": metadata.get('system_prompt_used', ''),
                "system_prompt_customized": True  # Indicates editable prompt was used
            },
            "retrieval_summary": {
                "total_initial_chunks": metadata.get('total_initial_chunks', 0),
                "total_final_chunks": metadata.get('total_final_chunks', 0),
                "interval_chunks_map": metadata.get('window_chunks_map', {}),  # Map to new name
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
                "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),  # Vector score
                "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),  # LLM score
                "metadata": chunk.get('metadata', {}),
                "zeit_interval": chunk.get('metadata', {}).get('time_window', 'Unknown'),  # UPDATED terminology
                "evaluation_text": chunk.get('metadata', {}).get('evaluation_text', ''),
                "score_details": {  # Detailed score breakdown
                    "vector_similarity": chunk.get('vector_similarity_score', 0.0),
                    "llm_evaluation": chunk.get('llm_evaluation_score', 0.0),
                    "primary_display_score": chunk.get('relevance_score', 0.0),
                    "evaluation_reasoning": chunk.get('metadata', {}).get('evaluation_text', ''),
                    "score_difference": chunk.get('llm_evaluation_score', 0.0) - chunk.get('vector_similarity_score', 0.0),
                    "evaluation_temperature": metadata.get('evaluation_temperature', 'not specified')  # NEW
                }
            }
            export_data["selected_chunks"].append(chunk_data)
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            prefix='spiegel_llm_assisted_comprehensive_enhanced_',  # UPDATED
            delete=False,
            encoding='utf-8'
        )
        
        json.dump(export_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        
        logger.info(f"Created enhanced comprehensive LLM-assisted download with system prompt and temperature info at {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating enhanced comprehensive download: {e}")
        return None