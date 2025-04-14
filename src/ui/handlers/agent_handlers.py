# src/ui/handlers/agent_handlers.py
"""
Handler functions for agent-based search operations.
These functions connect the agent panel UI components to the RAG engine.
"""
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import gradio as gr

# Configure logging
logger = logging.getLogger(__name__)

# Global reference to the RAG engine
# This will be shared with the main app
rag_engine = None

def set_rag_engine(engine: Any) -> None:
    """
    Set the global RAG engine reference.
    
    Args:
        engine: The SpiegelRAGEngine instance
    """
    global rag_engine
    rag_engine = engine

def perform_agent_search(
    question: str,
    content_description: str,
    year_start: int,
    year_end: int,
    chunk_size: int,
    keywords: str,
    search_in: List[str],
    enforce_keywords: bool,
    initial_count: int,
    filter_stage1: int,
    filter_stage2: int,
    filter_stage3: int,
    model: str,
    openai_api_key: str,
    system_prompt_template: str,
    custom_system_prompt: str
) -> Dict[str, Any]:
    """
    Perform agent-based search with the RAG engine.
    
    Args:
        Various search parameters from the UI
        
    Returns:
        Dict with search results and metadata
    """
    try:
        if not rag_engine:
            return {
                "error": "RAG Engine failed to initialize",
                "status": "Error: System not properly initialized."
            }
        
        # Prepare filter stages
        filter_stages = [filter_stage1, filter_stage2, filter_stage3]
        # Remove duplicates and ensure descending order
        filter_stages = sorted(list(set(filter_stages)), reverse=True)
        
        # Clear empty content description
        if not content_description.strip():
            content_description = None
            
        # Process keywords
        keywords_to_use = keywords.strip() if keywords and keywords.strip() else None
        
        # Determine system prompt
        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        else:
            from src.config import settings
            system_prompt = settings.SYSTEM_PROMPTS.get(system_prompt_template, settings.SYSTEM_PROMPTS["default"])
        
        # Determine model name
        model_to_use = "hu-llm"  # Default
        if model == "openai-gpt4o":
            model_to_use = "gpt-4o"
        elif model == "openai-gpt35":
            model_to_use = "gpt-3.5-turbo"
        
        # Set search fields
        search_fields = search_in if search_in else ["Text"]
        
        logger.info(f"Starting agent search with question: '{question}'")
        logger.info(f"Filter stages: {filter_stages}")
        
        # Perform agent-based search
        start_time = time.time()
        results = rag_engine.agent_search(
            question=question,
            content_description=content_description,
            year_range=[year_start, year_end],
            chunk_size=chunk_size,
            keywords=keywords_to_use,
            search_in=search_fields,
            model=model_to_use,
            openai_api_key=openai_api_key,
            system_prompt=system_prompt,
            initial_retrieval_count=initial_count,
            filter_stages=filter_stages,
            enforce_keywords=enforce_keywords
        )
        
        search_time = time.time() - start_time
        logger.info(f"Agent search completed in {search_time:.2f} seconds")
        
        # Add overall search time to results
        results["search_time"] = search_time
        
        # Add status message
        if "error" in results:
            results["status"] = f"Error: {results['error']}"
        else:
            results["status"] = f"Suche erfolgreich abgeschlossen in {search_time:.2f} Sekunden."
        
        return results
        
    except Exception as e:
        logger.error(f"Error in agent search: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": f"Error: {str(e)}"
        }

def perform_agent_search_and_update_ui(
    question: str,
    content_description: str,
    year_start: int,
    year_end: int,
    chunk_size: int,
    keywords: str,
    search_in: List[str],
    enforce_keywords: bool,
    initial_count: int,
    filter_stage1: int,
    filter_stage2: int,
    filter_stage3: int,
    model: str,
    openai_api_key: str,
    system_prompt_template: str,
    custom_system_prompt: str
) -> Tuple[Dict[str, Any], str, str, str, str, str]:
    """
    Perform agent search and update the UI components.
    
    Args:
        Various search parameters from the UI
        
    Returns:
        Tuple of (results_state, status, answer_output, process_output, evaluations_output, chunks_output, metadata_output)
    """
    # Perform the search
    results = perform_agent_search(
        question, content_description, year_start, year_end, 
        chunk_size, keywords, search_in, enforce_keywords,
        initial_count, filter_stage1, filter_stage2, filter_stage3,
        model, openai_api_key, system_prompt_template, custom_system_prompt
    )
    
    # Status update
    status = results.get("status", "Unknown status")
    
    # Check for errors
    if "error" in results:
        return (
            results,  # results_state
            status,   # status
            f"### Error\n\n{results['error']}",  # answer_output
            "Error occurred during processing.",  # process_output
            "No evaluations available due to error.",  # evaluations_output
            "No chunks available due to error.",  # chunks_output
            "No metadata available due to error."  # metadata_output
        )
    
    # Format the answer
    answer_output = results.get("answer", "No answer generated")
    
    # Format the process visualization
    process_output = format_process_visualization(results)
    
    # Format the evaluations
    evaluations_output = format_evaluations(results)
    
    # Format the chunks
    chunks_output = format_chunks(results)
    
    # Format the metadata
    metadata_output = format_metadata(results)
    
    return (
        results,        # results_state
        status,         # status
        answer_output,  # answer_output
        process_output, # process_output
        evaluations_output, # evaluations_output
        chunks_output,  # chunks_output
        metadata_output # metadata_output
    )
    
def format_process_visualization(results: Dict[str, Any]) -> str:
    """
    Format the process visualization as HTML.
    
    Args:
        results: Search results dictionary
        
    Returns:
        HTML string with process visualization
    """
    # Extract data for visualization
    agent_metadata = results.get("metadata", {}).get("agent_metadata", {})
    stage_times = agent_metadata.get("stage_times", [])
    stage_results = agent_metadata.get("stage_results", [])
    initial_count = agent_metadata.get("initial_retrieval_count", 100)
    
    # If no data available
    if not stage_times or not stage_results:
        return "<div>Keine Prozessdaten verfügbar.</div>"
    
    # Create HTML for visualization
    html = "<div class='filter-stages'>"
    
    # Add overall info
    html += f"<h3>Filterungsprozess ({len(stage_times)} Stufen)</h3>"
    html += f"<p>Gesamtzeit: {agent_metadata.get('total_time', 0):.2f} Sekunden</p>"
    
    # Add each stage
    for i, ((stage_name, stage_time), stage_result) in enumerate(zip(stage_times, stage_results)):
        percentage = 100.0
        if i == 0:  # Initial retrieval
            max_value = initial_count * 1.2  # Add some margin
        else:
            max_value = stage_results[i-1]  # Previous stage result
            
        # Calculate percentage (avoid division by zero)
        if max_value > 0:
            percentage = (stage_result / max_value) * 100
            
        html += f"""
        <div class='filter-stage'>
            <div class='filter-stage-title'>{stage_name} ({stage_time:.2f}s)</div>
            <div class='filter-progress'>
                <div class='filter-bar' style='width: {min(percentage, 100)}%;'>{stage_result} Texte</div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html

def format_evaluations(results: Dict[str, Any]) -> str:
    """
    Format the chunk evaluations as HTML.
    
    Args:
        results: Search results dictionary
        
    Returns:
        HTML string with evaluations
    """
    evaluations = results.get("evaluations", [])
    
    if not evaluations:
        return "<div>Keine Bewertungen verfügbar.</div>"
    
    html = "<div class='evaluations'>"
    html += f"<h3>Bewertungen der {len(evaluations)} ausgewählten Texte</h3>"
    
    for i, eval_data in enumerate(evaluations):
        # Extract data
        title = eval_data.get("title", "Unbekannter Titel")
        date = eval_data.get("date", "Unbekanntes Datum")
        score = eval_data.get("relevance_score", 0.0)
        evaluation = eval_data.get("evaluation", "Keine Bewertung verfügbar")
        
        # Format as card
        html += f"""
        <div class='evaluation-card'>
            <h4>{i+1}. {title} ({date})</h4>
            <p><strong>Relevanz:</strong> {score:.3f}</p>
            <p><strong>Bewertung:</strong> {evaluation}</p>
        </div>
        """
    
    html += "</div>"
    return html

def format_chunks(results: Dict[str, Any]) -> str:
    """
    Format the chunks as markdown.
    
    Args:
        results: Search results dictionary
        
    Returns:
        Markdown string with chunks
    """
    chunks = results.get("chunks", [])
    
    if not chunks:
        return "Keine Texte verfügbar."
    
    # Group chunks by year for better readability
    chunks_by_year = {}
    for chunk in chunks:
        year = chunk["metadata"].get("Jahrgang", "Unknown")
        if year not in chunks_by_year:
            chunks_by_year[year] = []
        chunks_by_year[year].append(chunk)
    
    # Format chunks
    markdown = f"# Gefundene Texte ({len(chunks)})\n\n"
    
    for year in sorted(chunks_by_year.keys()):
        markdown += f"## {year}\n\n"
        
        for i, chunk in enumerate(chunks_by_year[year], 1):
            metadata = chunk["metadata"]
            markdown += f"### {i}. {metadata.get('Artikeltitel', 'Kein Titel')}\n\n"
            markdown += f"**Datum**: {metadata.get('Datum', 'Unbekannt')} | "
            markdown += f"**Relevanz**: {chunk['relevance_score']:.3f}"
            
            url = metadata.get('URL')
            if url and url != 'Keine URL':
                markdown += f" | [**Link zum Artikel**]({url})"
                
            markdown += "\n\n"
            markdown += f"**Text**:\n{chunk['content']}\n\n"
            markdown += "---\n\n"
    
    return markdown

def format_metadata(results: Dict[str, Any]) -> str:
    """
    Format the metadata as markdown.
    
    Args:
        results: Search results dictionary
        
    Returns:
        Markdown string with metadata
    """
    metadata = results.get("metadata", {})
    agent_metadata = metadata.get("agent_metadata", {})
    
    markdown = "## Suchparameter\n"
    markdown += f"- **Frage**: {metadata.get('question', 'Keine Frage')}\n"
    markdown += f"- **Modell**: {metadata.get('model', 'Unbekannt')}\n"
    markdown += f"- **Suchzeit**: {results.get('search_time', 0):.2f} Sekunden\n\n"
    
    markdown += "## Agenten-Metadaten\n"
    markdown += f"- **Initiale Textmenge**: {agent_metadata.get('initial_retrieval_count', 0)}\n"
    markdown += f"- **Filterstufen**: {agent_metadata.get('filter_stages', [])}\n"
    
    # Add stage times
    stage_times = agent_metadata.get("stage_times", [])
    if stage_times:
        markdown += "\n## Stufen-Zeiten\n"
        for stage_name, stage_time in stage_times:
            markdown += f"- **{stage_name}**: {stage_time:.2f} Sekunden\n"
    
    # Add stage results
    stage_results = agent_metadata.get("stage_results", [])
    if stage_results and stage_times:
        markdown += "\n## Stufen-Ergebnisse\n"
        for i, (stage_count, (stage_name, _)) in enumerate(zip(stage_results, stage_times)):
            markdown += f"- **{stage_name}**: {stage_count} Texte\n"
    
    return markdown