# src/ui/handlers/agent_handlers.py - Updated for new architecture
"""
Updated agent handlers to work with the new strategy-based architecture
"""
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import gradio as gr

from src.core.search.strategies import AgentSearchStrategy, SearchConfig
from src.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Global reference to the RAG engine
rag_engine = None

def set_rag_engine(engine: Any) -> None:
    """Set the global RAG engine reference."""
    global rag_engine
    rag_engine = engine

def perform_agent_search_and_analysis(
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
    Perform agent search and analysis in one step using the new architecture.
    
    Returns:
        Tuple of (results_state, status, answer_output, process_output, evaluations_output, chunks_output, metadata_output)
    """
    try:
        if not rag_engine:
            error_msg = "RAG Engine failed to initialize"
            return {}, f"Error: {error_msg}", f"### Error\n\n{error_msg}", "", "", "", ""
        
        if not question.strip():
            error_msg = "Bitte geben Sie eine Frage ein"
            return {}, f"Error: {error_msg}", f"### Error\n\n{error_msg}", "", "", "", ""
        
        start_time = time.time()
        logger.info(f"Starting agent search and analysis for question: '{question}'")
        
        # Clean parameters
        content_description = content_description.strip() if content_description else question
        keywords_cleaned = keywords.strip() if keywords else None
        search_fields = search_in if search_in else ["Text"]
        
        # Prepare filter stages (remove duplicates and sort descending)
        filter_stages = [filter_stage1, filter_stage2, filter_stage3]
        filter_stages = sorted(list(set([f for f in filter_stages if f > 0])), reverse=True)
        
        logger.info(f"Filter stages: {filter_stages}")
        
        # Determine model
        model_to_use = "hu-llm"
        if model == "openai-gpt4o":
            model_to_use = "gpt-4o"
        elif model == "openai-gpt35":
            model_to_use = "gpt-3.5-turbo"
        
        # Determine system prompt
        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        else:
            system_prompt = settings.SYSTEM_PROMPTS.get(system_prompt_template, settings.SYSTEM_PROMPTS["default"])
        
        # Step 1: Create search configuration
        config = SearchConfig(
            content_description=content_description,
            year_range=(year_start, year_end),
            chunk_size=chunk_size,
            keywords=keywords_cleaned,
            search_fields=search_fields,
            enforce_keywords=enforce_keywords,
            top_k=filter_stages[-1] if filter_stages else 10  # Use final filter stage as top_k
        )
        
        # Step 2: Create and execute agent strategy directly
        agent_strategy = AgentSearchStrategy(
            initial_count=initial_count,
            filter_stages=filter_stages,
            llm_service=rag_engine.llm_service,
            model=model_to_use
        )
        
        logger.info("Executing agent search...")
        
        # Call the strategy directly with additional parameters
        search_result = agent_strategy.search(
            config=config,
            vector_store=rag_engine.vector_store,
            question=question,
            openai_api_key=openai_api_key if model_to_use.startswith("gpt") else None
        )
        
        search_time = time.time() - start_time
        
        # Check for search errors
        if "error" in search_result.metadata:
            error_msg = search_result.metadata["error"]
            return {}, f"Error: {error_msg}", f"### Error\n\n{error_msg}", "", "", "", ""
        
        # Step 3: Perform analysis with the found chunks
        logger.info(f"Analyzing {len(search_result.chunks)} chunks...")
        
        # Convert search result chunks to Document format for analysis
        from langchain.docstore.document import Document
        documents = [doc for doc, score in search_result.chunks]
        
        analysis_result = rag_engine.analyze(
            question=question,
            chunks=documents,
            model=model_to_use,
            system_prompt=system_prompt,
            temperature=0.3,
            openai_api_key=openai_api_key if model_to_use.startswith("gpt") else None
        )
        
        total_time = time.time() - start_time
        logger.info(f"Agent search and analysis completed in {total_time:.2f} seconds")
        
        # Prepare results for UI
        results = {
            "answer": analysis_result.answer,
            "chunks": [{"content": doc.page_content, "metadata": doc.metadata, "relevance_score": score} 
                      for doc, score in search_result.chunks],
            "evaluations": search_result.metadata.get("evaluations", []),
            "metadata": {
                **search_result.metadata,
                "analysis_model": analysis_result.model,
                "total_time": total_time,
                "question": question
            }
        }
        
        # Format outputs
        status = f"Suche und Analyse erfolgreich abgeschlossen in {total_time:.2f} Sekunden."
        answer_output = analysis_result.answer
        process_output = format_process_visualization(results)
        evaluations_output = format_evaluations(results)
        chunks_output = format_chunks(results)
        metadata_output = format_metadata(results)
        
        return results, status, answer_output, process_output, evaluations_output, chunks_output, metadata_output
        
    except Exception as e:
        logger.error(f"Error in agent search and analysis: {e}", exc_info=True)
        error_msg = str(e)
        return {}, f"Error: {error_msg}", f"### Error\n\n{error_msg}", "", "", "", ""

def format_process_visualization(results: Dict[str, Any]) -> str:
    """Format the process visualization as HTML."""
    agent_metadata = results.get("metadata", {}).get("agent_metadata", {})
    stage_times = agent_metadata.get("stage_times", [])
    stage_results = agent_metadata.get("stage_results", [])
    initial_count = agent_metadata.get("initial_retrieval_count", 100)
    
    if not stage_times or not stage_results:
        return "<div>Keine Prozessdaten verfügbar.</div>"
    
    html = "<div class='filter-stages'>"
    html += f"<h3>Filterungsprozess ({len(stage_times)} Stufen)</h3>"
    html += f"<p>Gesamtzeit: {agent_metadata.get('total_time', 0):.2f} Sekunden</p>"
    
    for i, ((stage_name, stage_time), stage_result) in enumerate(zip(stage_times, stage_results)):
        percentage = 100.0
        if i == 0:
            max_value = initial_count * 1.2
        else:
            max_value = stage_results[i-1]
            
        if max_value > 0:
            percentage = (stage_result / max_value) * 100
            
        html += f"""
        <div class='filter-stage'>
            <div class='filter-stage-title'>{stage_name} ({stage_time:.2f}s)</div>
            <div class='filter-progress'>
                <div class='filter-bar' style='width: {min(percentage, 100)}%; background-color: #3498db; height: 25px; display: flex; align-items: center; justify-content: center; color: white;'>
                    {stage_result} Texte
                </div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html

def format_evaluations(results: Dict[str, Any]) -> str:
    """Format the chunk evaluations as HTML."""
    evaluations = results.get("evaluations", [])
    
    if not evaluations:
        return "<div>Keine Bewertungen verfügbar.</div>"
    
    html = "<div class='evaluations'>"
    html += f"<h3>Bewertungen der {len(evaluations)} ausgewählten Texte</h3>"
    
    for i, eval_data in enumerate(evaluations):
        title = eval_data.get("title", "Unbekannter Titel")
        date = eval_data.get("date", "Unbekanntes Datum")
        score = eval_data.get("relevance_score", 0.0)
        evaluation = eval_data.get("evaluation", "Keine Bewertung verfügbar")
        
        html += f"""
        <div class='evaluation-card' style='border-left: 4px solid #3498db; padding: 10px; margin-bottom: 10px; background-color: #f8f9fa;'>
            <h4>{i+1}. {title} ({date})</h4>
            <p><strong>Relevanz:</strong> {score:.3f}</p>
            <p><strong>Bewertung:</strong> {evaluation}</p>
        </div>
        """
    
    html += "</div>"
    return html

def format_chunks(results: Dict[str, Any]) -> str:
    """Format the chunks as markdown."""
    chunks = results.get("chunks", [])
    
    if not chunks:
        return "Keine Texte verfügbar."
    
    chunks_by_year = {}
    for chunk in chunks:
        year = chunk["metadata"].get("Jahrgang", "Unknown")
        if year not in chunks_by_year:
            chunks_by_year[year] = []
        chunks_by_year[year].append(chunk)
    
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
    """Format the metadata as markdown."""
    metadata = results.get("metadata", {})
    agent_metadata = metadata.get("agent_metadata", {})
    
    markdown = "## Suchparameter\n"
    markdown += f"- **Frage**: {metadata.get('question', 'Keine Frage')}\n"
    markdown += f"- **Modell**: {metadata.get('analysis_model', 'Unbekannt')}\n"
    markdown += f"- **Gesamtzeit**: {metadata.get('total_time', 0):.2f} Sekunden\n\n"
    
    markdown += "## Agenten-Metadaten\n"
    markdown += f"- **Initiale Textmenge**: {agent_metadata.get('initial_retrieval_count', 0)}\n"
    markdown += f"- **Filterstufen**: {agent_metadata.get('filter_stages', [])}\n"
    markdown += f"- **Finale Textanzahl**: {agent_metadata.get('final_chunk_count', 0)}\n"
    
    stage_times = agent_metadata.get("stage_times", [])
    if stage_times:
        markdown += "\n## Stufen-Zeiten\n"
        for stage_name, stage_time in stage_times:
            markdown += f"- **{stage_name}**: {stage_time:.2f} Sekunden\n"
    
    return markdown