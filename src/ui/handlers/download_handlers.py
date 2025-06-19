# src/ui/handlers/download_handlers.py - Updated for dual score support
"""
Updated download handlers that properly create temporary files for Gradio downloads.
Now includes support for both vector similarity and LLM evaluation scores.
"""
import json
import csv
import os
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import gradio as gr

# Configure logging
logger = logging.getLogger(__name__)

def create_download_json(retrieved_chunks: Optional[Dict[str, Any]]) -> str:
    """
    Create a JSON file for download containing retrieved chunks and metadata.
    Enhanced to include both vector and LLM scores when available.
    
    Args:
        retrieved_chunks: Dictionary containing chunks and metadata from search
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            logger.warning("No chunks available for JSON download")
            return None
        
        # Determine if this is agent search with dual scores
        has_dual_scores = any(
            'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
            for chunk in retrieved_chunks.get('chunks', [])
        )
        
        # Prepare data for JSON export
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "source": "Der Spiegel RAG System",
                "total_chunks": len(retrieved_chunks.get('chunks', [])),
                "has_dual_scores": has_dual_scores  # NEW: Indicate if dual scores are available
            },
            "search_metadata": retrieved_chunks.get('metadata', {}),
            "chunks": []
        }
        
        # Process each chunk
        for i, chunk in enumerate(retrieved_chunks.get('chunks', [])):
            chunk_data = {
                "chunk_id": i + 1,
                "relevance_score": chunk.get('relevance_score', 0.0),
                "content": chunk.get('content', ''),
                "metadata": {
                    "titel": chunk.get('metadata', {}).get('Artikeltitel', 'Kein Titel'),
                    "datum": chunk.get('metadata', {}).get('Datum', 'Unbekannt'),
                    "jahrgang": chunk.get('metadata', {}).get('Jahrgang', None),
                    "ausgabe": chunk.get('metadata', {}).get('Ausgabe', None),
                    "url": chunk.get('metadata', {}).get('URL', ''),
                    "autoren": chunk.get('metadata', {}).get('Autoren', ''),
                    "schlagworte": chunk.get('metadata', {}).get('Schlagworte', ''),
                    "untertitel": chunk.get('metadata', {}).get('Untertitel', ''),
                    "nr_in_issue": chunk.get('metadata', {}).get('nr_in_issue', None)
                },
                "search_context": {
                    "time_window": chunk.get('metadata', {}).get('time_window', None),
                    "window_start": chunk.get('metadata', {}).get('window_start', None),
                    "window_end": chunk.get('metadata', {}).get('window_end', None)
                }
            }
            
            # NEW: Add dual scores if available (for agent search)
            if has_dual_scores:
                chunk_data["scoring"] = {
                    "primary_relevance_score": chunk.get('relevance_score', 0.0),
                    "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),
                    "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),
                    "evaluation_text": chunk.get('metadata', {}).get('evaluation_text', ''),
                    "score_type": "dual_scores_agent" if 'llm_evaluation_score' in chunk else "single_score_standard"
                }
            
            export_data["chunks"].append(chunk_data)
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_suffix = "_dual_scores" if has_dual_scores else ""
        
        # Create temporary file with proper name
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            prefix=f'spiegel_rag{filename_suffix}_', 
            delete=False,  # Important: don't delete automatically
            encoding='utf-8'
        )
        
        # Write JSON content to temporary file
        json.dump(export_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        
        logger.info(f"Created JSON download with {len(export_data['chunks'])} chunks at {temp_file.name}")
        
        # Return the file path
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating JSON download: {e}")
        return None

def create_download_csv(retrieved_chunks: Optional[Dict[str, Any]]) -> str:
    """
    Create a CSV file for download containing retrieved chunks and metadata.
    Enhanced to include both vector and LLM scores when available.
    
    Args:
        retrieved_chunks: Dictionary containing chunks and metadata from search
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            logger.warning("No chunks available for CSV download")
            return None
        
        # Determine if this is agent search with dual scores
        has_dual_scores = any(
            'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
            for chunk in retrieved_chunks.get('chunks', [])
        )
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_suffix = "_dual_scores" if has_dual_scores else ""
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            prefix=f'spiegel_rag{filename_suffix}_', 
            delete=False,  # Important: don't delete automatically
            encoding='utf-8',
            newline=''  # Important for CSV files on Windows
        )
        
        # Create CSV writer
        writer = csv.writer(temp_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Write CSV header - ENHANCED for dual scores
        if has_dual_scores:
            headers = [
                'chunk_id',
                'primary_relevance_score',  # Main UI display score
                'vector_similarity_score',  # NEW: Vector retrieval score
                'llm_evaluation_score',     # NEW: LLM evaluation score
                'score_difference',         # NEW: Difference between LLM and vector scores
                'titel',
                'datum',
                'jahrgang',
                'ausgabe',
                'autoren',
                'schlagworte',
                'untertitel',
                'url',
                'nr_in_issue',
                'time_window',
                'evaluation_text',          # NEW: LLM evaluation reasoning
                'content_preview',
                'content_length',
                'full_content'
            ]
        else:
            # Standard headers for non-agent search
            headers = [
                'chunk_id',
                'relevance_score',
                'titel',
                'datum',
                'jahrgang',
                'ausgabe',
                'autoren',
                'schlagworte',
                'untertitel',
                'url',
                'nr_in_issue',
                'time_window',
                'content_preview',
                'content_length',
                'full_content'
            ]
        
        writer.writerow(headers)
        
        # Write data rows
        for i, chunk in enumerate(retrieved_chunks.get('chunks', [])):
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            
            # Create safe content preview (first 200 chars)
            content_preview = content[:200].replace('\n', ' ').replace('\r', ' ')
            if len(content) > 200:
                content_preview += '...'
            
            # Clean content for CSV (remove problematic characters)
            clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Replace multiple spaces with single space
            clean_content = ' '.join(clean_content.split())
            
            if has_dual_scores:
                # Enhanced row with dual scores
                vector_score = chunk.get('vector_similarity_score', 0.0)
                llm_score = chunk.get('llm_evaluation_score', 0.0)
                primary_score = chunk.get('relevance_score', 0.0)
                score_difference = llm_score - vector_score if (llm_score and vector_score) else 0.0
                evaluation_text = metadata.get('evaluation_text', '')
                
                row = [
                    i + 1,  # chunk_id
                    primary_score,  # primary_relevance_score
                    vector_score,   # vector_similarity_score
                    llm_score,      # llm_evaluation_score
                    score_difference,  # score_difference
                    metadata.get('Artikeltitel', 'Kein Titel'),
                    metadata.get('Datum', 'Unbekannt'),
                    metadata.get('Jahrgang', ''),
                    metadata.get('Ausgabe', ''),
                    metadata.get('Autoren', ''),
                    metadata.get('Schlagworte', ''),
                    metadata.get('Untertitel', ''),
                    metadata.get('URL', ''),
                    metadata.get('nr_in_issue', ''),
                    metadata.get('time_window', ''),
                    evaluation_text,  # evaluation_text
                    content_preview,
                    len(content),
                    clean_content
                ]
            else:
                # Standard row for non-agent search
                row = [
                    i + 1,  # chunk_id
                    chunk.get('relevance_score', 0.0),
                    metadata.get('Artikeltitel', 'Kein Titel'),
                    metadata.get('Datum', 'Unbekannt'),
                    metadata.get('Jahrgang', ''),
                    metadata.get('Ausgabe', ''),
                    metadata.get('Autoren', ''),
                    metadata.get('Schlagworte', ''),
                    metadata.get('Untertitel', ''),
                    metadata.get('URL', ''),
                    metadata.get('nr_in_issue', ''),
                    metadata.get('time_window', ''),
                    content_preview,
                    len(content),
                    clean_content
                ]
            
            writer.writerow(row)
        
        temp_file.close()
        
        score_info = "with dual scores" if has_dual_scores else "with single scores"
        logger.info(f"Created CSV download {score_info} with {len(retrieved_chunks.get('chunks', []))} chunks at {temp_file.name}")
        
        # Return the file path
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating CSV download: {e}")
        return None

def create_agent_download_json(agent_results: Optional[Dict[str, Any]]) -> str:
    """
    Create a JSON file for download containing agent search results with evaluations.
    Enhanced to properly handle dual scores.
    
    Args:
        agent_results: Dictionary containing agent search results and evaluations
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not agent_results or not agent_results.get('chunks'):
            logger.warning("No agent results available for JSON download")
            return None
        
        # Prepare data for JSON export
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "source": "Der Spiegel RAG System - Agent Search",
                "total_chunks": len(agent_results.get('chunks', [])),
                "search_type": "agent_based",
                "includes_dual_scores": True  # Agent search always has dual scores
            },
            "search_metadata": agent_results.get('metadata', {}),
            "agent_evaluations": agent_results.get('evaluations', []),
            "answer": agent_results.get('answer', ''),
            "chunks": []
        }
        
        # Process each chunk with agent evaluation data and dual scores
        for i, chunk in enumerate(agent_results.get('chunks', [])):
            chunk_data = {
                "chunk_id": i + 1,
                "relevance_score": chunk.get('relevance_score', 0.0),  # Primary score
                "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),  # NEW
                "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),  # NEW
                "content": chunk.get('content', ''),
                "metadata": {
                    "titel": chunk.get('metadata', {}).get('Artikeltitel', 'Kein Titel'),
                    "datum": chunk.get('metadata', {}).get('Datum', 'Unbekannt'),
                    "jahrgang": chunk.get('metadata', {}).get('Jahrgang', None),
                    "ausgabe": chunk.get('metadata', {}).get('Ausgabe', None),
                    "url": chunk.get('metadata', {}).get('URL', ''),
                    "autoren": chunk.get('metadata', {}).get('Autoren', ''),
                    "schlagworte": chunk.get('metadata', {}).get('Schlagworte', ''),
                    "untertitel": chunk.get('metadata', {}).get('Untertitel', ''),
                    "nr_in_issue": chunk.get('metadata', {}).get('nr_in_issue', None)
                },
                "scoring_analysis": {  # NEW: Enhanced scoring analysis
                    "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),
                    "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),
                    "score_difference": chunk.get('llm_evaluation_score', 0.0) - chunk.get('vector_similarity_score', 0.0),
                    "evaluation_text": chunk.get('metadata', {}).get('evaluation_text', ''),
                    "primary_selection_criterion": "llm_evaluation"  # Agent search prioritizes LLM scores
                }
            }
            
            # Add evaluation data if available
            if i < len(export_data["agent_evaluations"]):
                evaluation = export_data["agent_evaluations"][i]
                chunk_data["agent_evaluation"] = {
                    "llm_score": evaluation.get('llm_evaluation_score', 0.0),
                    "vector_score": evaluation.get('vector_similarity_score', 0.0),
                    "evaluation_text": evaluation.get('evaluation', ''),
                    "confidence": evaluation.get('confidence', 'medium')
                }
            
            export_data["chunks"].append(chunk_data)
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            prefix='spiegel_rag_agent_dual_scores_', 
            delete=False,  # Important: don't delete automatically
            encoding='utf-8'
        )
        
        # Write JSON content
        json.dump(export_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        
        logger.info(f"Created agent JSON download with dual scores and {len(export_data['chunks'])} chunks at {temp_file.name}")
        
        # Return the file path
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating agent JSON download: {e}")
        return None

def cleanup_temp_files():
    """
    Clean up old temporary files (optional utility function).
    You might want to call this periodically or on app shutdown.
    """
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith('spiegel_rag_') and (filename.endswith('.json') or filename.endswith('.csv')):
                file_path = os.path.join(temp_dir, filename)
                # Delete files older than 1 hour
                if os.path.getctime(file_path) < (datetime.now().timestamp() - 3600):
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old temp file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {filename}: {e}")
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")

def format_download_summary(chunks_count: int, format_type: str, has_dual_scores: bool = False) -> str:
    """
    Create a summary message for successful downloads.
    
    Args:
        chunks_count: Number of chunks exported
        format_type: Format type (JSON or CSV)
        has_dual_scores: Whether the export includes dual scores
        
    Returns:
        str: Formatted summary message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    score_info = ""
    if has_dual_scores:
        score_info = "\n\n**Enthaltene Scores**: Vector-Similarity, LLM-Evaluation und Score-Differenz für detaillierte Analyse."
    
    return f"""
    ### Download erfolgreich erstellt ({format_type.upper()})
    
    **Exportiert am**: {timestamp}  
    **Anzahl Texte**: {chunks_count}  
    **Format**: {format_type.upper()}
    {score_info}
    
    Die Datei enthält alle gefundenen Texte mit vollständigen Metadaten und Relevanz-Scores.
    """