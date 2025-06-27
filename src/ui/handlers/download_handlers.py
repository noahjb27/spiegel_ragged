# src/ui/handlers/download_handlers.py
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
    Args:
        retrieved_chunks: Dictionary containing chunks and metadata from search
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            logger.warning("No chunks available for JSON download")
            return None
        
        # Determine if this is LLM-assisted search with dual scores
        has_dual_scores = any(
            'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
            for chunk in retrieved_chunks.get('chunks', [])
        )
        
        # Prepare data for JSON export with updated terminology
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "source": "SPIEGEL RAG System - Heuristik", 
                "total_chunks": len(retrieved_chunks.get('chunks', [])),
                "has_dual_scores": has_dual_scores,
                "supports_chunk_selection": True,
                "search_method": retrieved_chunks.get('metadata', {}).get('retrieval_method', 'standard')
            },
            "heuristik_metadata": retrieved_chunks.get('metadata', {}),
            "chunks": []
        }
        
        # Process each chunk with updated field names and ranking
        for i, chunk in enumerate(retrieved_chunks.get('chunks', [])):
            chunk_data = {
                "chunk_id": i + 1,
                "rank": i + 1,  # Add ranking information
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
                    "zeit_interval": chunk.get('metadata', {}).get('time_window', None),  # UPDATED: from time_window
                    "interval_start": chunk.get('metadata', {}).get('window_start', None),
                    "interval_end": chunk.get('metadata', {}).get('window_end', None)
                }
            }
            
            # Add dual scores if available (for LLM-assisted search)
            if has_dual_scores:
                eval_text = chunk.get('metadata', {}).get('evaluation_text', '')
                reasoning = ""
                # Improved reasoning extraction
                if eval_text:
                    if '**Argumentation:**' in eval_text:
                        # Extract text after "Argumentation:" 
                        reasoning = eval_text.split('**Argumentation:**', 1)[1].strip()
                        # Remove any score information at the end
                        if 'Score:' in reasoning:
                            reasoning = reasoning.split('Score:')[0].strip()
                    elif '-' in eval_text:
                        # Original dash separator format
                        reasoning = eval_text.split('-', 1)[1].strip()
                    else:
                        # Use full evaluation text as fallback, but clean it up
                        reasoning = eval_text.strip()
                        # Remove common prefixes that aren't part of reasoning
                        if reasoning.startswith('Text ') and ':' in reasoning:
                            reasoning = reasoning.split(':', 1)[1].strip()
                
                chunk_data["scoring"] = {
                    "rank": i + 1,  # Include ranking in scoring section
                    "primary_relevance_score": chunk.get('relevance_score', 0.0),
                    "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),
                    "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),
                    "evaluation_text": eval_text,
                    "evaluation_reasoning": reasoning,  # Add separated reasoning
                    "score_type": "dual_scores_llm_assisted" if 'llm_evaluation_score' in chunk else "single_score_standard"  # UPDATED terminology
                }
            
            export_data["chunks"].append(chunk_data)
        
        # Create temporary file with proper UTF-8 encoding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_suffix = "_llm_assisted" if has_dual_scores else "_standard"  # UPDATED: from _dual_scores
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            prefix=f'spiegel_heuristik{filename_suffix}_',  # UPDATED: from spiegel_rag
            delete=False,
            encoding='utf-8'
        )
        
        # Write JSON content with proper Unicode handling
        json.dump(export_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        
        logger.info(f"Created JSON download with {len(export_data['chunks'])} chunks at {temp_file.name}")
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating JSON download: {e}")
        return None

def create_download_csv(retrieved_chunks: Optional[Dict[str, Any]]) -> str:
    """
    Create a CSV file for download with proper German text encoding.
    UPDATED: Enhanced with new terminology and better handling of LLM-assisted results.
    
    Args:
        retrieved_chunks: Dictionary containing chunks and metadata from search
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            logger.warning("No chunks available for CSV download")
            return None
        
        # Determine if this is LLM-assisted search with dual scores
        has_dual_scores = any(
            'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
            for chunk in retrieved_chunks.get('chunks', [])
        )
        
        # Create temporary file with UTF-8 BOM for Excel compatibility
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_suffix = "_llm_assisted" if has_dual_scores else "_standard"  # UPDATED terminology
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            prefix=f'spiegel_heuristik{filename_suffix}_',  # UPDATED: from spiegel_rag
            delete=False,
            encoding='utf-8-sig',  # UTF-8 with BOM for Excel compatibility
            newline=''
        )
        
        # Create CSV writer with proper quoting for German text
        writer = csv.writer(
            temp_file, 
            delimiter=',', 
            quotechar='"', 
            quoting=csv.QUOTE_ALL,
            lineterminator='\n'
        )
        
        # Write CSV header - Enhanced for dual scores and chunk selection
        if has_dual_scores:
            headers = [
                'chunk_id',
                'primary_relevance_score',
                'vector_similarity_score',
                'llm_evaluation_score',
                'score_difference',
                'titel',
                'datum',
                'jahrgang',
                'ausgabe',
                'autoren',
                'schlagworte',
                'untertitel',
                'url',
                'nr_in_issue',
                'zeit_interval', 
                'evaluation_text',
                'content_preview',
                'content_length',
                'full_content'
            ]
        else:
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
                'zeit_interval',  
                'content_preview',
                'content_length',
                'full_content'
            ]
        
        writer.writerow(headers)
        
        # Write data rows with proper text cleaning
        for i, chunk in enumerate(retrieved_chunks.get('chunks', [])):
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            
            # Better text cleaning for German characters
            def clean_text_for_csv(text: str) -> str:
                """Clean text for CSV export while preserving German characters."""
                if not text:
                    return ""
                
                # Replace problematic whitespace but keep German characters
                cleaned = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                
                # Replace multiple spaces with single space
                cleaned = ' '.join(cleaned.split())
                
                # Remove any remaining control characters but keep German umlauts
                import re
                cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)
                
                return cleaned.strip()
            
            # Create safe content preview (first 200 chars)
            content_preview = clean_text_for_csv(content)[:200]
            if len(content) > 200:
                content_preview += '...'
            
            # Clean full content
            clean_content = clean_text_for_csv(content)
            
            if has_dual_scores:
                # Enhanced row with dual scores for LLM-assisted search
                vector_score = chunk.get('vector_similarity_score', 0.0)
                llm_score = chunk.get('llm_evaluation_score', 0.0)
                primary_score = chunk.get('relevance_score', 0.0)
                score_difference = llm_score - vector_score if (llm_score and vector_score) else 0.0
                evaluation_text = clean_text_for_csv(metadata.get('evaluation_text', ''))
                
                row = [
                    i + 1,  # chunk_id
                    primary_score,
                    vector_score,
                    llm_score,
                    score_difference,
                    clean_text_for_csv(metadata.get('Artikeltitel', 'Kein Titel')),
                    clean_text_for_csv(metadata.get('Datum', 'Unbekannt')),
                    metadata.get('Jahrgang', ''),
                    metadata.get('Ausgabe', ''),
                    clean_text_for_csv(metadata.get('Autoren', '')),
                    clean_text_for_csv(metadata.get('Schlagworte', '')),
                    clean_text_for_csv(metadata.get('Untertitel', '')),
                    metadata.get('URL', ''),
                    metadata.get('nr_in_issue', ''),
                    metadata.get('time_window', ''),  # Maps to zeit_interval
                    evaluation_text,
                    content_preview,
                    len(content),
                    clean_content
                ]
            else:
                # Standard row for non-LLM-assisted search
                row = [
                    i + 1,  # chunk_id
                    chunk.get('relevance_score', 0.0),
                    clean_text_for_csv(metadata.get('Artikeltitel', 'Kein Titel')),
                    clean_text_for_csv(metadata.get('Datum', 'Unbekannt')),
                    metadata.get('Jahrgang', ''),
                    metadata.get('Ausgabe', ''),
                    clean_text_for_csv(metadata.get('Autoren', '')),
                    clean_text_for_csv(metadata.get('Schlagworte', '')),
                    clean_text_for_csv(metadata.get('Untertitel', '')),
                    metadata.get('URL', ''),
                    metadata.get('nr_in_issue', ''),
                    metadata.get('time_window', ''),  # Maps to zeit_interval
                    content_preview,
                    len(content),
                    clean_content
                ]
            
            writer.writerow(row)
        
        temp_file.close()
        
        score_info = "with LLM evaluation scores" if has_dual_scores else "with similarity scores"  # UPDATED
        logger.info(f"Created CSV download {score_info} with proper German encoding and {len(retrieved_chunks.get('chunks', []))} chunks at {temp_file.name}")
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating CSV download: {e}")
        return None

def create_llm_assisted_download_json(llm_assisted_results: Optional[Dict[str, Any]]) -> str:
    """
    Create a JSON file for download containing LLM-assisted search results with evaluations.
    UPDATED: Renamed from create_agent_download_json with new terminology.
    
    Args:
        llm_assisted_results: Dictionary containing LLM-assisted search results and evaluations
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not llm_assisted_results or not llm_assisted_results.get('chunks'):
            logger.warning("No LLM-assisted results available for JSON download")
            return None
        
        # Prepare data for JSON export with updated terminology
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "source": "SPIEGEL RAG System - LLM-Unterstützte Auswahl",  # UPDATED terminology
                "total_chunks": len(llm_assisted_results.get('chunks', [])),
                "search_type": "llm_assisted_selection",  # UPDATED: from agent_based
                "includes_dual_scores": True,
                "supports_chunk_selection": True
            },
            "heuristik_metadata": llm_assisted_results.get('metadata', {}),  # UPDATED: from search_metadata
            "llm_evaluations": llm_assisted_results.get('evaluations', []),  # UPDATED: from agent_evaluations
            "answer": llm_assisted_results.get('answer', ''),
            "chunks": []
        }
        
        # Process each chunk with LLM evaluation data and dual scores
        for i, chunk in enumerate(llm_assisted_results.get('chunks', [])):
            chunk_data = {
                "chunk_id": i + 1,
                "relevance_score": chunk.get('relevance_score', 0.0),
                "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),
                "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),
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
                "scoring_analysis": {
                    "vector_similarity_score": chunk.get('vector_similarity_score', 0.0),
                    "llm_evaluation_score": chunk.get('llm_evaluation_score', 0.0),
                    "score_difference": chunk.get('llm_evaluation_score', 0.0) - chunk.get('vector_similarity_score', 0.0),
                    "evaluation_text": chunk.get('metadata', {}).get('evaluation_text', ''),
                    "primary_selection_criterion": "llm_evaluation"
                }
            }
            
            # Add evaluation data if available
            if i < len(export_data["llm_evaluations"]):
                evaluation = export_data["llm_evaluations"][i]
                chunk_data["llm_evaluation_details"] = {  # UPDATED: from agent_evaluation
                    "llm_score": evaluation.get('llm_evaluation_score', 0.0),
                    "vector_score": evaluation.get('vector_similarity_score', 0.0),
                    "evaluation_text": evaluation.get('evaluation', ''),
                    "confidence": evaluation.get('confidence', 'medium')
                }
            
            export_data["chunks"].append(chunk_data)
        
        # Create temporary file with proper UTF-8 encoding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            prefix='spiegel_llm_assisted_comprehensive_',  # UPDATED: from spiegel_rag_agent
            delete=False,
            encoding='utf-8'
        )
        
        # Write JSON content with proper Unicode handling
        json.dump(export_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        
        logger.info(f"Created LLM-assisted JSON download with dual scores and {len(export_data['chunks'])} chunks at {temp_file.name}")
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating LLM-assisted JSON download: {e}")
        return None

def cleanup_temp_files():
    """
    Clean up old temporary files (optional utility function).
    You might want to call this periodically or on app shutdown.
    """
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            # UPDATED: Look for files with new naming convention
            if (filename.startswith('spiegel_heuristik_') or 
                filename.startswith('spiegel_llm_assisted_') or 
                filename.startswith('spiegel_analysis_')) and (filename.endswith('.json') or filename.endswith('.csv') or filename.endswith('.txt')):
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
    UPDATED: Enhanced with new terminology.
    
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
        score_info = "\n\n**Enthaltene Scores**: Vector-Similarity und LLM-Bewertung für detaillierte Analyse der Quellenselektion."
    
    encoding_info = ""
    if format_type.upper() == "CSV":
        encoding_info = "\n\n**Encoding**: UTF-8 mit BOM für optimale Kompatibilität mit Excel und deutschen Umlauten."
    
    # UPDATED: Enhanced terminology and information
    search_method = "LLM-Unterstützte Auswahl" if has_dual_scores else "Standard-Heuristik"
    
    return f"""
    ### Download erfolgreich erstellt ({format_type.upper()})
    
    **Exportiert am**: {timestamp}  
    **Anzahl Texte**: {chunks_count}  
    **Format**: {format_type.upper()}  
    **Suchmethode**: {search_method}
    {score_info}
    {encoding_info}
    
    Die Datei enthält alle gefundenen Texte mit vollständigen Metadaten und Relevanz-Scores.
    **Chunk-IDs können für selektive Analyse verwendet werden.**
    
    **Hinweis**: Diese Ergebnisse stammen aus der Heuristik-Phase und können für die nachfolgende Analyse verwendet werden.
    """

def create_analysis_txt_download(answer_text: str, metadata_text: str, user_prompt: str = "", selected_chunks_info: str = "") -> str:
    """
    Create a TXT file for download containing analysis results.
    NEW: Enhanced analysis download with complete information.
    
    Args:
        answer_text: The generated analysis
        metadata_text: Analysis metadata
        user_prompt: The user's research question
        selected_chunks_info: Information about selected chunks
        
    Returns:
        str: Path to the created temporary file, or None if error
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create comprehensive TXT content
        txt_content = f"""SPIEGEL RAG System - Analyse-Ergebnisse
{'='*60}
Erstellt am: {timestamp}

{'='*60}
FORSCHUNGSFRAGE
{'='*60}

{user_prompt if user_prompt else 'Nicht angegeben'}

{'='*60}
ANALYSE-ERGEBNIS
{'='*60}

{answer_text}

{'='*60}
QUELLENAUSWAHL
{'='*60}

{selected_chunks_info if selected_chunks_info else 'Alle verfügbaren Quellen verwendet'}

{'='*60}
METADATEN & KONFIGURATION
{'='*60}

{metadata_text}

{'='*60}
SYSTEM-INFORMATION
{'='*60}

- Generiert von: SPIEGEL RAG System
- Zeitstempel: {timestamp}
- Methodik: Retrieval-Augmented Generation
- Quellenbasis: Der Spiegel Archiv (1948-1979)

Ende der Analyse
{'='*60}
"""
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.txt', 
            prefix='spiegel_analyse_', 
            delete=False,
            encoding='utf-8'
        )
        
        temp_file.write(txt_content)
        temp_file.close()
        
        logger.info(f"Created analysis TXT download at {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating analysis TXT download: {e}")
        return None