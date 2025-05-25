# src/ui/handlers/download_handlers.py
"""
Fixed download handlers that properly create temporary files for Gradio downloads.
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
    
    Args:
        retrieved_chunks: Dictionary containing chunks and metadata from search
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            logger.warning("No chunks available for JSON download")
            return None
        
        # Prepare data for JSON export
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "source": "Der Spiegel RAG System",
                "total_chunks": len(retrieved_chunks.get('chunks', []))
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
            export_data["chunks"].append(chunk_data)
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spiegel_rag_results_{timestamp}.json"
        
        # Create temporary file with proper name
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            prefix='spiegel_rag_', 
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
    
    Args:
        retrieved_chunks: Dictionary containing chunks and metadata from search
        
    Returns:
        str: Path to the created temporary file, or None if no data
    """
    try:
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            logger.warning("No chunks available for CSV download")
            return None
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            prefix='spiegel_rag_', 
            delete=False,  # Important: don't delete automatically
            encoding='utf-8',
            newline=''  # Important for CSV files on Windows
        )
        
        # Create CSV writer
        writer = csv.writer(temp_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Write CSV header
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
        
        logger.info(f"Created CSV download with {len(retrieved_chunks.get('chunks', []))} chunks at {temp_file.name}")
        
        # Return the file path
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error creating CSV download: {e}")
        return None

def create_agent_download_json(agent_results: Optional[Dict[str, Any]]) -> str:
    """
    Create a JSON file for download containing agent search results with evaluations.
    
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
                "search_type": "agent_based"
            },
            "search_metadata": agent_results.get('metadata', {}),
            "agent_evaluations": agent_results.get('evaluations', []),
            "answer": agent_results.get('answer', ''),
            "chunks": []
        }
        
        # Process each chunk with agent evaluation data
        for i, chunk in enumerate(agent_results.get('chunks', [])):
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
            prefix='spiegel_rag_agent_', 
            delete=False,  # Important: don't delete automatically
            encoding='utf-8'
        )
        
        # Write JSON content
        json.dump(export_data, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        
        logger.info(f"Created agent JSON download with {len(export_data['chunks'])} chunks at {temp_file.name}")
        
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

def format_download_summary(chunks_count: int, format_type: str) -> str:
    """
    Create a summary message for successful downloads.
    
    Args:
        chunks_count: Number of chunks exported
        format_type: Format type (JSON or CSV)
        
    Returns:
        str: Formatted summary message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""
    ### Download erfolgreich erstellt ({format_type.upper()})
    
    **Exportiert am**: {timestamp}  
    **Anzahl Texte**: {chunks_count}  
    **Format**: {format_type.upper()}
    
    Die Datei enthält alle gefundenen Texte mit vollständigen Metadaten und Relevanz-Scores.
    """