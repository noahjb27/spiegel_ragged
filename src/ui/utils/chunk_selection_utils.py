# src/utils/chunk_selection_utils.py
"""
Utilities for processing chunk selection files and managing chunk filtering.
"""
import json
import csv
import os
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class ChunkSelectionProcessor:
    """Handles processing of chunk selection files and validation."""
    
    @staticmethod
    def process_uploaded_file(file_path: str) -> Tuple[bool, List[int], str]:
        """
        Process an uploaded chunk selection file (CSV or JSON).
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Tuple of (success, chunk_ids, message)
        """
        if not os.path.exists(file_path):
            return False, [], "Datei nicht gefunden."
        
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.json':
                return ChunkSelectionProcessor._process_json_file(file_path)
            elif file_extension == '.csv':
                return ChunkSelectionProcessor._process_csv_file(file_path)
            else:
                return False, [], f"Nicht unterstütztes Dateiformat: {file_extension}"
                
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return False, [], f"Fehler beim Verarbeiten der Datei: {str(e)}"
    
    @staticmethod
    def _process_json_file(file_path: str) -> Tuple[bool, List[int], str]:
        """Process a JSON chunk selection file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunk_ids = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Simple list of IDs: [1, 2, 3, 4]
                chunk_ids = [int(x) for x in data if str(x).isdigit()]
                
            elif isinstance(data, dict):
                # Dictionary with chunk_ids key: {"chunk_ids": [1, 2, 3]}
                if 'chunk_ids' in data:
                    chunk_ids = [int(x) for x in data['chunk_ids'] if str(x).isdigit()]
                
                # Dictionary with chunks array: {"chunks": [{"chunk_id": 1}, {"chunk_id": 2}]}
                elif 'chunks' in data and isinstance(data['chunks'], list):
                    for chunk in data['chunks']:
                        if isinstance(chunk, dict):
                            if 'chunk_id' in chunk:
                                chunk_ids.append(int(chunk['chunk_id']))
                            elif 'id' in chunk:
                                chunk_ids.append(int(chunk['id']))
                
                # Export format from our own system
                elif 'export_info' in data and 'chunks' in data:
                    for chunk in data['chunks']:
                        if isinstance(chunk, dict) and 'chunk_id' in chunk:
                            chunk_ids.append(int(chunk['chunk_id']))
            
            if not chunk_ids:
                return False, [], "Keine gültigen Chunk-IDs in der JSON-Datei gefunden."
            
            # Remove duplicates and sort
            chunk_ids = sorted(list(set(chunk_ids)))
            
            message = f"✅ {len(chunk_ids)} Chunk-IDs aus JSON-Datei geladen."
            return True, chunk_ids, message
            
        except json.JSONDecodeError as e:
            return False, [], f"Ungültiges JSON-Format: {str(e)}"
        except Exception as e:
            return False, [], f"Fehler beim Lesen der JSON-Datei: {str(e)}"
    
    @staticmethod
    def _process_csv_file(file_path: str) -> Tuple[bool, List[int], str]:
        """Process a CSV chunk selection file."""
        try:
            chunk_ids = []
            
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252']
            data_read = False
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        csv_reader = csv.reader(f)
                        rows = list(csv_reader)
                        data_read = True
                        break
                except UnicodeDecodeError:
                    continue
            
            if not data_read:
                return False, [], "Konnte CSV-Datei mit keiner unterstützten Kodierung lesen."
            
            if not rows:
                return False, [], "CSV-Datei ist leer."
            
            # Detect header and chunk_id column
            headers = rows[0] if rows else []
            chunk_id_col = None
            include_col = None
            
            # Look for chunk_id column
            for i, header in enumerate(headers):
                header_lower = header.lower().strip()
                if 'chunk_id' in header_lower or header_lower == 'id':
                    chunk_id_col = i
                elif 'include' in header_lower or 'selected' in header_lower or 'use' in header_lower:
                    include_col = i
            
            # If no chunk_id column found, assume first column
            if chunk_id_col is None:
                chunk_id_col = 0
            
            # Process data rows
            data_rows = rows[1:] if len(rows) > 1 and any(h.strip() for h in headers) else rows
            
            for row_idx, row in enumerate(data_rows):
                try:
                    if len(row) <= chunk_id_col:
                        continue
                    
                    chunk_id_str = row[chunk_id_col].strip()
                    
                    # Skip empty cells
                    if not chunk_id_str:
                        continue
                    
                    # Check if this row should be included (if include column exists)
                    if include_col is not None and len(row) > include_col:
                        include_value = row[include_col].strip().lower()
                        if include_value in ['no', 'false', '0', 'n', 'nein']:
                            continue
                    
                    # Try to parse chunk_id
                    if chunk_id_str.isdigit():
                        chunk_ids.append(int(chunk_id_str))
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping row {row_idx + 1}: {e}")
                    continue
            
            if not chunk_ids:
                return False, [], "Keine gültigen Chunk-IDs in der CSV-Datei gefunden."
            
            # Remove duplicates and sort
            chunk_ids = sorted(list(set(chunk_ids)))
            
            message = f"✅ {len(chunk_ids)} Chunk-IDs aus CSV-Datei geladen."
            return True, chunk_ids, message
            
        except Exception as e:
            return False, [], f"Fehler beim Lesen der CSV-Datei: {str(e)}"
    
    @staticmethod
    def parse_manual_ids(ids_text: str) -> Tuple[bool, List[int], str]:
        """
        Parse manually entered chunk IDs.
        
        Args:
            ids_text: Text containing chunk IDs (comma-separated, ranges supported)
            
        Returns:
            Tuple of (success, chunk_ids, message)
        """
        if not ids_text.strip():
            return False, [], "Keine IDs eingegeben."
        
        try:
            chunk_ids = []
            
            # Split by comma and process each part
            for part in ids_text.split(','):
                part = part.strip()
                
                if not part:
                    continue
                
                # Handle simple numbers
                if part.isdigit():
                    chunk_ids.append(int(part))
                
                # Handle ranges like "1-5"
                elif '-' in part and len(part.split('-')) == 2:
                    try:
                        start_str, end_str = part.split('-')
                        start_id = int(start_str.strip())
                        end_id = int(end_str.strip())
                        
                        if start_id <= end_id:
                            chunk_ids.extend(range(start_id, end_id + 1))
                        else:
                            return False, [], f"Ungültiger Bereich: {part} (Start > Ende)"
                    except ValueError:
                        return False, [], f"Ungültiger Bereich: {part}"
                
                # Handle space-separated numbers in a single part
                elif ' ' in part:
                    for sub_part in part.split():
                        if sub_part.strip().isdigit():
                            chunk_ids.append(int(sub_part.strip()))
                
                else:
                    logger.warning(f"Skipping invalid part: {part}")
            
            if not chunk_ids:
                return False, [], "Keine gültigen Chunk-IDs gefunden."
            
            # Remove duplicates and sort
            chunk_ids = sorted(list(set(chunk_ids)))
            
            message = f"✅ {len(chunk_ids)} Chunk-IDs verarbeitet: {', '.join(map(str, chunk_ids[:10]))}" + ("..." if len(chunk_ids) > 10 else "")
            return True, chunk_ids, message
            
        except Exception as e:
            return False, [], f"Fehler beim Verarbeiten der IDs: {str(e)}"
    
    @staticmethod
    def validate_chunk_ids(chunk_ids: List[int], max_chunks: int) -> Tuple[List[int], List[int], str]:
        """
        Validate chunk IDs against available chunks.
        
        Args:
            chunk_ids: List of requested chunk IDs
            max_chunks: Maximum number of available chunks
            
        Returns:
            Tuple of (valid_ids, invalid_ids, message)
        """
        if not chunk_ids:
            return [], [], "Keine Chunk-IDs zu validieren."
        
        valid_ids = []
        invalid_ids = []
        
        for chunk_id in chunk_ids:
            if 1 <= chunk_id <= max_chunks:
                valid_ids.append(chunk_id)
            else:
                invalid_ids.append(chunk_id)
        
        if invalid_ids:
            message = f"⚠️ {len(invalid_ids)} ungültige IDs gefunden (gültig: 1-{max_chunks}): {invalid_ids[:5]}"
            if len(invalid_ids) > 5:
                message += "..."
        else:
            message = f"✅ Alle {len(valid_ids)} Chunk-IDs sind gültig."
        
        return valid_ids, invalid_ids, message
    
    @staticmethod
    def filter_chunks_by_ids(chunks: List[Dict], chunk_ids: List[int]) -> List[Dict]:
        """
        Filter chunks list by selected chunk IDs.
        
        Args:
            chunks: List of chunk dictionaries
            chunk_ids: List of chunk IDs to select (1-based)
            
        Returns:
            Filtered list of chunks
        """
        if not chunk_ids:
            return chunks
        
        filtered_chunks = []
        
        for chunk_id in chunk_ids:
            # Convert to 0-based index
            index = chunk_id - 1
            if 0 <= index < len(chunks):
                chunk = chunks[index].copy()
                chunk['selected_chunk_id'] = chunk_id  # Add selection metadata
                filtered_chunks.append(chunk)
            else:
                logger.warning(f"Chunk ID {chunk_id} out of range (1-{len(chunks)})")
        
        return filtered_chunks
    
    @staticmethod
    def create_selection_summary(chunk_ids: List[int], total_chunks: int) -> str:
        """
        Create a summary of the chunk selection.
        
        Args:
            chunk_ids: Selected chunk IDs
            total_chunks: Total number of available chunks
            
        Returns:
            Summary string
        """
        if not chunk_ids:
            return f"**Aktuelle Auswahl**: Alle {total_chunks} gefundenen Quellen werden verwendet."
        
        percentage = (len(chunk_ids) / total_chunks) * 100 if total_chunks > 0 else 0
        
        id_preview = ', '.join(map(str, chunk_ids[:10]))
        if len(chunk_ids) > 10:
            id_preview += f" ... (+{len(chunk_ids) - 10} weitere)"
        
        return f"""**Aktuelle Auswahl**: {len(chunk_ids)} von {total_chunks} Quellen ausgewählt ({percentage:.1f}%)
**Ausgewählte IDs**: {id_preview}"""


def create_chunk_selection_template() -> Dict[str, Any]:
    """
    Create a template structure for chunk selection files.
    
    Returns:
        Dictionary with template data
    """
    template = {
        "export_info": {
            "description": "Chunk Selection Template",
            "format": "template",
            "instructions": [
                "Bearbeiten Sie die 'chunk_ids' Liste mit den gewünschten Text-IDs",
                "IDs beziehen sich auf die Reihenfolge in den Suchergebnissen (1, 2, 3, ...)",
                "Entfernen Sie IDs, die Sie nicht verwenden möchten",
                "Fügen Sie neue IDs hinzu, falls gewünscht"
            ]
        },
        "chunk_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "example_ranges": {
            "description": "Sie können auch Bereiche verwenden",
            "examples": ["1-5 für IDs 1 bis 5", "10,15,20 für spezifische IDs"]
        }
    }
    
    return template