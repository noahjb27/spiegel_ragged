# src/ui/handlers/keyword_handlers.py
"""
Handler functions for keyword analysis and expansion operations.
These functions are connected to UI events in the keyword analysis panel.
"""
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Global reference to the embedding service
# This will be initialized in the main app
embedding_service = None

def set_embedding_service(service: Any) -> None:
    """
    Set the global embedding service reference.
    
    Args:
        service: The WordEmbeddingService instance
    """
    global embedding_service
    embedding_service = service

def find_similar_words(keyword: str, expansion_factor: int) -> str:
    """
    Find similar words for a given keyword using FastText embeddings.
    
    Args:
        keyword: The keyword to find similar words for
        expansion_factor: Number of similar words to return
        
    Returns:
        Markdown formatted string with similar words
    """
    if not keyword.strip():
        return "Please enter a keyword."
    
    if not embedding_service:
        return "Embedding service not available."
    
    try:
        # Get similar words
        similar_words = embedding_service.find_similar_words(keyword.strip(), top_n=expansion_factor)
        
        # Format results for display
        if not similar_words:
            return f"No similar words found for '{keyword}'"
        
        result = f"### Similar words for '{keyword}':\n\n"
        for word_info in similar_words:
            result += f"- **{word_info['word']}** (similarity: {word_info['similarity']:.4f})\n"
        
        return result
    except Exception as e:
        logger.error(f"Error finding similar words: {e}")
        return f"Error finding similar words: {str(e)}"

def expand_boolean_expression(expression: str, expansion_factor: int) -> Tuple[str, str]:
    """
    Expand a boolean expression with semantically similar words.
    
    Args:
        expression: Boolean expression to expand
        expansion_factor: Number of similar words to find for each term
        
    Returns:
        Tuple of (display text, JSON string of expanded words)
    """
    if not expression.strip():
        return "Please enter a boolean expression.", ""
    
    if not embedding_service:
        return "Embedding service not available.", ""
    
    try:
        # Parse the boolean expression
        parsed_terms = embedding_service.parse_boolean_expression(expression)
        
        # Expand terms with semantically similar words
        expanded_terms = embedding_service.filter_by_semantic_similarity(
            parsed_terms, 
            expansion_factor=expansion_factor
        )
        
        # Format the expanded terms for display
        display_result = f"## Expanded Keywords\n\n"
        
        # Save expanded words for potential use in search
        expanded_words = {}
        
        for category, terms in expanded_terms.items():
            if terms:
                display_result += f"### {category.capitalize()} Terms:\n\n"
                for term_data in terms:
                    original = term_data.get('original', '')
                    expanded = term_data.get('expanded', {}).get(original, [])
                    
                    display_result += f"**{original}** → "
                    if expanded:
                        expanded_list = [f"{item['word']} ({item['similarity']:.2f})" for item in expanded]
                        display_result += ", ".join(expanded_list)
                        
                        # Add to expanded_words for search
                        if original not in expanded_words:
                            expanded_words[original] = []
                        for item in expanded:
                            expanded_words[original].append(item['word'])
                    else:
                        display_result += "No similar words found."
                    display_result += "\n\n"
        
        # Prepare a JSON structure to be used later in search
        encoded_expanded = json.dumps(expanded_words)
        
        return display_result, encoded_expanded
    except Exception as e:
        logger.error(f"Error expanding boolean expression: {e}")
        return f"Error expanding expression: {str(e)}", ""