# src/ui/handlers/keyword_handlers.py - Updated to show word frequencies
"""
Updated handler functions for keyword analysis with frequency information.
Shows both similarity scores and corpus frequency for better keyword understanding.
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
    UPDATED: Now includes frequency information from the corpus.
    
    Args:
        keyword: The keyword to find similar words for
        expansion_factor: Number of similar words to return
        
    Returns:
        Markdown formatted string with similar words and frequencies
    """
    if not keyword.strip():
        return "Bitte geben Sie ein Schlagwort ein."
    
    if not embedding_service:
        return "Embedding-Service nicht verfügbar."
    
    try:
        keyword_clean = keyword.strip().lower()
        
        # Get frequency for the input word
        input_word_frequency = embedding_service.get_word_frequency(keyword_clean)
        
        # Get similar words with their frequencies
        similar_words = embedding_service.find_similar_words(keyword_clean, top_n=expansion_factor)
        
        # Format results for display with frequency information
        if not similar_words:
            result = f"### Keine ähnlichen Wörter für '{keyword}' gefunden\n\n"
            if input_word_frequency > 0:
                result += f"**Häufigkeit von '{keyword}'**: {input_word_frequency:,} mal im Korpus\n"
            else:
                result += f"**'{keyword}'** kommt nicht im Korpus vor.\n"
            return result
        
        # UPDATED: Enhanced formatting with frequency information
        result = f"### Ähnliche Wörter für '{keyword}':\n\n"
        
        # Show input word frequency
        if input_word_frequency > 0:
            result += f"**Eingabewort '{keyword}'**: {input_word_frequency:,} mal im Korpus\n\n"
        else:
            result += f"**Eingabewort '{keyword}'**: Nicht im Korpus gefunden\n\n"
        
        result += "**Ähnliche Begriffe** (sortiert nach Ähnlichkeit):\n\n"
        result += "| Wort | Ähnlichkeit | Korpus-Häufigkeit |\n"
        result += "|------|-------------|-------------------|\n"
        
        for word_info in similar_words:
            word = word_info['word']
            similarity = word_info['similarity']
            frequency = word_info.get('frequency', 0)
            
            # Format frequency with thousands separator
            freq_display = f"{frequency:,}" if frequency > 0 else "0"
            
            result += f"| **{word}** | {similarity:.4f} | {freq_display} |\n"
        
        # Add summary statistics
        total_frequency = sum(word_info.get('frequency', 0) for word_info in similar_words)
        avg_frequency = total_frequency / len(similar_words) if similar_words else 0
        
        result += f"\n**Zusammenfassung**:\n"
        result += f"- Gefundene ähnliche Wörter: {len(similar_words)}\n"
        result += f"- Durchschnittliche Häufigkeit: {avg_frequency:,.1f}\n"
        result += f"- Häufigster Begriff: {max(similar_words, key=lambda x: x.get('frequency', 0))['word']} ({max(similar_words, key=lambda x: x.get('frequency', 0)).get('frequency', 0):,} mal)\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error finding similar words: {e}")
        return f"Fehler bei der Suche nach ähnlichen Wörtern: {str(e)}"

def expand_boolean_expression(expression: str, expansion_factor: int) -> Tuple[str, str]:
    """
    Expand a boolean expression with semantically similar words.
    UPDATED: Now includes frequency information for better term selection.
    
    Args:
        expression: Boolean expression to expand
        expansion_factor: Number of similar words to find for each term
        
    Returns:
        Tuple of (display text with frequencies, JSON string of expanded words)
    """
    if not expression.strip():
        return "Bitte geben Sie einen booleschen Ausdruck ein.", ""
    
    if not embedding_service:
        return "Embedding-Service nicht verfügbar.", ""
    
    try:
        # Parse the boolean expression
        parsed_terms = embedding_service.parse_boolean_expression(expression)
        
        # Expand terms with semantically similar words
        expanded_terms = embedding_service.filter_by_semantic_similarity(
            parsed_terms, 
            expansion_factor=expansion_factor
        )
        
        # Format the expanded terms for display with frequency information
        display_result = f"## Erweiterte Schlagwörter mit Korpus-Häufigkeiten\n\n"
        
        # Save expanded words for potential use in search
        expanded_words = {}
        
        for category, terms in expanded_terms.items():
            if terms:
                display_result += f"### {category.capitalize()} Begriffe:\n\n"
                for term_data in terms:
                    original = term_data.get('original', '')
                    expanded = term_data.get('expanded', {}).get(original, [])
                    
                    # Get frequency for original term
                    original_frequency = embedding_service.get_word_frequency(original.lower())
                    
                    display_result += f"**{original}** (Häufigkeit: {original_frequency:,})\n\n"
                    
                    if expanded:
                        # Create table for similar words with frequencies
                        display_result += "| Ähnliches Wort | Ähnlichkeit | Häufigkeit |\n"
                        display_result += "|----------------|-------------|------------|\n"
                        
                        for item in expanded:
                            word = item['word']
                            similarity = item['similarity']
                            
                            # Get frequency for each similar word
                            frequency = embedding_service.get_word_frequency(word.lower())
                            freq_display = f"{frequency:,}" if frequency > 0 else "0"
                            
                            display_result += f"| {word} | {similarity:.3f} | {freq_display} |\n"
                        
                        display_result += "\n"
                        
                        # Add to expanded_words for search
                        if original not in expanded_words:
                            expanded_words[original] = []
                        for item in expanded:
                            expanded_words[original].append(item['word'])
                    else:
                        display_result += "Keine ähnlichen Wörter gefunden.\n\n"
        
        # Add overall statistics
        if expanded_words:
            total_original_terms = len(expanded_words)
            total_expanded_terms = sum(len(words) for words in expanded_words.values())
            
            display_result += f"## Zusammenfassung\n\n"
            display_result += f"- **Original-Begriffe**: {total_original_terms}\n"
            display_result += f"- **Erweiterte Begriffe**: {total_expanded_terms}\n"
            display_result += f"- **Erweiterungsfaktor**: {expansion_factor} pro Begriff\n"
            
            # Show frequency range
            all_frequencies = []
            for original, words in expanded_words.items():
                for word in words:
                    freq = embedding_service.get_word_frequency(word.lower())
                    if freq > 0:
                        all_frequencies.append(freq)
            
            if all_frequencies:
                display_result += f"- **Häufigkeitsbereich**: {min(all_frequencies):,} - {max(all_frequencies):,}\n"
                display_result += f"- **Durchschnittshäufigkeit**: {sum(all_frequencies)/len(all_frequencies):,.1f}\n"
        
        # Prepare a JSON structure to be used later in search
        encoded_expanded = json.dumps(expanded_words)
        
        return display_result, encoded_expanded
        
    except Exception as e:
        logger.error(f"Error expanding boolean expression: {e}")
        return f"Fehler bei der Erweiterung des Ausdrucks: {str(e)}", ""

def analyze_term_frequencies(terms: List[str]) -> str:
    """
    Analyze frequency patterns for a list of terms.
    NEW: Helper function for frequency analysis.
    
    Args:
        terms: List of terms to analyze
        
    Returns:
        Formatted analysis of term frequencies
    """
    if not embedding_service or not terms:
        return "Keine Begriffe zur Analyse verfügbar."
    
    try:
        term_frequencies = []
        
        for term in terms:
            frequency = embedding_service.get_word_frequency(term.lower().strip())
            term_frequencies.append((term, frequency))
        
        # Sort by frequency (descending)
        term_frequencies.sort(key=lambda x: x[1], reverse=True)
        
        result = "### Häufigkeitsanalyse\n\n"
        result += "| Begriff | Häufigkeit | Anteil |\n"
        result += "|---------|------------|--------|\n"
        
        total_frequency = sum(freq for _, freq in term_frequencies)
        
        for term, frequency in term_frequencies:
            percentage = (frequency / total_frequency * 100) if total_frequency > 0 else 0
            freq_display = f"{frequency:,}" if frequency > 0 else "0"
            result += f"| {term} | {freq_display} | {percentage:.1f}% |\n"
        
        result += f"\n**Gesamthäufigkeit**: {total_frequency:,}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing term frequencies: {e}")
        return f"Fehler bei der Häufigkeitsanalyse: {str(e)}"

def get_corpus_statistics() -> str:
    """
    Get general statistics about the corpus.
    NEW: Helper function for corpus information.
    
    Returns:
        Formatted corpus statistics
    """
    if not embedding_service:
        return "Embedding-Service nicht verfügbar."
    
    try:
        # This would need to be implemented in the embedding service
        # For now, return basic information
        result = "### Korpus-Informationen\n\n"
        result += "- **Zeitraum**: 1948-1979 (Der Spiegel)\n"
        result += "- **Modell**: FastText Word Embeddings\n"
        result += "- **Verfügbare Funktionen**: Wortähnlichkeit, Häufigkeitsanalyse\n"
        
        # Try to get vocabulary size if available
        try:
            vocab_size = len(embedding_service.model.wv) if embedding_service.model else 0
            result += f"- **Vokabular-Größe**: {vocab_size:,} Wörter\n"
        except:
            result += "- **Vokabular-Größe**: Nicht verfügbar\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting corpus statistics: {e}")
        return f"Fehler beim Abrufen der Korpus-Statistiken: {str(e)}"