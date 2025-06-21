# src/ui/handlers/keyword_handlers.py - FIXED: Boolean expression expansion
"""
FIXED: Handler functions for keyword analysis with working boolean expansion.
- Fixed the logic error in expand_boolean_expression
- Simplified the code structure
- Better error handling
"""
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Global reference to the embedding service
embedding_service = None

def set_embedding_service(service: Any) -> None:
    """Set the global embedding service reference."""
    global embedding_service
    embedding_service = service

def find_similar_words(keyword: str, expansion_factor: int) -> str:
    """Find similar words for a given keyword using FastText embeddings."""
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
            
            freq_display = f"{frequency:,}" if frequency > 0 else "0"
            result += f"| **{word}** | {similarity:.4f} | {freq_display} |\n"
        
        # Add summary statistics
        total_frequency = sum(word_info.get('frequency', 0) for word_info in similar_words)
        avg_frequency = total_frequency / len(similar_words) if similar_words else 0
        
        result += f"\n**Zusammenfassung**:\n"
        result += f"- Gefundene ähnliche Wörter: {len(similar_words)}\n"
        result += f"- Durchschnittliche Häufigkeit: {avg_frequency:,.1f}\n"
        
        if similar_words:
            most_frequent = max(similar_words, key=lambda x: x.get('frequency', 0))
            result += f"- Häufigster Begriff: {most_frequent['word']} ({most_frequent.get('frequency', 0):,} mal)\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error finding similar words: {e}")
        return f"Fehler bei der Suche nach ähnlichen Wörtern: {str(e)}"

def expand_boolean_expression(expression: str, expansion_factor: int) -> Tuple[str, str]:
    """
    FIXED: Expand a boolean expression with semantically similar words.
    The original logic was incorrect - fixed the iteration over expanded terms.
    """
    if not expression.strip():
        return "Bitte geben Sie einen booleschen Ausdruck ein.", ""
    
    if not embedding_service:
        return "Embedding-Service nicht verfügbar.", ""
    
    try:
        # SIMPLIFIED: Direct approach to parsing and expanding
        # Split the expression into individual terms (ignoring operators for now)
        import re
        
        # Extract words (ignore AND, OR, NOT, parentheses)
        terms = re.findall(r'\b(?!AND|OR|NOT\b)[a-zA-ZäöüÄÖÜß]+\b', expression, re.IGNORECASE)
        
        if not terms:
            return "Keine gültigen Begriffe im Ausdruck gefunden.", ""
        
        display_result = f"## Erweiterte Schlagwörter mit Korpus-Häufigkeiten\n\n"
        display_result += f"**Original-Ausdruck**: `{expression}`\n\n"
        
        # Save expanded words for potential use in search
        expanded_words = {}
        
        # FIXED: Simplified logic - process each unique term
        for term in set(terms):  # Remove duplicates
            term_clean = term.lower().strip()
            
            # Get frequency for original term
            original_frequency = embedding_service.get_word_frequency(term_clean)
            
            display_result += f"**{term}** (Häufigkeit: {original_frequency:,})\n\n"
            
            # Get similar words
            similar_words = embedding_service.find_similar_words(term_clean, top_n=expansion_factor)
            
            if similar_words:
                # Create table for similar words with frequencies
                display_result += "| Ähnliches Wort | Ähnlichkeit | Häufigkeit |\n"
                display_result += "|----------------|-------------|------------|\n"
                
                expanded_for_term = []
                for word_info in similar_words:
                    word = word_info['word']
                    similarity = word_info['similarity']
                    frequency = word_info.get('frequency', 0)
                    
                    freq_display = f"{frequency:,}" if frequency > 0 else "0"
                    display_result += f"| {word} | {similarity:.3f} | {freq_display} |\n"
                    
                    expanded_for_term.append(word)
                
                display_result += "\n"
                
                # Store for search use
                expanded_words[term] = expanded_for_term
            else:
                display_result += "Keine ähnlichen Wörter gefunden.\n\n"
        
        # FIXED: Create proper JSON structure
        encoded_expanded = json.dumps(expanded_words, ensure_ascii=False)
        
        return display_result, encoded_expanded
        
    except Exception as e:
        logger.error(f"Error expanding boolean expression: {e}")
        return f"Fehler bei der Erweiterung des Ausdrucks: {str(e)}", ""

def analyze_term_frequencies(terms: List[str]) -> str:
    """Analyze frequency patterns for a list of terms."""
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
    """Get general statistics about the corpus."""
    if not embedding_service:
        return "Embedding-Service nicht verfügbar."
    
    try:
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