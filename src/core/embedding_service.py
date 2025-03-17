"""
Enhanced word embedding service using FastText models.
Provides word filtering, semantic search enhancements, and word frequency analysis.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from gensim.models import FastText

from src.config import settings

logger = logging.getLogger(__name__)

class WordEmbeddingService:
    """Enhanced service for word embedding operations."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize word embedding service.
        
        Args:
            model_path: Path to FastText model. Defaults to settings value.
        """
        self.model_path = model_path or settings.WORD_EMBEDDING_MODEL_PATH
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the FastText model."""
        try:
            self.model = FastText.load(self.model_path)
            logger.info(f"Loaded FastText model from {self.model_path} with {len(self.model.wv)} words")
        except Exception as e:
            logger.error(f"Failed to load FastText model: {e}")
            self.model = None
    
    def find_similar_words(self, word: str, top_n: int = 10) -> List[Dict[str, Union[str, float, int]]]:
        """
        Find similar words based on embeddings.
        
        Args:
            word: Query word
            top_n: Number of similar words to return
            
        Returns:
            List of similar words with similarity scores and frequencies
        """
        if not self.model:
            logger.error("FastText model not loaded")
            return []
        
        word = word.lower()  # Convert to lowercase
        
        try:
            similar_words = self.model.wv.most_similar(positive=[word], topn=top_n)
            result = []
            
            for similar_word, similarity in similar_words:
                try:
                    frequency = self.model.wv.get_vecattr(similar_word, "count")
                except (KeyError, AttributeError):
                    frequency = 0
                
                result.append({
                    "word": similar_word,
                    "similarity": similarity,
                    "frequency": frequency
                })
            
            logger.info(f"Found {len(result)} similar words for '{word}'")
            return result
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary")
            return []
        except Exception as e:
            logger.error(f"Error finding similar words: {e}")
            return []
    
    def get_word_frequency(self, word: str) -> int:
        """
        Get frequency of a word in the training corpus.
        
        Args:
            word: The word to check
            
        Returns:
            Frequency count (0 if word not found)
        """
        if not self.model:
            logger.error("FastText model not loaded")
            return 0
        
        word = word.lower()  # Convert to lowercase
        
        try:
            return self.model.wv.get_vecattr(word, "count")
        except (KeyError, AttributeError):
            return 0
    
    def get_word_vector(self, word: str) -> Optional[List[float]]:
        """
        Get the vector representation of a word.
        
        Args:
            word: The word to get vector for
            
        Returns:
            Vector as list of floats, or None if word not found
        """
        if not self.model:
            logger.error("FastText model not loaded")
            return None
        
        word = word.lower()  # Convert to lowercase
        
        try:
            # Convert numpy array to list for JSON serialization
            return self.model.wv[word].tolist()
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary")
            return None
        except Exception as e:
            logger.error(f"Error getting word vector: {e}")
            return None
            
    def parse_boolean_expression(self, expression: str) -> Dict[str, List[str]]:
        """
        Parse a boolean search expression into components.
        
        Args:
            expression: Boolean search expression (e.g., "berlin AND (wall OR mauer) NOT soviet")
            
        Returns:
            Dictionary with 'must', 'should', and 'must_not' terms
        """
        if not expression or not expression.strip():
            return {"must": [], "should": [], "must_not": []}
            
        # Simple parsing for: 
        # - Terms with AND are 'must'
        # - Terms with OR are 'should'
        # - Terms with NOT are 'must_not'
        
        expression = expression.lower()
        
        # Handle NOT terms first
        not_parts = re.split(r'\sNOT\s', expression, flags=re.IGNORECASE)
        main_expr = not_parts[0]
        must_not = []
        
        # Get NOT terms
        if len(not_parts) > 1:
            for not_part in not_parts[1:]:
                # Remove parentheses if present
                not_part = re.sub(r'[\(\)]', '', not_part).strip()
                must_not.append(not_part)
        
        # Handle AND/OR in main expression
        must = []
        should = []
        
        # Check if we have OR terms
        if ' OR ' in main_expr:
            or_parts = re.split(r'\sOR\s', main_expr, flags=re.IGNORECASE)
            for part in or_parts:
                part = re.sub(r'[\(\)]', '', part).strip()
                if part:
                    should.append(part)
        # Check if we have AND terms
        elif ' AND ' in main_expr:
            and_parts = re.split(r'\sAND\s', main_expr, flags=re.IGNORECASE)
            for part in and_parts:
                part = re.sub(r'[\(\)]', '', part).strip()
                if part:
                    must.append(part)
        # Simple term with no operators
        else:
            term = re.sub(r'[\(\)]', '', main_expr).strip()
            if term:
                must.append(term)
        
        return {
            "must": must,
            "should": should,
            "must_not": must_not
        }
    
    def expand_search_terms(self, terms: List[str], expansion_factor: int = 3) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Expand search terms with semantically similar words.
        
        Args:
            terms: List of search terms
            expansion_factor: Number of similar words to add per term
            
        Returns:
            Dictionary mapping original terms to lists of similar words with scores
        """
        expanded_terms = {}
        
        for term in terms:
            term = term.lower().strip()
            if not term:
                continue
                
            try:
                similar_words = self.find_similar_words(term, top_n=expansion_factor)
                expanded_terms[term] = similar_words
            except Exception as e:
                logger.error(f"Error expanding term '{term}': {e}")
                expanded_terms[term] = []
                
        return expanded_terms
    
    def filter_by_semantic_similarity(
        self, 
        query_terms: Dict[str, List[str]], 
        min_similarity: float = 0.7,
        expansion_factor: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a semantic filter based on word embeddings.
        
        Args:
            query_terms: Dictionary with 'must', 'should', 'must_not' terms
            min_similarity: Minimum similarity threshold
            expansion_factor: Number of similar words to consider per term
            
        Returns:
            Dictionary with expanded query terms
        """
        result = {
            "must": [],
            "should": [],
            "must_not": []
        }
        
        # Process must terms
        for term in query_terms["must"]:
            result["must"].append({
                "original": term,
                "expanded": self.expand_search_terms([term], expansion_factor)
            })
            
        # Process should terms
        for term in query_terms["should"]:
            result["should"].append({
                "original": term,
                "expanded": self.expand_search_terms([term], expansion_factor)
            })
            
        # Process must_not terms
        for term in query_terms["must_not"]:
            result["must_not"].append({
                "original": term,
                "expanded": self.expand_search_terms([term], expansion_factor)
            })
            
        return result
    
    def get_word_frequency_trends(self, words: List[str], time_periods: List[Tuple[int, int]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get frequency trends for words across different time periods.
        
        This is a placeholder - actual implementation would depend on having access to 
        frequency data by time period, which would need to be stored separately.
        
        Args:
            words: List of words to analyze
            time_periods: List of (start_year, end_year) tuples
            
        Returns:
            Dictionary mapping words to frequency data across time periods
        """
        # Placeholder - this would need to be implemented with actual time-based data
        return {}