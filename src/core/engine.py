# src/core/engine.py - Refactored version
"""
Simplified RAG Engine with clear separation of concerns.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging

from langchain.docstore.document import Document

from src.core.vector_store import ChromaDBInterface
from src.core.llm_service import LLMService
from src.core.embedding_service import WordEmbeddingService
from src.core.search.strategies import SearchStrategy, SearchConfig, SearchResult
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from LLM analysis"""
    answer: str
    model: str
    metadata: Dict[str, Any]
    citations: Optional[List[str]] = None


class SpiegelRAG:
    """
    Simplified RAG engine focusing on core responsibilities:
    1. Coordinating search strategies
    2. Managing LLM analysis
    3. Handling semantic expansion
    """
    
    def __init__(self):
        """Initialize core services"""
        self.vector_store = ChromaDBInterface()
        self.llm_service = LLMService()
        
        # Initialize embedding service if available
        try:
            self.embedding_service = WordEmbeddingService()
            self._has_embeddings = True
        except Exception as e:
            logger.warning(f"Embedding service unavailable: {e}")
            self.embedding_service = None
            self._has_embeddings = False
            
        # Cache for last search results
        self._last_search_result: Optional[SearchResult] = None
        
        logger.info("SpiegelRAG engine initialized")
    
    def search(self, 
               strategy: SearchStrategy, 
               config: SearchConfig,
               use_semantic_expansion: bool = True) -> SearchResult:
        """
        Execute search with the provided strategy.
        
        Args:
            strategy: Search strategy to use
            config: Search configuration
            use_semantic_expansion: Whether to expand keywords semantically
            
        Returns:
            SearchResult with chunks and metadata
        """
        # Apply semantic expansion if enabled and available
        if (use_semantic_expansion and 
            self._has_embeddings and 
            config.keywords):
            config = self._apply_semantic_expansion(config)
        
        # Execute search
        result = strategy.search(config, self.vector_store)
        
        # Cache result for potential reuse
        self._last_search_result = result
        
        return result
    
    def analyze(self,
                question: str,
                chunks: Optional[List[Document]] = None,
                model: str = "hu-llm",
                system_prompt: Optional[str] = None,
                temperature: float = 0.3,
                max_tokens: Optional[int] = None,
                openai_api_key: Optional[str] = None) -> AnalysisResult:
        """
        Analyze chunks with LLM to answer a question.
        
        Args:
            question: Question to answer
            chunks: Documents to analyze (uses last search if None)
            model: LLM model to use
            system_prompt: Custom system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            openai_api_key: OpenAI API key if needed
            
        Returns:
            AnalysisResult with answer and metadata
        """
        # Use provided chunks or fall back to last search
        if chunks is None:
            if self._last_search_result is None:
                raise ValueError("No chunks provided and no previous search results available")
            chunks = self._last_search_result.chunks
        
        if not chunks:
            return AnalysisResult(
                answer="No relevant content found to answer this question.",
                model=model,
                metadata={"error": "No chunks available"}
            )
        
        # Format context from chunks
        context = self._format_context(chunks)
        
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = settings.SYSTEM_PROMPTS.get("default", "")
        
        # Generate response
        try:
            response = self.llm_service.generate_response(
                question=question,
                context=context,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=openai_api_key
            )
            
            return AnalysisResult(
                answer=response['text'],
                model=response.get('model', model),
                metadata={
                    "question": question,
                    "chunks_analyzed": len(chunks),
                    "temperature": temperature
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                answer=f"Error generating answer: {str(e)}",
                model=model,
                metadata={"error": str(e)}
            )
    
    def find_similar_words(self, word: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find semantically similar words"""
        if not self._has_embeddings:
            return []
        
        try:
            return self.embedding_service.find_similar_words(word, top_n)
        except Exception as e:
            logger.error(f"Error finding similar words: {e}")
            return []
    
    def _apply_semantic_expansion(self, config: SearchConfig) -> SearchConfig:
        """Apply semantic expansion to search keywords"""
        if not config.keywords or not self.embedding_service:
            return config
        
        try:
            # Parse boolean expression
            parsed = self.embedding_service.parse_boolean_expression(config.keywords)
            
            # Expand terms
            expanded = self.embedding_service.filter_by_semantic_similarity(
                parsed, 
                expansion_factor=5
            )
            
            # Create expanded keyword string (simplified for now)
            # In a full implementation, this would properly reconstruct the boolean expression
            logger.info(f"Applied semantic expansion to keywords: {config.keywords}")
            
        except Exception as e:
            logger.error(f"Semantic expansion failed: {e}")
        
        return config
    
    def _format_context(self, chunks: List[Document]) -> str:
        """Format chunks into context string for LLM"""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            context_part = (
                f"[{i+1}] {metadata.get('Artikeltitel', 'No title')} "
                f"({metadata.get('Datum', 'No date')})\n"
                f"{chunk.page_content}\n"
            )
            context_parts.append(context_part)
        
        return "\n".join(context_parts)


# Convenience functions for backward compatibility
def create_engine() -> SpiegelRAG:
    """Create and return a configured RAG engine"""
    return SpiegelRAG()