# src/services/search_service.py
"""
Search service that coordinates between UI and core engine.
Provides a clean interface for the UI layer.
"""
from typing import Dict, List, Optional, Any
import logging

from src.core.engine import SpiegelRAG, AnalysisResult
from src.core.search.strategies import (
    SearchStrategy,
    StandardSearchStrategy,
    TimeWindowSearchStrategy,
    AgentSearchStrategy,
    SearchConfig,
    SearchResult
)
from src.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """
    High-level service for search operations.
    Handles strategy selection and coordination.
    """
    
    def __init__(self):
        """Initialize service with RAG engine"""
        self.engine = SpiegelRAG()
        
        # Initialize available strategies
        self._strategies = {
            'standard': self._create_standard_strategy,
            'zeitfenster': self._create_timewindow_strategy,
            'agent': self._create_agent_strategy
        }
        
        logger.info("SearchService initialized")
    
    def execute_search(self,
                      mode: str,
                      config: SearchConfig,
                      window_size: Optional[int] = None,
                      initial_count: Optional[int] = None,
                      filter_stages: Optional[List[int]] = None,
                      progress_callback: Optional[Any] = None) -> SearchResult:
        """
        Execute search with the specified mode and configuration.
        
        Args:
            mode: Search mode ('standard', 'zeitfenster', 'agent')
            config: Search configuration
            window_size: Window size for time window search
            initial_count: Initial retrieval count for agent search
            filter_stages: Filter stages for agent search
            progress_callback: Optional progress callback
            
        Returns:
            SearchResult with chunks and metadata
        """
        # Validate mode
        if mode not in self._strategies:
            raise ValueError(f"Unknown search mode: {mode}")
        
        # Create appropriate strategy
        strategy = self._strategies[mode](
            window_size=window_size,
            initial_count=initial_count,
            filter_stages=filter_stages
        )
        
        # Execute search
        try:
            logger.info(f"Executing {mode} search")
            result = self.engine.search(
                strategy=strategy,
                config=config,
                use_semantic_expansion=True
            )
            
            logger.info(f"Search completed: {result.chunk_count} chunks found")
            return result
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def analyze_results(self,
                       question: str,
                       search_result: SearchResult,
                       model: str = "hu-llm",
                       temperature: float = 0.3,
                       openai_api_key: Optional[str] = None,
                       system_prompt: Optional[str] = None) -> AnalysisResult:
        """
        Analyze search results with LLM.
        
        Args:
            question: Question to answer
            search_result: Previous search results
            model: LLM model to use
            temperature: Generation temperature
            openai_api_key: OpenAI API key if needed
            system_prompt: Custom system prompt
            
        Returns:
            AnalysisResult with answer and metadata
        """
        # Extract chunks from search result
        chunks = [doc for doc, _ in search_result.chunks]
        
        # Perform analysis
        try:
            logger.info(f"Analyzing {len(chunks)} chunks with {model}")
            
            result = self.engine.analyze(
                question=question,
                chunks=chunks,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                openai_api_key=openai_api_key
            )
            
            # Add search metadata to analysis result
            result.metadata.update({
                "search_mode": search_result.metadata.get("strategy", "unknown"),
                "chunks_searched": len(chunks)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def find_similar_words(self, word: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar words using embeddings.
        
        Args:
            word: Word to find similar words for
            top_n: Number of similar words to return
            
        Returns:
            List of similar words with scores
        """
        return self.engine.find_similar_words(word, top_n)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics and system status"""
        return {
            "vector_store_connected": self.engine.vector_store is not None,
            "llm_available": self.engine.llm_service is not None,
            "embeddings_available": self.engine._has_embeddings,
            "available_chunk_sizes": settings.AVAILABLE_CHUNK_SIZES,
            "year_range": [settings.MIN_YEAR, settings.MAX_YEAR]
        }
    
    # Strategy creation methods
    def _create_standard_strategy(self, **kwargs) -> SearchStrategy:
        """Create standard search strategy"""
        return StandardSearchStrategy()
    
    def _create_timewindow_strategy(self, window_size: Optional[int] = None, **kwargs) -> SearchStrategy:
        """Create time window search strategy"""
        return TimeWindowSearchStrategy(
            window_size=window_size or 5
        )
    
    def _create_agent_strategy(self,
                              initial_count: Optional[int] = None,
                              filter_stages: Optional[List[int]] = None,
                              **kwargs) -> SearchStrategy:
        """Create agent search strategy"""
        return AgentSearchStrategy(
            initial_count=initial_count or 100,
            filter_stages=filter_stages or [50, 20, 10],
            llm_service=self.engine.llm_service,
            model="hu-llm"  # Could be made configurable
        )


# Singleton instance for easy access
_service_instance: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create the search service instance"""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = SearchService()
    
    return _service_instance