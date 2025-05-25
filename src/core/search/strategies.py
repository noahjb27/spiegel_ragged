# src/core/search/strategies.py
"""
Search strategy implementations following the Strategy pattern.
Each strategy encapsulates a different search approach.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)



@dataclass
class SearchConfig:
    """Unified configuration for all search types with consistent field names"""
    content_description: str
    year_range: Tuple[int, int] = (1948, 1979)
    chunk_size: int = 3000
    keywords: Optional[str] = None
    search_fields: List[str] = field(default_factory=lambda: ["Text"])  # Changed from search_in
    enforce_keywords: bool = True
    top_k: int = 10
    min_relevance_score: float = 0.3
    
    # Add compatibility property for legacy code
    @property
    def search_in(self) -> List[str]:
        """Compatibility property for legacy code that uses 'search_in'"""
        return self.search_fields
    
    @search_in.setter
    def search_in(self, value: List[str]):
        """Compatibility setter for legacy code"""
        self.search_fields = value

@dataclass
class SearchResult:
    """Unified result structure"""
    chunks: List[Document]
    metadata: Dict[str, Any]
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class SearchStrategy(ABC):
    """Base class for all search strategies"""
    
    @abstractmethod
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable[[str, float], None]] = None) -> SearchResult:
        """
        Execute search with this strategy.
        
        Args:
            config: Search configuration
            vector_store: Vector store interface
            progress_callback: Optional callback for progress updates
            
        Returns:
            SearchResult with chunks and metadata
        """
        pass
    
    def _build_metadata_filter(self, 
                              vector_store: Any,
                              year_range: Tuple[int, int]) -> Optional[Dict]:
        """Build metadata filter for year range"""
        return vector_store.build_metadata_filter(
            year_range=list(year_range),
            keywords=None,
            search_in=None
        )


class StandardSearchStrategy(SearchStrategy):
    """Simple similarity search - the most basic strategy"""
    
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable] = None) -> SearchResult:
        """Execute standard similarity search"""
        
        start_time = time.time()
        
        if progress_callback:
            progress_callback("Starting standard search...", 0.0)
        
        # Build filter
        filter_dict = self._build_metadata_filter(vector_store, config.year_range)
        
        # Execute search
        chunks = vector_store.similarity_search(
            query=config.content_description,
            chunk_size=config.chunk_size,
            k=config.top_k,
            filter_dict=filter_dict,
            min_relevance_score=config.min_relevance_score,
            keywords=config.keywords,
            search_in=config.search_fields,
            enforce_keywords=config.enforce_keywords
        )
        
        if progress_callback:
            progress_callback(f"Found {len(chunks)} chunks", 1.0)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            chunks=chunks,
            metadata={
                "strategy": "standard",
                "search_time": search_time,
                "config": {
                    "year_range": config.year_range,
                    "chunk_size": config.chunk_size,
                    "keywords": config.keywords
                }
            }
        )


class TimeWindowSearchStrategy(SearchStrategy):
    """Search across time windows for better temporal coverage"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable] = None) -> SearchResult:
        """Execute time-windowed search"""
        
        start_time = time.time()
        start_year, end_year = config.year_range
        
        # Create time windows
        windows = []
        for window_start in range(start_year, end_year + 1, self.window_size):
            window_end = min(window_start + self.window_size - 1, end_year)
            windows.append((window_start, window_end))
        
        logger.info(f"Searching across {len(windows)} time windows of size {self.window_size}")
        logger.info(f"Windows: {windows}")
        
        if progress_callback:
            progress_callback(f"Searching {len(windows)} time windows...", 0.0)
        
        # Calculate results per window - ensure we get at least 1 result per window
        base_results_per_window = max(1, config.top_k // len(windows))
        # Add some extra to account for filtering and ensure we have enough
        results_per_window = base_results_per_window + 2
        
        all_chunks = []
        window_counts = {}
        
        # Search each window
        for i, (window_start, window_end) in enumerate(windows):
            if progress_callback:
                progress = (i / len(windows))
                progress_callback(f"Searching {window_start}-{window_end}...", progress)
            
            logger.info(f"Searching window {i+1}/{len(windows)}: {window_start}-{window_end}")
            
            # Create window-specific filter
            window_filter = vector_store.build_metadata_filter(
                year_range=[window_start, window_end],
                keywords=None,
                search_in=None
            )
            
            logger.info(f"Window filter: {window_filter}")
            
            try:
                # Search this window
                window_chunks = vector_store.similarity_search(
                    query=config.content_description,
                    chunk_size=config.chunk_size,
                    k=results_per_window,
                    filter_dict=window_filter,
                    min_relevance_score=config.min_relevance_score,
                    keywords=config.keywords,
                    search_in=config.search_fields,
                    enforce_keywords=config.enforce_keywords
                )
                
                logger.info(f"Window {window_start}-{window_end}: found {len(window_chunks)} chunks")
                
                # Track window statistics
                window_key = f"{window_start}-{window_end}"
                window_counts[window_key] = len(window_chunks)
                
                # Add window metadata to each chunk
                for doc, score in window_chunks:
                    # Add window info to metadata
                    doc.metadata['time_window'] = window_key
                    doc.metadata['window_start'] = window_start
                    doc.metadata['window_end'] = window_end
                
                all_chunks.extend(window_chunks)
                
            except Exception as e:
                logger.error(f"Error searching window {window_start}-{window_end}: {e}")
                window_counts[f"{window_start}-{window_end}"] = 0
                # Continue with other windows
                continue
        
        logger.info(f"Total chunks from all windows: {len(all_chunks)}")
        logger.info(f"Window distribution: {window_counts}")
        
        if not all_chunks:
            logger.warning("No chunks found in any time window")
            if progress_callback:
                progress_callback("No results found", 1.0)
            
            return SearchResult(
                chunks=[],
                metadata={
                    "strategy": "time_window",
                    "search_time": time.time() - start_time,
                    "window_size": self.window_size,
                    "windows": windows,
                    "window_counts": window_counts,
                    "error": "No results found in any time window",
                    "config": {
                        "year_range": config.year_range,
                        "chunk_size": config.chunk_size,
                        "keywords": config.keywords
                    }
                }
            )
        
        # Sort by relevance score (highest first) and take top_k
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        final_chunks = all_chunks[:config.top_k]
        
        logger.info(f"Final selection: {len(final_chunks)} chunks")
        
        if progress_callback:
            progress_callback(f"Found {len(final_chunks)} chunks", 1.0)
        
        search_time = time.time() - start_time
        
        # Log final distribution
        final_window_dist = {}
        for doc, score in final_chunks:
            window_key = doc.metadata.get('time_window', 'unknown')
            final_window_dist[window_key] = final_window_dist.get(window_key, 0) + 1
        
        logger.info(f"Final distribution across windows: {final_window_dist}")
        
        return SearchResult(
            chunks=final_chunks,
            metadata={
                "strategy": "time_window",
                "search_time": search_time,
                "window_size": self.window_size,
                "windows": windows,
                "window_counts": window_counts,
                "final_distribution": final_window_dist,
                "total_chunks_found": len(all_chunks),
                "final_chunks_selected": len(final_chunks),
                "config": {
                    "year_range": config.year_range,
                    "chunk_size": config.chunk_size,
                    "keywords": config.keywords
                }
            }
        )

class AgentSearchStrategy(SearchStrategy):
    """Multi-stage filtered search with LLM evaluation using existing RetrievalAgent"""
    
    def __init__(self, 
                 initial_count: int = 100,
                 filter_stages: List[int] = None,
                 llm_service: Any = None,
                 model: str = "hu-llm"):
        self.initial_count = initial_count
        self.filter_stages = filter_stages or [50, 20, 10]
        self.llm_service = llm_service
        self.model = model
        
        # Import and initialize RetrievalAgent
        from src.core.retrieval_agent import RetrievalAgent
        if llm_service:
            # We'll need the vector store too, but we'll get it in the search method
            self.retrieval_agent_class = RetrievalAgent
        else:
            raise ValueError("LLM service required for agent search")
    
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable] = None,
              **kwargs) -> SearchResult:
        """
        Execute agent-based search with progressive filtering using RetrievalAgent
        """
        
        start_time = time.time()
        
        if progress_callback:
            progress_callback("Initializing agent search...", 0.0)
        
        logger.info(f"Starting agent search with {self.initial_count} initial chunks")
        logger.info(f"Filter stages: {self.filter_stages}")
        
        try:
            # Initialize RetrievalAgent with vector store and LLM service
            retrieval_agent = self.retrieval_agent_class(vector_store, self.llm_service)
            
            # Extract additional parameters from kwargs
            question = kwargs.get('question')
            openai_api_key = kwargs.get('openai_api_key')
            
            # Use content_description as the question if no specific question provided
            search_question = question or config.content_description
            
            if progress_callback:
                progress_callback("Running agent retrieval and refinement...", 0.2)
            
            # Call the existing retrieve_and_refine method
            refined_chunks, agent_metadata = retrieval_agent.retrieve_and_refine(
                question=search_question,
                content_description=config.content_description,
                year_range=list(config.year_range),
                chunk_size=config.chunk_size,
                keywords=config.keywords,
                search_in=config.search_fields,
                enforce_keywords=config.enforce_keywords,
                initial_retrieval_count=self.initial_count,
                filter_stages=self.filter_stages,
                model=self.model,
                openai_api_key=openai_api_key,
                with_evaluations=True
            )
            
            if progress_callback:
                progress_callback("Processing agent results...", 0.8)
            
            # Convert the agent results to the expected format
            # refined_chunks format: List[Tuple[Document, float, Optional[str]]]
            # SearchResult expects: List[Tuple[Document, float]]
            
            final_chunks = []
            evaluations = []
            
            for item in refined_chunks:
                if len(item) == 3:
                    doc, vector_score, eval_text = item
                    final_chunks.append((doc, vector_score))
                    
                    # Store evaluation for metadata
                    evaluations.append({
                        "title": doc.metadata.get('Artikeltitel', 'Unknown'),
                        "date": doc.metadata.get('Datum', 'Unknown'),
                        "relevance_score": vector_score,
                        "evaluation": eval_text or "No evaluation available"
                    })
                else:
                    # Handle case where evaluation text is not available
                    doc, vector_score = item[:2]
                    final_chunks.append((doc, vector_score))
                    
                    evaluations.append({
                        "title": doc.metadata.get('Artikeltitel', 'Unknown'),
                        "date": doc.metadata.get('Datum', 'Unknown'),
                        "relevance_score": vector_score,
                        "evaluation": "Evaluation not available"
                    })
            
            if progress_callback:
                progress_callback(f"Agent search completed: {len(final_chunks)} chunks", 1.0)
            
            search_time = time.time() - start_time
            
            logger.info(f"Agent search completed: {len(final_chunks)} final chunks in {search_time:.2f}s")
            
            return SearchResult(
                chunks=final_chunks,
                metadata={
                    "strategy": "agent",
                    "search_time": search_time,
                    "initial_count": self.initial_count,
                    "filter_stages": self.filter_stages,
                    "agent_metadata": agent_metadata,
                    "evaluations": evaluations,
                    "model_used": self.model,
                    "question": search_question,
                    "config": {
                        "year_range": config.year_range,
                        "chunk_size": config.chunk_size,
                        "keywords": config.keywords
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Agent search failed: {e}", exc_info=True)
            
            if progress_callback:
                progress_callback(f"Agent search failed: {str(e)}", 1.0)
            
            return SearchResult(
                chunks=[],
                metadata={
                    "strategy": "agent",
                    "search_time": time.time() - start_time,
                    "error": str(e),
                    "initial_count": self.initial_count,
                    "filter_stages": self.filter_stages
                }
            )