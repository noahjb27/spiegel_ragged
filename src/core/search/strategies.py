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
    """Unified configuration for all search types"""
    content_description: str
    year_range: Tuple[int, int] = (1948, 1979)
    chunk_size: int = 3000
    keywords: Optional[str] = None
    search_fields: List[str] = field(default_factory=lambda: ["Text"])
    enforce_keywords: bool = True
    top_k: int = 10
    min_relevance_score: float = 0.3
    

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
        
        logger.info(f"Searching across {len(windows)} time windows")
        
        if progress_callback:
            progress_callback(f"Searching {len(windows)} time windows...", 0.0)
        
        # Calculate results per window
        results_per_window = max(1, config.top_k // len(windows))
        all_chunks = []
        
        # Search each window
        for i, (window_start, window_end) in enumerate(windows):
            if progress_callback:
                progress = (i / len(windows))
                progress_callback(f"Searching {window_start}-{window_end}...", progress)
            
            # Create window-specific filter
            window_filter = vector_store.build_metadata_filter(
                year_range=[window_start, window_end],
                keywords=None,
                search_in=None
            )
            
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
            
            all_chunks.extend(window_chunks)
        
        # Sort by relevance and take top_k
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        final_chunks = all_chunks[:config.top_k]
        
        if progress_callback:
            progress_callback(f"Found {len(final_chunks)} chunks", 1.0)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            chunks=final_chunks,
            metadata={
                "strategy": "time_window",
                "search_time": search_time,
                "window_size": self.window_size,
                "windows": windows,
                "config": {
                    "year_range": config.year_range,
                    "chunk_size": config.chunk_size,
                    "keywords": config.keywords
                }
            }
        )


class AgentSearchStrategy(SearchStrategy):
    """Multi-stage filtered search with LLM evaluation"""
    
    def __init__(self, 
                 initial_count: int = 100,
                 filter_stages: List[int] = None,
                 llm_service: Any = None,
                 model: str = "hu-llm"):
        self.initial_count = initial_count
        self.filter_stages = filter_stages or [50, 20, 10]
        self.llm_service = llm_service
        self.model = model
    
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable] = None) -> SearchResult:
        """Execute agent-based search with progressive filtering"""
        
        if not self.llm_service:
            raise ValueError("LLM service required for agent search")
        
        start_time = time.time()
        stage_times = []
        
        if progress_callback:
            progress_callback("Starting agent search...", 0.0)
        
        # Stage 1: Initial broad retrieval
        stage_start = time.time()
        filter_dict = self._build_metadata_filter(vector_store, config.year_range)
        
        initial_chunks = vector_store.similarity_search(
            query=config.content_description,
            chunk_size=config.chunk_size,
            k=self.initial_count,
            filter_dict=filter_dict,
            min_relevance_score=0.25,  # Lower threshold for initial retrieval
            keywords=config.keywords,
            search_in=config.search_fields,
            enforce_keywords=config.enforce_keywords
        )
        
        stage_times.append(("Initial Retrieval", time.time() - stage_start))
        
        if not initial_chunks:
            return SearchResult(
                chunks=[],
                metadata={
                    "strategy": "agent",
                    "error": "No initial chunks found",
                    "search_time": time.time() - start_time
                }
            )
        
        # Progressive filtering stages
        current_chunks = initial_chunks
        evaluations = []
        
        for i, target_count in enumerate(self.filter_stages):
            if len(current_chunks) <= target_count:
                continue
            
            stage_start = time.time()
            
            if progress_callback:
                progress = (i + 1) / (len(self.filter_stages) + 1)
                progress_callback(f"Filtering stage {i+1}: {len(current_chunks)} → {target_count}", progress)
            
            # Evaluate chunks with LLM
            evaluated_chunks = self._evaluate_chunks(
                chunks=current_chunks,
                question=config.content_description,
                top_k=target_count
            )
            
            current_chunks = evaluated_chunks[:target_count]
            evaluations.extend(evaluated_chunks)
            
            stage_times.append((f"Filter Stage {i+1}", time.time() - stage_start))
        
        if progress_callback:
            progress_callback(f"Completed with {len(current_chunks)} chunks", 1.0)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            chunks=current_chunks,
            metadata={
                "strategy": "agent",
                "search_time": search_time,
                "initial_count": self.initial_count,
                "filter_stages": self.filter_stages,
                "stage_times": stage_times,
                "evaluations": evaluations[:len(current_chunks)],
                "config": {
                    "year_range": config.year_range,
                    "chunk_size": config.chunk_size,
                    "keywords": config.keywords
                }
            }
        )
    
    def _evaluate_chunks(self, 
                        chunks: List[Tuple[Document, float]], 
                        question: str,
                        top_k: int) -> List[Tuple[Document, float]]:
        """
        Evaluate chunks using LLM (simplified version).
        In production, this would batch evaluate chunks.
        """
        # This is a simplified placeholder
        # The actual implementation would use the LLM to score chunks
        # For now, just return sorted by existing scores
        return sorted(chunks, key=lambda x: x[1], reverse=True)[:top_k]