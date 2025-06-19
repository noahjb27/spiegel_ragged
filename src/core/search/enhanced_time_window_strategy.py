# src/core/search/enhanced_time_window_strategy.py
"""
Enhanced time window search strategy with precise control over chunks per window.
"""
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable

from langchain.docstore.document import Document

from src.core.search.strategies import SearchStrategy, SearchConfig, SearchResult

logger = logging.getLogger(__name__)


class EnhancedTimeWindowSearchStrategy(SearchStrategy):
    """
    Enhanced time window search strategy with precise chunks per window control.
    
    This strategy divides the search time range into windows and retrieves a specific
    number of chunks per window, ensuring balanced temporal coverage.
    """
    
    def __init__(self, window_size: int = 5, chunks_per_window: int = 5):
        """
        Initialize the enhanced time window strategy.
        
        Args:
            window_size: Size of each time window in years
            chunks_per_window: Number of chunks to retrieve per window
        """
        self.window_size = window_size
        self.chunks_per_window = chunks_per_window
        logger.info(f"Initialized EnhancedTimeWindowSearchStrategy: {window_size}y windows, {chunks_per_window} chunks/window")
    
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable[[str, float], None]] = None) -> SearchResult:
        """
        Execute enhanced time-windowed search with precise chunk control.
        
        Args:
            config: Search configuration
            vector_store: Vector store interface
            progress_callback: Optional progress callback
            
        Returns:
            SearchResult with balanced temporal chunks
        """
        start_time = time.time()
        start_year, end_year = config.year_range
        
        # Create time windows
        windows = self._create_time_windows(start_year, end_year)
        logger.info(f"Enhanced time window search: {len(windows)} windows, {self.chunks_per_window} chunks per window")
        logger.info(f"Windows: {windows}")
        
        if progress_callback:
            progress_callback(f"Searching {len(windows)} time windows...", 0.0)
        
        all_chunks = []
        window_counts = {}
        window_details = {}
        failed_windows = []
        
        # Search each window
        for i, (window_start, window_end) in enumerate(windows):
            if progress_callback:
                progress = (i / len(windows))
                progress_callback(f"Searching {window_start}-{window_end}...", progress)
            
            window_key = f"{window_start}-{window_end}"
            logger.info(f"Processing window {i+1}/{len(windows)}: {window_key}")
            
            try:
                # Search this specific window
                window_chunks = self._search_window(
                    config, vector_store, window_start, window_end, window_key
                )
                
                window_counts[window_key] = len(window_chunks)
                window_details[window_key] = {
                    "start_year": window_start,
                    "end_year": window_end,
                    "chunks_found": len(window_chunks),
                    "target_chunks": self.chunks_per_window
                }
                
                all_chunks.extend(window_chunks)
                
                logger.info(f"Window {window_key}: found {len(window_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error searching window {window_start}-{window_end}: {e}")
                window_counts[window_key] = 0
                window_details[window_key] = {
                    "start_year": window_start,
                    "end_year": window_end,
                    "chunks_found": 0,
                    "target_chunks": self.chunks_per_window,
                    "error": str(e)
                }
                failed_windows.append(window_key)
                continue
        
        # Sort all chunks by relevance score (highest first)
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        
        search_time = time.time() - start_time
        
        # Calculate statistics
        total_chunks_found = len(all_chunks)
        expected_chunks = len(windows) * self.chunks_per_window
        coverage_percentage = (total_chunks_found / expected_chunks) * 100 if expected_chunks > 0 else 0
        
        # Create final distribution summary
        final_distribution = {}
        for doc, score in all_chunks:
            window_key = doc.metadata.get('time_window', 'unknown')
            final_distribution[window_key] = final_distribution.get(window_key, 0) + 1
        
        logger.info(f"Enhanced time window search completed:")
        logger.info(f"  Total chunks found: {total_chunks_found}")
        logger.info(f"  Expected chunks: {expected_chunks}")
        logger.info(f"  Coverage: {coverage_percentage:.1f}%")
        logger.info(f"  Failed windows: {len(failed_windows)}")
        logger.info(f"  Search time: {search_time:.2f}s")
        
        if progress_callback:
            progress_callback(f"Found {total_chunks_found} chunks across {len(windows)} windows", 1.0)
        
        return SearchResult(
            chunks=all_chunks,
            metadata={
                "strategy": "enhanced_time_window",
                "search_time": search_time,
                "window_size": self.window_size,
                "chunks_per_window": self.chunks_per_window,
                "windows": windows,
                "window_counts": window_counts,
                "window_details": window_details,
                "final_distribution": final_distribution,
                "total_chunks_found": total_chunks_found,
                "expected_chunks": expected_chunks,
                "coverage_percentage": coverage_percentage,
                "failed_windows": failed_windows,
                "successful_windows": len(windows) - len(failed_windows)
            }
        )
    
    def _create_time_windows(self, start_year: int, end_year: int) -> List[Tuple[int, int]]:
        """
        Create time windows for the given year range.
        
        Args:
            start_year: Start year of the range
            end_year: End year of the range
            
        Returns:
            List of (start_year, end_year) tuples for each window
        """
        windows = []
        
        current_start = start_year
        while current_start <= end_year:
            current_end = min(current_start + self.window_size - 1, end_year)
            windows.append((current_start, current_end))
            current_start = current_end + 1
        
        return windows
    
    def _search_window(self, 
                      config: SearchConfig, 
                      vector_store: Any, 
                      window_start: int, 
                      window_end: int,
                      window_key: str) -> List[Tuple[Document, float]]:
        """
        Search a specific time window.
        
        Args:
            config: Search configuration
            vector_store: Vector store interface
            window_start: Start year of window
            window_end: End year of window
            window_key: String identifier for the window
            
        Returns:
            List of (Document, relevance_score) tuples
        """
        # Create window-specific filter
        window_filter = vector_store.build_metadata_filter(
            year_range=[window_start, window_end],
            keywords=None,
            search_in=None
        )
        
        logger.debug(f"Window {window_key} filter: {window_filter}")
        
        # Perform search for this window
        window_chunks = vector_store.similarity_search(
            query=config.content_description,
            chunk_size=config.chunk_size,
            k=self.chunks_per_window,  # Use per-window limit
            filter_dict=window_filter,
            min_relevance_score=config.min_relevance_score,
            keywords=config.keywords,
            search_in=config.search_fields,
            enforce_keywords=config.enforce_keywords
        )
        
        # Add window metadata to each chunk
        for doc, score in window_chunks:
            doc.metadata['time_window'] = window_key
            doc.metadata['window_start'] = window_start
            doc.metadata['window_end'] = window_end
            doc.metadata['chunks_per_window_target'] = self.chunks_per_window
        
        return window_chunks
    
    def get_window_summary(self, search_result: SearchResult) -> Dict[str, Any]:
        """
        Get a summary of the window search results.
        
        Args:
            search_result: Result from the search method
            
        Returns:
            Dictionary with summary information
        """
        metadata = search_result.metadata
        
        summary = {
            "total_windows": len(metadata.get('windows', [])),
            "successful_windows": metadata.get('successful_windows', 0),
            "failed_windows": len(metadata.get('failed_windows', [])),
            "total_chunks": len(search_result.chunks),
            "expected_chunks": metadata.get('expected_chunks', 0),
            "coverage_percentage": metadata.get('coverage_percentage', 0),
            "avg_chunks_per_window": 0,
            "window_distribution": metadata.get('final_distribution', {}),
            "search_time": metadata.get('search_time', 0),
            "chunks_per_window_target": self.chunks_per_window,
            "window_size_years": self.window_size
        }
        
        # Calculate average chunks per window
        if summary["successful_windows"] > 0:
            summary["avg_chunks_per_window"] = summary["total_chunks"] / summary["successful_windows"]
        
        return summary


class AdaptiveTimeWindowSearchStrategy(EnhancedTimeWindowSearchStrategy):
    """
    Adaptive time window strategy that adjusts window size based on data density.
    
    This strategy can adapt window sizes if certain windows have very few results,
    potentially merging adjacent windows or adjusting the search parameters.
    """
    
    def __init__(self, 
                 base_window_size: int = 5, 
                 chunks_per_window: int = 5,
                 min_chunks_threshold: int = 2,
                 adaptive_mode: bool = True):
        """
        Initialize the adaptive time window strategy.
        
        Args:
            base_window_size: Base size of each time window in years
            chunks_per_window: Target number of chunks per window
            min_chunks_threshold: Minimum chunks before considering adaptation
            adaptive_mode: Whether to enable adaptive behavior
        """
        super().__init__(base_window_size, chunks_per_window)
        self.min_chunks_threshold = min_chunks_threshold
        self.adaptive_mode = adaptive_mode
        logger.info(f"Initialized AdaptiveTimeWindowSearchStrategy: adaptive={adaptive_mode}, threshold={min_chunks_threshold}")
    
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable[[str, float], None]] = None) -> SearchResult:
        """
        Execute adaptive time-windowed search.
        
        This method first runs the standard time window search, then optionally
        applies adaptive logic to improve coverage.
        """
        # First, run the standard enhanced search
        result = super().search(config, vector_store, progress_callback)
        
        if not self.adaptive_mode:
            return result
        
        # Apply adaptive logic if needed
        metadata = result.metadata
        window_details = metadata.get('window_details', {})
        
        # Check for windows with insufficient results
        low_yield_windows = []
        for window_key, details in window_details.items():
            if details.get('chunks_found', 0) < self.min_chunks_threshold:
                low_yield_windows.append(window_key)
        
        if low_yield_windows and progress_callback:
            progress_callback(f"Applying adaptive logic to {len(low_yield_windows)} low-yield windows...", 0.9)
        
        # For now, just log the adaptation opportunities
        # In a full implementation, you could re-search with adjusted parameters
        if low_yield_windows:
            logger.info(f"Adaptive strategy identified {len(low_yield_windows)} windows for potential adjustment: {low_yield_windows}")
            metadata['adaptive_analysis'] = {
                'low_yield_windows': low_yield_windows,
                'adaptation_applied': False,
                'reason': 'Basic implementation - adaptation not yet implemented'
            }
        
        return result