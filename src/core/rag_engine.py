"""
Core RAG Engine for Spiegel search and question answering.
Refactored to separate retrieval and analysis steps.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any

from langchain.docstore.document import Document

from src.core.vectore_store import ChromaDBInterface
from src.core.llm_service import LLMService
try:
    from src.core.embedding_service import WordEmbeddingService
    EMBEDDING_SERVICE_AVAILABLE = True
except Exception:
    EMBEDDING_SERVICE_AVAILABLE = False
    
from src.config import settings

logger = logging.getLogger(__name__)

class SpiegelRAGEngine:
    """Main RAG engine for Spiegel search and question answering."""
    
    def __init__(self):
        """Initialize the RAG engine components."""
        self.vector_store = ChromaDBInterface()
        self.llm_service = LLMService()
        
        # Conditionally initialize embedding service
        if EMBEDDING_SERVICE_AVAILABLE:
            try:
                self.embedding_service = WordEmbeddingService()
                self._has_embedding_service = True
            except Exception as e:
                logger.error(f"Failed to initialize embedding service: {e}")
                self._has_embedding_service = False
        else:
            self._has_embedding_service = False
            
        # Storage for cached retrieval results
        self.last_retrieval_results = None
        self.last_retrieval_params = None
            
        logger.info("Initialized SpiegelRAGEngine")
    
    def retrieve(
        self,
        content_description: str,
        year_range: Optional[List[int]] = None,
        chunk_size: Optional[int] = None,
        keywords: Optional[str] = None,
        search_in: Optional[List[str]] = None,
        use_iterative_search: bool = False,
        time_window_size: int = 5,
        min_relevance_score: float = 0.3,
        top_k: int = 10,
        use_semantic_expansion: bool = True,
        semantic_expansion_factor: int = 3,
        enforce_keywords: bool = True
    ) -> Dict[str, Any]:
        """
        Perform content retrieval based on description and filters.
        
        Args:
            content_description: Description of the content to retrieve
            year_range: Optional range of years to search
            chunk_size: Optional size of chunks to retrieve
            keywords: Optional boolean expression for keyword filtering
            search_in: Optional list of fields to search in
            use_iterative_search: Whether to use iterative time window search
            time_window_size: Size of time windows in years
            min_relevance_score: Minimum relevance score for chunks
            top_k: Maximum number of chunks to retrieve
            use_semantic_expansion: Whether to expand keywords semantically
            semantic_expansion_factor: Number of similar words to add per term
            enforce_keywords: Whether to strictly enforce keyword presence
            
        Returns:
            Dict containing retrieved chunks and metadata
        """
        # Use defaults if not provided
        chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        year_range = year_range or [settings.MIN_YEAR, settings.MAX_YEAR]
        
        # Store retrieval parameters for potential caching
        self.last_retrieval_params = {
            "content_description": content_description,
            "year_range": year_range,
            "chunk_size": chunk_size,
            "keywords": keywords,
            "search_in": search_in,
            "use_iterative_search": use_iterative_search,
            "time_window_size": time_window_size,
            "min_relevance_score": min_relevance_score,
            "top_k": top_k,
            "use_semantic_expansion": use_semantic_expansion,
            "semantic_expansion_factor": semantic_expansion_factor,
            "enforce_keywords": enforce_keywords
        }
        
        # Perform semantic expansion of keywords if requested
        expanded_keywords = keywords
        if use_semantic_expansion and keywords and self._has_embedding_service:
            try:
                # Parse the boolean expression
                parsed_terms = self.embedding_service.parse_boolean_expression(keywords)
                
                # Expand terms with semantically similar words
                expanded_terms = self.embedding_service.filter_by_semantic_similarity(
                    parsed_terms, 
                    expansion_factor=semantic_expansion_factor
                )
                
                # Log the expanded terms for debugging
                logger.info(f"Expanded keywords: {expanded_terms}")
                
            except Exception as e:
                logger.error(f"Error in semantic expansion: {e}")
        
        # Build metadata filter
        filter_dict = self.vector_store.build_metadata_filter(
            year_range=year_range,
            # Handle keywords separately for hard filtering
            keywords=None,
            search_in=None
        )
                   
        # Store all keyword filtering parameters for consistent reuse
        keyword_params = {
            "keywords": keywords,
            "search_in": search_in,
            "enforce_keywords": enforce_keywords
        }
                
        try:
            # Choose the appropriate search method based on configuration
            if use_iterative_search:
                # Iterative time window search
                chunks = self._perform_iterative_search(
                    content_description,
                    year_range,
                    chunk_size,
                    time_window_size,
                    filter_dict,
                    min_relevance_score,
                    top_k,
                    **keyword_params  # Pass keyword filtering parameters
                )
            else:
                # Standard single query search
                chunks = self.vector_store.similarity_search(
                    content_description,
                    chunk_size,
                    k=top_k,
                    filter_dict=filter_dict,
                    min_relevance_score=min_relevance_score,
                    keywords=keywords,
                    search_in=search_in,
                    enforce_keywords=enforce_keywords
                )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            chunks = []
        
        # Format chunks for response
        formatted_chunks = self._format_chunks_for_response(chunks)
        
        # Store retrieval results for reuse
        self.last_retrieval_results = chunks
        
        # Return retrieval results
        return {
            "chunks": formatted_chunks,
            "metadata": {
                "content_description": content_description,
                "chunk_size": chunk_size,
                "year_range": year_range,
                "chunks_count": len(chunks)
            }
        }
    
    def analyze(
        self,
        question: str,
        chunks: Optional[List[Tuple[Document, float]]] = None,
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        with_citations: Optional[bool] = None,
        temperature: float = 0.3,  
        max_tokens: Optional[int] = None

    ) -> Dict[str, Any]:
        """
        Generate an answer to a question based on previously retrieved chunks.
        
        Args:
            question: Question to answer based on the chunks
            chunks: Optional list of document chunks (if None, uses last retrieved)
            model: Optional LLM model to use
            openai_api_key: Optional OpenAI API key
            system_prompt: Optional system prompt
            with_citations: Whether to include citations
            
        Returns:
            Dict containing answer and metadata
        """
        # Use defaults if not provided
        model = model or settings.DEFAULT_LLM_MODEL
        with_citations = settings.ENABLE_CITATIONS if with_citations is None else with_citations
        
        # Use provided chunks or fall back to last retrieval results
        if chunks is None:
            if self.last_retrieval_results is None:
                return {
                    "answer": "No content has been retrieved yet. Please retrieve content first.",
                    "metadata": {
                        "error": "No retrieval results available"
                    }
                }
            chunks = self.last_retrieval_results
        
        if not chunks:
            return {
                "answer": "No relevant content was found to answer this question.",
                "metadata": {
                    "question": question,
                    "model": model
                }
            }
        
        # Format context from chunks with or without citations
        context, citations = self._format_context(chunks, with_citations)
        
        # Select appropriate system prompt if not provided
        if system_prompt is None:
            system_prompt = settings.SYSTEM_PROMPTS["with_citations"] if with_citations else settings.SYSTEM_PROMPTS["default"]

        logger.info(f"About to generate LLM response with model {model}")

        # Generate answer
        try:
            llm_response = self.llm_service.generate_response(
                question=question,
                context=context,
                model=model,
                system_prompt=system_prompt,
                openai_api_key=openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return {
                "answer": f"Fehler bei der Antwortgenerierung: {str(e)}",
                "metadata": {
                    "question": question,
                    "model": model,
                    "error": str(e)
                }
            }
        
        result = {
            "answer": llm_response["text"],
            "metadata": {
                "question": question,
                "model": llm_response.get("model", model)
            }
        }
        
        # Add citations if enabled
        if with_citations and citations:
            result["citations"] = citations
            
        return result
    
    def search(
        self,
        question: str,
        content_description: Optional[str] = None,
        year_range: Optional[List[int]] = None,
        chunk_size: Optional[int] = None,
        keywords: Optional[str] = None,
        search_in: Optional[List[str]] = None,
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        use_iterative_search: bool = False,
        time_window_size: int = 5,
        system_prompt: Optional[str] = None,
        min_relevance_score: float = 0.3,
        top_k: int = 10,
        use_query_refinement: Optional[bool] = None,
        with_citations: Optional[bool] = None,
        use_semantic_expansion: bool = True,
        semantic_expansion_factor: int = 3,
        enforce_keywords: bool = True,
        reuse_last_retrieval: bool = False
    ) -> Dict[str, Any]:
        """
        Combined method for backwards compatibility - performs both retrieval and analysis.
        
        Args:
            All parameters from both retrieve() and analyze() methods
            reuse_last_retrieval: Whether to reuse the last retrieval results
            
        Returns:
            Dict containing answer, chunks, and metadata
        """
        # Use query refinement from settings if not specified
        use_query_refinement = settings.ENABLE_QUERY_REFINEMENT if use_query_refinement is None else use_query_refinement
        
        # Decide whether to perform a new retrieval or reuse the last one
        if reuse_last_retrieval and self.last_retrieval_results is not None:
            chunks = self.last_retrieval_results
            retrieval_metadata = {
                "content_description": self.last_retrieval_params.get("content_description", "Unknown"),
                "chunk_size": self.last_retrieval_params.get("chunk_size", chunk_size or settings.DEFAULT_CHUNK_SIZE),
                "year_range": self.last_retrieval_params.get("year_range", year_range or [settings.MIN_YEAR, settings.MAX_YEAR]),
                "reused_retrieval": True
            }
        else:
            # Check if content description is provided when not reusing retrieval
            if not content_description and not reuse_last_retrieval:
                content_description = question  # Fall back to using question as content description
            
            # If query refinement is enabled, get better search queries
            refined_queries = []
            if use_query_refinement:
                try:
                    refined_queries = self._refine_search_query(content_description, question)
                    
                    # Use refined queries for retrieval
                    if refined_queries:
                        chunks = self._perform_refined_search(
                            refined_queries, 
                            content_description,
                            year_range or [settings.MIN_YEAR, settings.MAX_YEAR],
                            chunk_size or settings.DEFAULT_CHUNK_SIZE,
                            self.vector_store.build_metadata_filter(
                                year_range=year_range,
                                keywords=None,
                                search_in=None
                            ),
                            min_relevance_score,
                            top_k,
                            keywords=keywords,
                            search_in=search_in,
                            enforce_keywords=enforce_keywords
                        )
                        # Store the results for future reuse
                        self.last_retrieval_results = chunks
                        self.last_retrieval_params = {
                            "content_description": content_description,
                            "refined_queries": refined_queries,
                            "year_range": year_range or [settings.MIN_YEAR, settings.MAX_YEAR],
                            "chunk_size": chunk_size or settings.DEFAULT_CHUNK_SIZE,
                            "keywords": keywords,
                            "search_in": search_in
                        }
                    else:
                        # Perform regular retrieval
                        retrieval_result = self.retrieve(
                            content_description=content_description,
                            year_range=year_range,
                            chunk_size=chunk_size,
                            keywords=keywords,
                            search_in=search_in,
                            use_iterative_search=use_iterative_search,
                            time_window_size=time_window_size,
                            min_relevance_score=min_relevance_score,
                            top_k=top_k,
                            use_semantic_expansion=use_semantic_expansion,
                            semantic_expansion_factor=semantic_expansion_factor,
                            enforce_keywords=enforce_keywords
                        )
                        chunks = self.last_retrieval_results  # Use the stored results
                except Exception as e:
                    logger.error(f"Query refinement failed: {e}")
                    refined_queries = []
                    
                    # Fall back to regular retrieval
                    retrieval_result = self.retrieve(
                        content_description=content_description,
                        year_range=year_range,
                        chunk_size=chunk_size,
                        keywords=keywords,
                        search_in=search_in,
                        use_iterative_search=use_iterative_search,
                        time_window_size=time_window_size,
                        min_relevance_score=min_relevance_score,
                        top_k=top_k,
                        use_semantic_expansion=use_semantic_expansion,
                        semantic_expansion_factor=semantic_expansion_factor,
                        enforce_keywords=enforce_keywords
                    )
                    chunks = self.last_retrieval_results  # Use the stored results
            else:
                # Perform regular retrieval without refinement
                retrieval_result = self.retrieve(
                    content_description=content_description,
                    year_range=year_range,
                    chunk_size=chunk_size,
                    keywords=keywords,
                    search_in=search_in,
                    use_iterative_search=use_iterative_search,
                    time_window_size=time_window_size,
                    min_relevance_score=min_relevance_score,
                    top_k=top_k,
                    use_semantic_expansion=use_semantic_expansion,
                    semantic_expansion_factor=semantic_expansion_factor,
                    enforce_keywords=enforce_keywords
                )
                chunks = self.last_retrieval_results  # Use the stored results
            
            retrieval_metadata = {
                "content_description": content_description,
                "chunk_size": chunk_size or settings.DEFAULT_CHUNK_SIZE,
                "year_range": year_range or [settings.MIN_YEAR, settings.MAX_YEAR],
                "refined_queries": refined_queries if refined_queries else None,
                "reused_retrieval": False
            }
        
        # Perform analysis with retrieved chunks
        analysis_result = self.analyze(
            question=question,
            chunks=chunks,
            model=model,
            openai_api_key=openai_api_key,
            system_prompt=system_prompt,
            with_citations=with_citations
        )
        
        # Combine results
        result = {
            "answer": analysis_result["answer"],
            "chunks": self._format_chunks_for_response(chunks),
            "metadata": {
                **retrieval_metadata,
                **analysis_result.get("metadata", {})
            }
        }
        
        # Add citations if available
        if "citations" in analysis_result:
            result["citations"] = analysis_result["citations"]
            
        return result
    
    def _format_chunks_for_response(self, chunks: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """Format chunks for the API response."""
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            }
            for doc, score in chunks
        ]
    
    def _format_context(self, chunks: List[Tuple[Document, float]], with_citations: bool = False) -> Tuple[str, List[str]]:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            chunks: List of (Document, score) tuples
            with_citations: Whether to format for citations
            
        Returns:
            Tuple of (formatted_context, citations)
        """
        context_list, citation_list = self.vector_store.format_search_results(
            chunks, with_citations=with_citations
        )
        
        return "\n\n".join(context_list), citation_list
        
    def _refine_search_query(self, original_query: str, question: str) -> List[str]:
        """
        Use LLM to refine the search query for better results.
        
        Args:
            original_query: Original search query
            question: User's question
            
        Returns:
            List of refined search queries
        """
        logger.info(f"Refining search query: {original_query}")
        
        # Get a few initial results to provide context
        try:
            initial_results = self.vector_store.similarity_search(
                original_query,
                1200,  # Use a medium chunk size for initial search
                k=5,
                min_relevance_score=0.3
            )
        except Exception as e:
            logger.error(f"Error in initial search for query refinement: {e}")
            initial_results = []
        
        # Format context for the refinement prompt
        context, _ = self._format_context(initial_results)
        
        # Create a prompt for query refinement
        preprompt = f"""Die aktuelle Anfrage lautete:
```
{original_query}
```
und ergibt die nachfolgenden Textauszüge.
"""
        
        postprompt = f"""Formuliere nun den neuen Text für die Quellen-Anfrage. Beantworte NICHT die Fragen."""
        
        # Generate refined queries
        llm_response = self.llm_service.generate_response(
            question=question,
            context=context,
            system_prompt=settings.SYSTEM_PROMPTS["query_refinement"],
            preprompt=preprompt,
            postprompt=postprompt,
            temperature=0.7
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response text
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, llm_response["text"], re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                queries = json_data.get("queries", [])
                
                if queries:
                    logger.info(f"Generated {len(queries)} refined queries")
                    return queries
            
            # Fallback to original query if JSON parsing fails
            logger.warning("Failed to parse refined queries JSON, using original query")
            return [original_query]
        except Exception as e:
            logger.error(f"Error parsing refined queries: {e}")
            return [original_query]
    
    def _perform_refined_search(
        self,
        refined_queries: List[str],
        original_query: str,
        year_range: List[int],
        chunk_size: int,
        filter_dict: Optional[Dict] = None,
        min_relevance_score: float = 0.3,
        top_k: int = 10,
        keywords: Optional[str] = None,
        search_in: Optional[List[str]] = None,
        enforce_keywords: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Perform search with multiple refined queries.
        
        Args:
            refined_queries: List of refined search queries
            original_query: Original search query (fallback)
            year_range: Range of years to search
            chunk_size: Size of chunks to search
            filter_dict: Optional filter dictionary
            min_relevance_score: Minimum relevance score
            top_k: Maximum number of results to return
            keywords: Optional boolean expression for keyword filtering
            search_in: Where to search for keywords
            enforce_keywords: Whether to strictly enforce keyword presence
            
        Returns:
            Combined and deduplicated results
        """
        all_results = []
        seen_ids = set()
        
        # Calculate results per query, keeping at least 1 result per query
        results_per_query = max(1, top_k // (len(refined_queries) + 1))
        
        # First search with all refined queries
        for query in refined_queries:
            logger.info(f"Searching with refined query: {query[:50]}...")
            
            try:
                results = self.vector_store.similarity_search(
                    query,
                    chunk_size,
                    k=results_per_query,
                    filter_dict=filter_dict,
                    min_relevance_score=min_relevance_score,
                    keywords=keywords,
                    search_in=search_in,
                    enforce_keywords=enforce_keywords
                )
                
                # Add unique results to the combined list
                for doc, score in results:
                    # Use ID from metadata if available, otherwise use content hash
                    doc_id = doc.metadata.get('id_chunk') or hash(doc.page_content)
                    
                    if doc_id not in seen_ids:
                        all_results.append((doc, score))
                        seen_ids.add(doc_id)
            except Exception as e:
                logger.error(f"Error searching with refined query: {e}")
        
        # Also search with the original query
        try:
            logger.info(f"Searching with original query: {original_query[:50]}...")
            original_results = self.vector_store.similarity_search(
                original_query,
                chunk_size,
                k=results_per_query,
                filter_dict=filter_dict,
                min_relevance_score=min_relevance_score,
                keywords=keywords,
                search_in=search_in,
                enforce_keywords=enforce_keywords
            )
            
            # Add unique results from original query
            for doc, score in original_results:
                doc_id = doc.metadata.get('id_chunk') or hash(doc.page_content)
                
                if doc_id not in seen_ids:
                    all_results.append((doc, score))
                    seen_ids.add(doc_id)
        except Exception as e:
            logger.error(f"Error searching with original query: {e}")
        
        # Sort by relevance and take top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    def _perform_iterative_search(
        self,
        query: str,
        year_range: List[int],
        chunk_size: int,
        window_size: int,
        filter_dict: Optional[Dict] = None,
        min_relevance_score: float = 0.3,
        top_k: int = 10,
        keywords: Optional[str] = None,
        search_in: Optional[List[str]] = None,
        enforce_keywords: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Perform iterative search across time windows.
        
        Args:
            query: Search query
            year_range: Range of years to search
            chunk_size: Size of chunks to search
            window_size: Size of time windows
            filter_dict: Base filter dictionary
            min_relevance_score: Minimum relevance score
            top_k: Number of top results to retrieve
            keywords: Optional boolean expression for keyword filtering
            search_in: Where to search for keywords
            enforce_keywords: Whether to strictly enforce keyword presence
            
        Returns:
            Combined results across time windows
        """
        start_year, end_year = year_range
        all_results = []
        
        # Create time windows
        windows = []
        for window_start in range(start_year, end_year + 1, window_size):
            window_end = min(window_start + window_size - 1, end_year)
            windows.append([window_start, window_end])
        
        logger.info(f"Performing iterative search with {len(windows)} time windows")
        
        # Calculate results per window
        results_per_window = max(1, top_k // len(windows))
        
        # Search each window
        for window in windows:
            window_start, window_end = window
            
            try:
                # Create window-specific filter
                window_filter = self.vector_store.build_metadata_filter(
                    year_range=[window_start, window_end],
                    keywords=None,
                    search_in=None
                )
                
                window_results = self.vector_store.similarity_search(
                    query,
                    chunk_size,
                    k=results_per_window,
                    filter_dict=window_filter,
                    min_relevance_score=min_relevance_score,
                    keywords=keywords,
                    search_in=search_in,
                    enforce_keywords=enforce_keywords
                )
                
                all_results.extend(window_results)
            except Exception as e:
                logger.error(f"Error searching time window {window}: {e}")
        
        # Sort by relevance and take top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    def find_similar_words(self, word: str, top_n: int = 10) -> List[Dict[str, Union[str, float, int]]]:
        """
        Find similar words using word embeddings.
        
        Args:
            word: Query word
            top_n: Number of similar words to return
            
        Returns:
            List of similar words with metadata
        """
        if not self._has_embedding_service:
            logger.warning("Word embedding service not available")
            return []
            
        try:
            return self.embedding_service.find_similar_words(word, top_n)
        except Exception as e:
            logger.error(f"Error finding similar words: {e}")
            return []