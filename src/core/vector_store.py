"""
ChromaDB interface for Spiegel RAG.
Connects to a remote ChromaDB instance with Ollama embeddings.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import chromadb
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)

class ChromaDBInterface:
    """Interface for remote ChromaDB operations."""
    
    def __init__(self):
        """
        Initialize remote ChromaDB interface.
        """
        # Initialize the Ollama embedding model
        self.embedding_model = OllamaEmbeddings(
            model=settings.OLLAMA_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # Initialize the ChromaDB client
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_DB_HOST,
                port=settings.CHROMA_DB_PORT,
                ssl=settings.CHROMA_DB_SSL
            )
            logger.info(f"Connected to remote ChromaDB at {settings.CHROMA_DB_HOST}:{settings.CHROMA_DB_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to remote ChromaDB: {e}")
            raise
        
        # Cache for vectorstores to avoid reloading
        self._vectorstore_cache: Dict[str, Chroma] = {}
    
    def get_vectorstore(self, chunk_size: int, chunk_overlap: Optional[int] = None) -> Chroma:
        """
        Get a vectorstore for a specific chunk size.
        
        Args:
            chunk_size: Size of chunks in the collection
            chunk_overlap: Overlap size (defaults to specific values based on chunk size)
            
        Returns:
            Chroma: The vectorstore instance
        """
        # Behandle spezielle Fälle für die zwei verfügbaren Sammlungen
        if chunk_size == 2000:
            chunk_overlap = 400  # Spezieller Fall für 2000 chunk size
        elif chunk_size == 3000:
            chunk_overlap = 300  # Spezieller Fall für 3000 chunk size
        else:
            # Fallback zu einer der verfügbaren Größen
            nearest_size = min(settings.AVAILABLE_CHUNK_SIZES, key=lambda x: abs(x - chunk_size))
            logger.warning(f"Chunk size {chunk_size} not available, using nearest size {nearest_size}")
            chunk_size = nearest_size
            chunk_overlap = 400 if chunk_size == 2000 else 300
            
        collection_name = f"recursive_chunks_{chunk_size}_{chunk_overlap}_TH_cosine_nomic-embed-text"
        
        # Check cache first
        if collection_name in self._vectorstore_cache:
            return self._vectorstore_cache[collection_name]
        
        # Load vectorstore if not in cache
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                client=self.client
            )
            self._vectorstore_cache[collection_name] = vectorstore
            logger.info(f"Loaded vectorstore for collection: {collection_name}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading vectorstore for collection {collection_name}: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        chunk_size: int, 
        k: int = 5,
        filter_dict: Optional[Dict] = None,
        min_relevance_score: float = 0.3,
        keywords: Optional[str] = None,
        search_in: Optional[List[str]] = None,
        enforce_keywords: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores and keyword filtering.
        Robust handling of UI parameter conversion issues.
        """
        
        # ROBUST PARAMETER CLEANING - Handle UI conversion issues
        def clean_keyword_param(kw):
            """Clean keyword parameter from potential UI conversion issues."""
            if kw is None:
                return None
            if isinstance(kw, str):
                kw = kw.strip()
                # Handle common UI conversion issues
                if kw == '' or kw.lower() in ['none', 'null', 'undefined']:
                    return None
                return kw
            return None  # Fallback for unexpected types
        
        # Clean the keywords parameter
        cleaned_keywords = clean_keyword_param(keywords)
        
        logger.info(f"Starting similarity search for query: '{query}', k={k}")
        logger.info(f"Keywords (raw): {repr(keywords)} -> (cleaned): {repr(cleaned_keywords)}")
        logger.info(f"Enforce keywords: {enforce_keywords}")
        
        try:
            vectorstore = self.get_vectorstore(chunk_size)
            logger.info("Got vectorstore, performing search")
            
            # Determine if we should apply keyword filtering
            should_filter_keywords = (
                enforce_keywords and 
                cleaned_keywords is not None and
                len(cleaned_keywords) > 0
            )
            
            logger.info(f"Should apply keyword filtering: {should_filter_keywords}")
            
            if not should_filter_keywords:
                # Standard search without keyword filtering
                logger.info(f"Performing standard search (no keyword filtering)")
                try:
                    results = vectorstore.similarity_search_with_relevance_scores(
                        query, k=k*2, filter=filter_dict  # Get extra results for filtering
                    )
                    
                    # Filter by minimum relevance score
                    filtered_results = [
                        (doc, score) for doc, score in results
                        if score >= min_relevance_score
                    ]
                    
                    logger.info(f"Standard search returned {len(filtered_results)} results after relevance filtering")
                    return filtered_results[:k]  # Limit to requested amount
                    
                except Exception as e:
                    logger.error(f"Error in standard similarity search: {e}", exc_info=True)
                    raise
            
            # Keyword filtering path
            logger.info(f"Performing keyword-filtered search with keywords: '{cleaned_keywords}'")
            
            # Get more results for filtering
            search_multiplier = max(3, min(10, k * 2))
            search_k = k * search_multiplier
            
            logger.info(f"Retrieving {search_k} results for keyword filtering")
            
            try:
                results = vectorstore.similarity_search_with_relevance_scores(
                    query, k=search_k, filter=filter_dict
                )
                
                logger.info(f"Raw search returned {len(results)} results before keyword filtering")
                
                if not results:
                    logger.warning("No results returned from vector search")
                    return []
                
                # Parse keywords safely
                parsed_query = self._parse_keywords_safely(cleaned_keywords)
                logger.info(f"Parsed keywords: {parsed_query}")
                
                # Apply keyword filtering
                filtered_results = self._apply_keyword_filter(
                    results, parsed_query, search_in or ["Text"], min_relevance_score
                )
                
                logger.info(f"Keyword filtering returned {len(filtered_results)} final results")
                return filtered_results[:k]
                
            except Exception as e:
                logger.error(f"Error in keyword-filtered search: {e}", exc_info=True)
                # Fallback to standard search
                logger.info("Falling back to standard search due to keyword filtering error")
                return self.similarity_search(
                    query, chunk_size, k, filter_dict, min_relevance_score, 
                    keywords=None, enforce_keywords=False
                )
                
        except Exception as e:
            logger.error(f"Critical error in similarity_search: {e}", exc_info=True)
            raise

    def _parse_keywords_safely(self, keywords: str) -> Dict[str, List[str]]:
        """Safely parse keywords with fallback to simple parsing."""
        if not keywords:
            return {"must": [], "should": [], "must_not": []}
            
        try:
            # Try to use embedding service if available
            if hasattr(self, 'embedding_service') and self.embedding_service:
                return self.embedding_service.parse_boolean_expression(keywords)
            else:
                # Simple fallback parsing for AND/OR/NOT
                logger.info("Using simple keyword parsing (no embedding service)")
                
                # Handle NOT terms first
                parts = keywords.split(' NOT ')
                main_expr = parts[0].strip()
                must_not = [p.strip() for p in parts[1:] if p.strip()] if len(parts) > 1 else []
                
                # Handle OR terms
                if ' OR ' in main_expr:
                    should = [p.strip() for p in main_expr.split(' OR ') if p.strip()]
                    must = []
                # Handle AND terms  
                elif ' AND ' in main_expr:
                    must = [p.strip() for p in main_expr.split(' AND ') if p.strip()]
                    should = []
                # Single term
                else:
                    must = [main_expr.strip()] if main_expr.strip() else []
                    should = []
                
                result = {"must": must, "should": should, "must_not": must_not}
                logger.info(f"Simple parsing result: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Error parsing keywords '{keywords}': {e}")
            # Ultimate fallback - treat as single must term
            return {"must": [keywords.strip()], "should": [], "must_not": []}

    def _apply_keyword_filter(
        self, 
        results: List[Tuple[Document, float]], 
        parsed_query: Dict[str, List[str]], 
        search_in: List[str], 
        min_relevance_score: float
    ) -> List[Tuple[Document, float]]:
        """Apply keyword filtering to search results."""
        
        # Quick exit if no filtering criteria
        if not any([parsed_query["must"], parsed_query["should"], parsed_query["must_not"]]):
            logger.info("No keyword terms to filter by, returning relevance-filtered results")
            return [(doc, score) for doc, score in results if score >= min_relevance_score]
        
        filtered_results = []
        
        logger.info(f"Applying keyword filter to {len(results)} results")
        logger.info(f"Search fields: {search_in}")
        logger.info(f"Filter criteria: {parsed_query}")
        
        for doc, score in results:
            if score < min_relevance_score:
                continue
                
            # Get text fields to search in
            texts_to_search = []
            if "Text" in search_in:
                texts_to_search.append(doc.page_content.lower())
            if "Artikeltitel" in search_in and "Artikeltitel" in doc.metadata:
                texts_to_search.append(doc.metadata.get('Artikeltitel', '').lower())
            if "Schlagworte" in search_in and "Schlagworte" in doc.metadata:
                texts_to_search.append(doc.metadata.get('Schlagworte', '').lower())
            
            # Check if document matches keyword criteria
            if self._document_matches_keywords(texts_to_search, parsed_query):
                filtered_results.append((doc, score))
                # Early exit if we have enough results
                if len(filtered_results) >= 50:  # Reasonable limit
                    break
        
        logger.info(f"Keyword filtering kept {len(filtered_results)} out of {len(results)} results")
        return filtered_results

    def _document_matches_keywords(
        self, 
        texts_to_search: List[str], 
        parsed_query: Dict[str, List[str]]
    ) -> bool:
        """Check if a document matches the keyword criteria."""
        
        # Check MUST terms (all must be present)
        for term in parsed_query["must"]:
            term_lower = term.lower()
            term_found = any(term_lower in text for text in texts_to_search)
            if not term_found:
                return False
        
        # Check SHOULD terms (at least one must be present, if any exist)
        if parsed_query["should"]:
            should_found = any(
                term.lower() in text 
                for term in parsed_query["should"] 
                for text in texts_to_search
            )
            if not should_found:
                return False
        
        # Check MUST NOT terms (none must be present)
        for term in parsed_query["must_not"]:
            term_lower = term.lower()
            if any(term_lower in text for text in texts_to_search):
                return False
        
        return True
    
    def format_search_results(
        self,
        results: List[Tuple[Document, float]],
        with_citations: bool = False,
        start_index: int = 0
    ) -> Tuple[List[str], List[str]]:
        """
        Format search results for use in prompts and display.
        
        Args:
            results: List of (Document, score) tuples from similarity_search
            with_citations: Whether to format for citation references
            start_index: Starting index for citations
            
        Returns:
            Tuple of (context_list, citation_list)
        """
        context_list = []
        citation_list = []
        
        for i, (doc, score) in enumerate(results):
            idx = start_index + i + 1
            
            # Create citation reference
            citation = f"[{idx}]: {doc.metadata.get('Artikeltitel', 'Kein Titel')} - {doc.metadata.get('Datum', 'Kein Datum')} - {doc.metadata.get('URL', 'Keine URL')}"
            citation_list.append(citation)
            
            # Create context entry
            if with_citations:
                context = f"[{idx}]: Titel: {doc.metadata.get('Artikeltitel', 'Kein Titel')} (Datum: {doc.metadata.get('Datum', 'Kein Datum')}) \nURL: {doc.metadata.get('URL', 'Keine URL')} \n{doc.page_content}"
            else:
                context = f"Titel: {doc.metadata.get('Artikeltitel', 'Kein Titel')} (Datum: {doc.metadata.get('Datum', 'Kein Datum')}) \nURL: {doc.metadata.get('URL', 'Keine URL')} \n{doc.page_content}"
            
            context_list.append(context)
        
        return context_list, citation_list
        
    def build_metadata_filter(
        self,
        year_range: Optional[List[int]] = None,
        keywords: Optional[str] = None,
        search_in: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        Build a metadata filter for ChromaDB with enhanced keyword support.
        
        Args:
            year_range: Range of years [start, end]
            keywords: Boolean expression of keywords (e.g. "berlin AND (wall OR mauer) NOT soviet")
            search_in: Where to search for keywords (e.g. "Artikeltitel", "Text")
            
        Returns:
            Filter dictionary or None if no filters
        """
        where_conditions = []
        
        # Add year range filter if provided
        if year_range and len(year_range) == 2:
            start_year, end_year = year_range
            # Create proper ChromaDB format for range queries
            year_filter = {"$and": [
                {"Jahrgang": {"$gte": start_year}},
                {"Jahrgang": {"$lte": end_year}}
            ]}
            where_conditions.append(year_filter)
        
        # Add keyword filter if provided
        if keywords and isinstance(keywords, str):
            try:
                # This would use the WordEmbeddingService.parse_boolean_expression method
                if hasattr(self, 'embedding_service') and self.embedding_service:
                    parsed_query = self.embedding_service.parse_boolean_expression(keywords)
                    
                    keyword_conditions = []
                    
                    # Process MUST terms
                    for term in parsed_query["must"]:
                        if search_in:
                            field_conditions = []
                            for field in search_in:
                                field_conditions.append({field: {"$contains": term}})
                            if field_conditions:
                                keyword_conditions.append({"$or": field_conditions})
                        else:
                            # Default to searching in Text if no fields specified
                            keyword_conditions.append({"Text": {"$contains": term}})
                    
                    # Process SHOULD terms (at least one should match)
                    should_conditions = []
                    for term in parsed_query["should"]:
                        if search_in:
                            for field in search_in:
                                should_conditions.append({field: {"$contains": term}})
                        else:
                            should_conditions.append({"Text": {"$contains": term}})
                    
                    if should_conditions:
                        keyword_conditions.append({"$or": should_conditions})
                    
                    # Process MUST NOT terms
                    for term in parsed_query["must_not"]:
                        if search_in:
                            field_conditions = []
                            for field in search_in:
                                field_conditions.append({field: {"$not": {"$contains": term}}})
                            if field_conditions:
                                keyword_conditions.append({"$and": field_conditions})
                        else:
                            keyword_conditions.append({"Text": {"$not": {"$contains": term}}})
                    
                    if keyword_conditions:
                        # If we have multiple conditions, combine with $and
                        if len(keyword_conditions) > 1:
                            where_conditions.append({"$and": keyword_conditions})
                        else:
                            where_conditions.append(keyword_conditions[0])
                else:
                    # Fallback to simple keyword matching without boolean parsing
                    if search_in:
                        field_conditions = []
                        for field in search_in:
                            field_conditions.append({field: {"$contains": keywords}})
                        if field_conditions:
                            where_conditions.append({"$or": field_conditions})
                    else:
                        where_conditions.append({"Text": {"$contains": keywords}})
            except Exception as e:
                logger.error(f"Error parsing keywords '{keywords}': {e}")
                # Add simple contains filter as fallback
                where_conditions.append({"Text": {"$contains": keywords}})
        
        # If no conditions were added, return None
        if not where_conditions:
            return None
            
        # If there's only one condition, return it directly
        if len(where_conditions) == 1:
            return where_conditions[0]
            
        # Otherwise, combine conditions with $and
        return {"$and": where_conditions}