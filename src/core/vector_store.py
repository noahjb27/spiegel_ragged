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
        Perform similarity search with relevance scores and strict keyword filtering.
        
        Args:
            query: Query text
            chunk_size: Size of chunks to search in
            k: Number of results to return
            filter_dict: Optional filter dictionary for metadata filtering
            min_relevance_score: Minimum relevance score for results
            keywords: Optional boolean expression for keyword filtering
            search_in: Where to search for keywords
            enforce_keywords: Whether to strictly enforce keyword presence
            
        Returns:
            List of (Document, score) tuples
        """
        vectorstore = self.get_vectorstore(chunk_size)
        
        # If we don't need to enforce keywords, just use the standard search
        if not enforce_keywords or not keywords:
            try:
                results = vectorstore.similarity_search_with_relevance_scores(
                    query, k=k*2, filter=filter_dict  # Get more results to compensate for post-filtering
                )
                
                # Filter by minimum relevance score
                filtered_results = [
                    (doc, score) for doc, score in results
                    if score >= min_relevance_score
                ]
                
                return filtered_results[:k]
            except Exception as e:
                logger.error(f"Error in similarity search: {e}")
                raise
        
        # If we're enforcing keywords, we need to do post-filtering
        try:
            # Get more results since we'll be filtering some out
            results = vectorstore.similarity_search_with_relevance_scores(
                query, k=k*5, filter=filter_dict  # Get many more results than needed
            )
            
            # Parse the boolean expression
            if hasattr(self, 'embedding_service') and self.embedding_service:
                parsed_query = self.embedding_service.parse_boolean_expression(keywords)
            else:
                # Simple parsing if no embedding service
                parsed_query = {
                    "must": [term.strip() for term in keywords.split('AND')],
                    "should": [],
                    "must_not": []
                }
            
            # Post-filter based on keyword criteria
            hard_filtered_results = []
            
            for doc, score in results:
                if score < min_relevance_score:
                    continue
                    
                # Default to checking the content
                text_to_check = doc.page_content.lower()
                title_to_check = doc.metadata.get('Artikeltitel', '').lower()
                keywords_to_check = doc.metadata.get('Schlagworte', '').lower()
                
                # Determine which fields to check
                check_text = not search_in or 'Text' in search_in
                check_title = search_in and 'Artikeltitel' in search_in
                check_keywords = search_in and 'Schlagworte' in search_in
                
                # Check MUST terms (all must be present)
                must_present = True
                for term in parsed_query["must"]:
                    term_lower = term.lower()
                    term_found = False
                    
                    if check_text and term_lower in text_to_check:
                        term_found = True
                    elif check_title and term_lower in title_to_check:
                        term_found = True
                    elif check_keywords and term_lower in keywords_to_check:
                        term_found = True
                        
                    if not term_found:
                        must_present = False
                        break
                
                if not must_present:
                    continue
                
                # Check SHOULD terms (at least one must be present if there are any)
                should_present = True
                if parsed_query["should"]:
                    should_present = False
                    for term in parsed_query["should"]:
                        term_lower = term.lower()
                        
                        if check_text and term_lower in text_to_check:
                            should_present = True
                            break
                        elif check_title and term_lower in title_to_check:
                            should_present = True
                            break
                        elif check_keywords and term_lower in keywords_to_check:
                            should_present = True
                            break
                
                if not should_present:
                    continue
                    
                # Check MUST NOT terms (none must be present)
                must_not_absent = True
                for term in parsed_query["must_not"]:
                    term_lower = term.lower()
                    
                    if (check_text and term_lower in text_to_check) or \
                    (check_title and term_lower in title_to_check) or \
                    (check_keywords and term_lower in keywords_to_check):
                        must_not_absent = False
                        break
                
                if not must_not_absent:
                    continue
                    
                # If we get here, the document passed all keyword filters
                hard_filtered_results.append((doc, score))
                
                # Stop if we have enough results
                if len(hard_filtered_results) >= k:
                    break
            
            logger.info(
                f"Query: '{query}' returned {len(hard_filtered_results)} results after hard keyword filtering"
            )
            
            return hard_filtered_results
        
        except Exception as e:
            logger.error(f"Error in keyword-filtered search: {e}")
            # Fallback to normal search if keyword filtering fails
            logger.info("Falling back to standard search without keyword enforcement")
            return self.similarity_search(
                query, chunk_size, k, filter_dict, min_relevance_score, 
                keywords=None, enforce_keywords=False
        )
            
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