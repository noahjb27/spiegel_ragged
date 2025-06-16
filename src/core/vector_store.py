# src/core/vector_store.py
"""
ChromaDB interface for Spiegel RAG.
Connects to a remote ChromaDB instance with Ollama embeddings.
FIXED VERSION - addresses parameter handling and search issues
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
        """Initialize remote ChromaDB interface."""
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
    
    def get_collection_name(self, chunk_size: int) -> str:
        """Generate standardized collection name."""
        # Use the exact format that exists database
        overlap_map = {500: 100, 2000: 400, 3000: 300}
        overlap = overlap_map.get(chunk_size, 300)  # Default to 300 if unknown
        return f"recursive_chunks_{chunk_size}_{overlap}_TH_cosine_{settings.OLLAMA_MODEL_NAME}"
    
    def get_vectorstore(self, chunk_size: int, chunk_overlap: Optional[int] = None) -> Chroma:
        """Get a vectorstore for a specific chunk size."""
        # Use standardized collection name
        collection_name = self.get_collection_name(chunk_size)
        
        # Check cache first
        if collection_name in self._vectorstore_cache:
            logger.debug(f"Using cached vectorstore for {collection_name}")
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
    
    def clean_parameter(self, param: Any) -> Optional[str]:
        """Safely clean UI parameters that might be None, 'None', '', etc."""
        if param is None:
            return None
        if isinstance(param, str):
            param = param.strip()
            # Be more specific about what we consider "empty"
            if param == '' or param.lower() in ['none', 'null', 'undefined']:
                return None
            return param
        return None
    
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
        FIXED VERSION with better parameter handling and error reporting.
        """
        
        # Clean the keywords parameter safely
        cleaned_keywords = self.clean_parameter(keywords)
        
        logger.info(f"=== SIMILARITY SEARCH DEBUG ===")
        logger.info(f"Query: '{query}'")
        logger.info(f"Chunk size: {chunk_size}")
        logger.info(f"k: {k}")
        logger.info(f"Keywords (raw): {repr(keywords)} -> (cleaned): {repr(cleaned_keywords)}")
        logger.info(f"Enforce keywords: {enforce_keywords}")
        logger.info(f"Min relevance score: {min_relevance_score}")
        logger.info(f"Filter dict: {filter_dict}")
        
        try:
            vectorstore = self.get_vectorstore(chunk_size)
            logger.info(f"Got vectorstore for collection: {self.get_collection_name(chunk_size)}")
            
            # Determine if we should apply keyword filtering
            should_filter_keywords = (
                enforce_keywords and 
                cleaned_keywords is not None and
                len(cleaned_keywords) > 0
            )
            
            logger.info(f"Should apply keyword filtering: {should_filter_keywords}")
            
            if not should_filter_keywords:
                # Standard search without keyword filtering
                logger.info("Performing standard search (no keyword filtering)")
                try:
                    # Get more results initially for better filtering
                    search_k = max(k * 2, 20)  # Get at least 20 results
                    
                    results = vectorstore.similarity_search_with_relevance_scores(
                        query, k=search_k, filter=filter_dict
                    )
                    
                    logger.info(f"Raw search returned {len(results)} results")
                    
                    # Filter by minimum relevance score
                    filtered_results = [
                        (doc, score) for doc, score in results
                        if score >= min_relevance_score
                    ]
                    
                    logger.info(f"After relevance filtering: {len(filtered_results)} results")
                    
                    # Return top k results
                    final_results = filtered_results[:k]
                    logger.info(f"Final results: {len(final_results)}")
                    
                    # Log some details about the results
                    if final_results:
                        logger.info("Sample results:")
                        for i, (doc, score) in enumerate(final_results[:3]):
                            logger.info(f"  {i+1}. Score: {score:.4f}, Title: {doc.metadata.get('Artikeltitel', 'No title')}")
                    
                    return final_results
                    
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
            # Don't raise, return empty list with detailed error message
            logger.error(f"SEARCH FAILED COMPLETELY - Check:")
            logger.error(f"  1. ChromaDB connection: {settings.CHROMA_DB_HOST}:{settings.CHROMA_DB_PORT}")
            logger.error(f"  2. Collection exists: {self.get_collection_name(chunk_size)}")
            logger.error(f"  3. Ollama embedding service: {settings.OLLAMA_BASE_URL}")
            return []

    def _parse_keywords_safely(self, keywords: str) -> Dict[str, List[str]]:
        """Safely parse keywords with fallback to simple parsing."""
        if not keywords:
            return {"must": [], "should": [], "must_not": []}
            
        try:
            # Simple fallback parsing for AND/OR/NOT
            logger.info("Using simple keyword parsing")
            
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
    
    # Keep your existing methods for formatting and building filters
    def format_search_results(
        self,
        results: List[Tuple[Document, float]],
        with_citations: bool = False,
        start_index: int = 0
    ) -> Tuple[List[str], List[str]]:
        """Format search results for use in prompts and display."""
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
        """Build a metadata filter for ChromaDB with enhanced keyword support."""
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
        
        # If no conditions were added, return None
        if not where_conditions:
            return None
            
        # If there's only one condition, return it directly
        if len(where_conditions) == 1:
            return where_conditions[0]
            
        # Otherwise, combine conditions with $and
        return {"$and": where_conditions}