"""
FastAPI app for Spiegel RAG API.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.core.rag_engine import SpiegelRAGEngine
from src.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI(
    title="Spiegel RAG API",
    description="API for searching and analyzing Der Spiegel archives (1948-1979)",
    version="0.1.0"
)

# Initialize RAG engine
rag_engine = SpiegelRAGEngine()

# Define API models
class SearchRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the retrieved content")
    content_description: Optional[str] = Field(None, description="Description of content to retrieve")
    year_range: List[int] = Field([settings.MIN_YEAR, settings.MAX_YEAR], description="Range of years to search [start, end]")
    chunk_size: int = Field(settings.DEFAULT_CHUNK_SIZE, description="Size of chunks to search in")
    keywords: Optional[str] = Field(None, description="Boolean search expression (with AND, OR, NOT)")
    search_in: Optional[List[str]] = Field(["content"], description="Where to search (content, title, etc.)")
    model: str = Field(settings.DEFAULT_LLM_MODEL, description="LLM model to use")
    use_iterative_search: bool = Field(False, description="Whether to use iterative time window search")
    time_window_size: int = Field(5, description="Size of time windows for iterative search")
    system_prompt_key: Optional[str] = Field(None, description="Key for predefined system prompt")
    custom_system_prompt: Optional[str] = Field(None, description="Custom system prompt text (overrides system_prompt_key)")
    use_semantic_expansion: bool = Field(True, description="Expand keywords with semantically similar terms")
    semantic_expansion_factor: int = Field(3, description="Number of similar words to add per term")
    enforce_keywords: bool = Field(True, description="Strictly enforce keyword presence in results")

class KeywordExpansionRequest(BaseModel):
    keywords: str = Field(..., description="Boolean expression of keywords to expand")
    expansion_factor: int = Field(3, description="Number of similar words to add per keyword")

# Define API routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Spiegel RAG API",
        "version": "0.1.0",
        "endpoints": [
            "/search",
            "/similar-words",
            "/health"
        ]
    }

@app.post("/search")
async def search(request: SearchRequest):
    """
    Search the Spiegel archive and generate an answer.
    """
    try:
        # Determine which system prompt to use
        system_prompt = None
        
        # Custom system prompt takes precedence if provided
        if request.custom_system_prompt:
            system_prompt = request.custom_system_prompt
        # Otherwise use the predefined one if specified
        elif request.system_prompt_key:
            system_prompt = settings.SYSTEM_PROMPTS.get(
                request.system_prompt_key,
                settings.SYSTEM_PROMPTS["default"]
            )
        
        # Perform search with custom system prompt
        results = rag_engine.search(
            question=request.question,
            content_description=request.content_description,
            year_range=request.year_range,
            chunk_size=request.chunk_size,
            keywords=request.keywords,
            search_in=request.search_in,
            model=request.model,
            use_iterative_search=request.use_iterative_search,
            time_window_size=request.time_window_size,
            system_prompt=system_prompt,
            use_semantic_expansion=request.use_semantic_expansion,
            semantic_expansion_factor=request.semantic_expansion_factor,
            enforce_keywords=request.enforce_keywords
        )
        
        return results
    except Exception as e:
        logging.error(f"Error in search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar-words/{word}")
async def similar_words(word: str, top_n: int = 10):
    """
    Find similar words based on word embeddings.
    """
    try:
        similar_words = rag_engine.find_similar_words(word, top_n)
        return {"similar_words": similar_words}
    except Exception as e:
        logging.error(f"Error finding similar words: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    components_status = {
        "api": "healthy",
        "vector_store": "unknown",
        "llm_service": "unknown",
        "embedding_service": "unknown"
    }
    
    # Check vector store
    try:
        # Just try to get a vectorstore
        rag_engine.vector_store.get_vectorstore(settings.DEFAULT_CHUNK_SIZE)
        components_status["vector_store"] = "healthy"
    except Exception as e:
        components_status["vector_store"] = f"unhealthy: {str(e)}"
    
    # Check embedding service
    try:
        # Try to get a word vector
        if rag_engine.embedding_service.get_word_vector("deutschland") is not None:
            components_status["embedding_service"] = "healthy"
        else:
            components_status["embedding_service"] = "unhealthy: model loaded but not working"
    except Exception as e:
        components_status["embedding_service"] = f"unhealthy: {str(e)}"
    
    # Check LLM service (just check if initialized)
    if rag_engine.llm_service.hu_llm_models is not None:
        components_status["llm_service"] = "healthy"
    else:
        components_status["llm_service"] = "unhealthy: models not available"
    
    # Overall status
    overall_status = "healthy"
    if any(status.startswith("unhealthy") for status in components_status.values()):
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "components": components_status
    }

@app.post("/expand-keywords")
async def expand_keywords(request: KeywordExpansionRequest):
    """
    Expand keywords with semantically similar terms based on word embeddings.
    """
    try:
        # Parse boolean expression
        parsed_terms = rag_engine.embedding_service.parse_boolean_expression(request.keywords)
        
        # Expand terms with semantically similar words
        expanded_terms = rag_engine.embedding_service.filter_by_semantic_similarity(
            parsed_terms, 
            expansion_factor=request.expansion_factor
        )
        
        return {
            "original_keywords": request.keywords,
            "parsed_terms": parsed_terms,
            "expanded_terms": expanded_terms
        }
    except Exception as e:
        logging.error(f"Error expanding keywords: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/word-frequency/{word}")
async def word_frequency(word: str):
    """
    Get the frequency of a word in the corpus.
    """
    try:
        frequency = rag_engine.embedding_service.get_word_frequency(word)
        return {"word": word, "frequency": frequency}
    except Exception as e:
        logging.error(f"Error getting word frequency: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# Run the app (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)