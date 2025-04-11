# src/utils/component_test.py
import os
import sys
import logging
from typing import Dict, List, Any
import time

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.vector_store import ChromaDBInterface
from src.core.llm_service import LLMService
from src.core.embedding_service import WordEmbeddingService
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_chroma_connection():
    """Test connection to ChromaDB and basic vector search."""
    logger.info("Testing ChromaDB connection...")
    try:
        vector_store = ChromaDBInterface()
        logger.info("✅ ChromaDB connection successful")
        
        # Test getting a vectorstore
        for chunk_size in settings.AVAILABLE_CHUNK_SIZES:
            try:
                start_time = time.time()
                vs = vector_store.get_vectorstore(chunk_size)
                logger.info(f"✅ Got vectorstore for chunk_size={chunk_size} in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"❌ Failed to get vectorstore for chunk_size={chunk_size}: {e}")
        
        # Test basic search
        try:
            start_time = time.time()
            query = "Berlin Mauer"
            chunk_size = 600  # Use smaller chunk size for faster test
            results = vector_store.similarity_search(
                query, 
                chunk_size, 
                k=3,
                min_relevance_score=0.3
            )
            logger.info(f"✅ Vector search successful in {time.time() - start_time:.2f}s")
            logger.info(f"  Found {len(results)} chunks for query '{query}'")
            
            # Print first result
            if results:
                doc, score = results[0]
                logger.info(f"  Top result relevance: {score:.4f}")
                logger.info(f"  Title: {doc.metadata.get('Artikeltitel', 'No title')}")
                logger.info(f"  Content preview: {doc.page_content[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ ChromaDB connection failed: {e}")
        return False

def test_llm_service():
    """Test LLM service with a simple query."""
    logger.info("Testing LLM service...")
    try:
        llm_service = LLMService()
        logger.info("✅ LLM service initialization successful")
        
        # Test generating a response
        try:
            start_time = time.time()
            response = llm_service.generate_response(
                question="Was ist die Hauptstadt von Deutschland?",
                context="Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.",
                temperature=0.3
            )
            logger.info(f"✅ LLM response generated in {time.time() - start_time:.2f}s")
            logger.info(f"  Response: {response['text'][:100]}...")
            logger.info(f"  Model: {response.get('model', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"❌ LLM response generation failed: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ LLM service initialization failed: {e}")
        return False

def test_embedding_service():
    """Test word embedding service."""
    logger.info("Testing word embedding service...")
    try:
        embedding_service = WordEmbeddingService()
        logger.info("✅ Word embedding service initialization successful")
        
        # Test finding similar words
        try:
            start_time = time.time()
            test_word = "mauer"
            similar_words = embedding_service.find_similar_words(test_word, top_n=5)
            logger.info(f"✅ Found similar words in {time.time() - start_time:.2f}s")
            logger.info(f"  Similar words to '{test_word}':")
            for word_info in similar_words:
                logger.info(f"    {word_info['word']} (similarity: {word_info['similarity']:.4f})")
            
            # Test boolean expression parsing
            boolean_expr = "berlin AND (mauer OR wall) NOT soviet"
            parsed = embedding_service.parse_boolean_expression(boolean_expr)
            logger.info(f"✅ Parsed boolean expression: {boolean_expr}")
            logger.info(f"  Result: {parsed}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Word embedding operations failed: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Word embedding service initialization failed: {e}")
        return False

def run_all_tests():
    """Run all component tests."""
    logger.info("Starting component tests...")
    
    results = {
        "ChromaDB": test_chroma_connection(),
        "LLM Service": test_llm_service(),
        "Embedding Service": test_embedding_service()
    }
    
    logger.info("\n--- Test Results Summary ---")
    all_passed = True
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{component}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All tests passed!")
    else:
        logger.info("⚠️ Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    run_all_tests()