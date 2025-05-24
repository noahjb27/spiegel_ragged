# test_retrieval_diagnostic.py
"""
Diagnostic script to identify where the retrieval process is hanging.
Run this to test each component in isolation.
"""
import os
import sys
import time
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_1_vector_store_connection():
    """Test 1: Basic ChromaDB connection"""
    print("\n" + "="*50)
    print("TEST 1: Vector Store Connection")
    print("="*50)
    
    try:
        from src.core.vector_store import ChromaDBInterface
        
        start_time = time.time()
        logger.info("Creating ChromaDBInterface...")
        
        vector_store = ChromaDBInterface()
        
        elapsed = time.time() - start_time
        print(f"✅ Vector store initialized in {elapsed:.2f}s")
        
        # Test getting collection names
        print("\nTesting collection access...")
        for chunk_size in [2000, 3000]:
            try:
                collection_name = f"recursive_chunks_{chunk_size}_{400 if chunk_size == 2000 else 300}_TH_cosine_nomic-embed-text"
                print(f"  Checking collection: {collection_name}")
                
                # Try to get vectorstore
                vs = vector_store.get_vectorstore(chunk_size)
                print(f"  ✅ Got vectorstore for chunk_size={chunk_size}")
                
            except Exception as e:
                print(f"  ❌ Failed for chunk_size={chunk_size}: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("Vector store connection failed")
        return False


def test_2_simple_search():
    """Test 2: Simple search without strategies"""
    print("\n" + "="*50)
    print("TEST 2: Simple Direct Search")
    print("="*50)
    
    try:
        from src.core.vector_store import ChromaDBInterface
        
        vector_store = ChromaDBInterface()
        
        # Test parameters
        query = "Berlin"
        chunk_size = 3000
        k = 3
        
        print(f"Searching for: '{query}'")
        print(f"Parameters: chunk_size={chunk_size}, k={k}")
        
        # Add timeout mechanism
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Search timed out after 30 seconds")
        
        # Set 30 second timeout
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
        
        try:
            start_time = time.time()
            
            # Direct search call
            results = vector_store.similarity_search(
                query=query,
                chunk_size=chunk_size,
                k=k,
                filter_dict=None,
                min_relevance_score=0.3,
                keywords=None,
                search_in=["Text"],
                enforce_keywords=False
            )
            
            elapsed = time.time() - start_time
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            print(f"\n✅ Search completed in {elapsed:.2f}s")
            print(f"Found {len(results)} results")
            
            if results:
                doc, score = results[0]
                print(f"\nTop result:")
                print(f"  Score: {score:.4f}")
                print(f"  Title: {doc.metadata.get('Artikeltitel', 'No title')}")
                print(f"  Date: {doc.metadata.get('Datum', 'No date')}")
                print(f"  Preview: {doc.page_content[:100]}...")
                
            return True
            
        except TimeoutError:
            print("❌ Search timed out!")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("Simple search failed")
        return False


def test_3_search_with_filters():
    """Test 3: Search with metadata filters"""
    print("\n" + "="*50)
    print("TEST 3: Search with Filters")
    print("="*50)
    
    try:
        from src.core.vector_store import ChromaDBInterface
        
        vector_store = ChromaDBInterface()
        
        # Build a filter
        print("Building metadata filter for years 1960-1970...")
        
        filter_dict = vector_store.build_metadata_filter(
            year_range=[1960, 1970],
            keywords=None,
            search_in=None
        )
        
        print(f"Filter: {filter_dict}")
        
        # Search with filter
        start_time = time.time()
        
        results = vector_store.similarity_search(
            query="Politik",
            chunk_size=3000,
            k=5,
            filter_dict=filter_dict,
            min_relevance_score=0.3
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Filtered search completed in {elapsed:.2f}s")
        print(f"Found {len(results)} results")
        
        # Check years
        if results:
            years = [doc.metadata.get('Jahrgang', 'Unknown') for doc, _ in results]
            print(f"Years in results: {set(years)}")
            
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("Filtered search failed")
        return False


def test_4_strategy_pattern():
    """Test 4: Search using strategy pattern"""
    print("\n" + "="*50)
    print("TEST 4: Strategy Pattern Search")
    print("="*50)
    
    try:
        from src.core.search.strategies import StandardSearchStrategy, SearchConfig
        from src.core.vector_store import ChromaDBInterface
        
        vector_store = ChromaDBInterface()
        
        # Create search config
        config = SearchConfig(
            content_description="Berliner Mauer",
            year_range=(1960, 1970),
            chunk_size=3000,
            top_k=5
        )
        
        print(f"Search config: {config}")
        
        # Create strategy
        strategy = StandardSearchStrategy()
        
        # Execute search
        start_time = time.time()
        
        # Note: The strategy needs the vector_store passed to search()
        result = strategy.search(config, vector_store)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Strategy search completed in {elapsed:.2f}s")
        print(f"Found {result.chunk_count} chunks")
        print(f"Metadata: {result.metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("Strategy search failed")
        return False


def test_5_rag_engine():
    """Test 5: Full RAG engine"""
    print("\n" + "="*50)
    print("TEST 5: RAG Engine")
    print("="*50)
    
    try:
        from src.core.engine import SpiegelRAG
        from src.core.search.strategies import StandardSearchStrategy, SearchConfig
        
        print("Initializing RAG engine...")
        engine = SpiegelRAG()
        
        # Create search config
        config = SearchConfig(
            content_description="Deutsche Geschichte",
            year_range=(1965, 1970),
            chunk_size=3000,
            top_k=3
        )
        
        # Create strategy
        strategy = StandardSearchStrategy()
        
        # Execute search through engine
        print("\nExecuting search through engine...")
        start_time = time.time()
        
        result = engine.search(strategy, config)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Engine search completed in {elapsed:.2f}s")
        print(f"Found {result.chunk_count} chunks")
        
        # Test analysis (if search worked)
        if result.chunks:
            print("\nTesting analysis...")
            
            analysis_result = engine.analyze(
                question="Was sind die Hauptthemen?",
                chunks=[doc for doc, _ in result.chunks],
                model="hu-llm"
            )
            
            print(f"✅ Analysis completed")
            print(f"Answer preview: {analysis_result.answer[:200]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        logger.exception("RAG engine test failed")
        return False


def test_6_embedding_service():
    """Test 6: Embedding service (if available)"""
    print("\n" + "="*50)
    print("TEST 6: Embedding Service")
    print("="*50)
    
    try:
        from src.core.embedding_service import WordEmbeddingService
        
        print("Initializing embedding service...")
        embedding_service = WordEmbeddingService()
        
        # Test similar words
        test_word = "mauer"
        similar = embedding_service.find_similar_words(test_word, top_n=5)
        
        print(f"\n✅ Found {len(similar)} similar words for '{test_word}':")
        for word_info in similar:
            print(f"  - {word_info['word']} (similarity: {word_info['similarity']:.4f})")
            
        return True
        
    except Exception as e:
        print(f"⚠️  Embedding service not available: {e}")
        return True  # Not critical


def test_7_check_imports():
    """Test 7: Check all imports work correctly"""
    print("\n" + "="*50)
    print("TEST 7: Import Check")
    print("="*50)
    
    imports_to_test = [
        ("ChromaDB HttpClient", "import chromadb"),
        ("Langchain ChromaDB", "from langchain_chroma import Chroma"),
        ("Ollama Embeddings", "from langchain_ollama import OllamaEmbeddings"),
        ("OpenAI", "from openai import OpenAI"),
        ("Gradio", "import gradio as gr"),
    ]
    
    all_good = True
    
    for name, import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            all_good = False
            
    return all_good


def run_all_tests():
    """Run all diagnostic tests"""
    print("\n" + "="*70)
    print("SPIEGEL RAG DIAGNOSTIC TESTS")
    print("="*70)
    
    tests = [
        ("Import Check", test_7_check_imports),
        ("Vector Store Connection", test_1_vector_store_connection),
        ("Simple Search", test_2_simple_search),
        ("Filtered Search", test_3_search_with_filters),
        ("Strategy Pattern", test_4_strategy_pattern),
        ("RAG Engine", test_5_rag_engine),
        ("Embedding Service", test_6_embedding_service),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            result = test_func()
            results[test_name] = result
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            logger.exception(f"Test {test_name} crashed")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if not results.get("Import Check", False):
        print("⚠️  Missing dependencies. Run: pip install -r requirements.txt")
    
    if not results.get("Vector Store Connection", False):
        print("⚠️  Cannot connect to ChromaDB. Check:")
        print("   - VPN/Network connection to HU Berlin")
        print("   - ChromaDB host/port in .env file")
        print("   - SSL settings")
    
    if results.get("Vector Store Connection", False) and not results.get("Simple Search", False):
        print("⚠️  Connection works but search hangs. Possible issues:")
        print("   - Ollama embedding service not responding")
        print("   - Collection names mismatch")
        print("   - Network timeout issues")
    
    if results.get("Simple Search", False) and not results.get("Strategy Pattern", False):
        print("⚠️  Direct search works but strategy pattern fails. Check:")
        print("   - Strategy implementation")
        print("   - How vector_store is passed to strategies")


if __name__ == "__main__":
    run_all_tests()