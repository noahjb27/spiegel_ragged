# debug_hanging_issue.py
"""
Focused debugging script to identify exactly where the retrieval process hangs.
"""
import os
import sys
import time
import logging
import threading
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ultra-verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Also log ChromaDB and httpx
logging.getLogger("chromadb").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def monitor_thread(stop_event):
    """Monitor thread to show the script is still alive"""
    counter = 0
    while not stop_event.is_set():
        counter += 1
        print(f"\r⏳ Still running... ({counter}s)", end="", flush=True)
        time.sleep(1)


def test_chromadb_connection_detailed():
    """Test ChromaDB connection with detailed debugging"""
    print("\n=== TESTING CHROMADB CONNECTION ===\n")
    
    # Start monitor thread
    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_thread, args=(stop_event,))
    monitor.start()
    
    try:
        print("1. Importing chromadb...")
        import chromadb
        print("   ✅ chromadb imported")
        
        print("\n2. Reading settings...")
        from src.config import settings
        print(f"   Host: {settings.CHROMA_DB_HOST}")
        print(f"   Port: {settings.CHROMA_DB_PORT}")
        print(f"   SSL: {settings.CHROMA_DB_SSL}")
        
        print("\n3. Creating HttpClient...")
        client = chromadb.HttpClient(
            host=settings.CHROMA_DB_HOST,
            port=settings.CHROMA_DB_PORT,
            ssl=settings.CHROMA_DB_SSL
        )
        print("   ✅ HttpClient created")
        
        print("\n4. Testing client connection...")
        # Try to list collections (this often reveals connection issues)
        try:
            collections = client.list_collections()
            print(f"   ✅ Connected! Found {len(collections)} collections")
            
            # Print collection names
            for col in collections[:5]:  # First 5 only
                print(f"      - {col.name}")
            if len(collections) > 5:
                print(f"      ... and {len(collections) - 5} more")
                
        except Exception as e:
            print(f"   ❌ Failed to list collections: {e}")
            
        stop_event.set()
        monitor.join()
        return True
        
    except Exception as e:
        stop_event.set()
        monitor.join()
        print(f"\n❌ Connection test failed: {e}")
        logger.exception("Detailed connection test failed")
        return False


def test_ollama_embeddings():
    """Test Ollama embeddings connection"""
    print("\n=== TESTING OLLAMA EMBEDDINGS ===\n")
    
    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_thread, args=(stop_event,))
    monitor.start()
    
    try:
        print("1. Importing OllamaEmbeddings...")
        from langchain_ollama import OllamaEmbeddings
        print("   ✅ Imported")
        
        print("\n2. Reading Ollama settings...")
        from src.config import settings
        print(f"   Model: {settings.OLLAMA_MODEL_NAME}")
        print(f"   Base URL: {settings.OLLAMA_BASE_URL}")
        
        print("\n3. Creating embedding model...")
        embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL
        )
        print("   ✅ Model created")
        
        print("\n4. Testing embedding generation...")
        test_text = "Berlin"
        
        # This is where it often hangs
        print(f"   Embedding text: '{test_text}'")
        start_time = time.time()
        
        # Try with timeout
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(embeddings.embed_query, test_text)
            try:
                result = future.result(timeout=10)  # 10 second timeout
                elapsed = time.time() - start_time
                print(f"   ✅ Embedding generated in {elapsed:.2f}s")
                print(f"   Vector dimension: {len(result)}")
            except concurrent.futures.TimeoutError:
                print("   ❌ Embedding generation timed out after 10s!")
                print("   This is likely where your search is hanging.")
                stop_event.set()
                monitor.join()
                return False
                
        stop_event.set()
        monitor.join()
        return True
        
    except Exception as e:
        stop_event.set()
        monitor.join()
        print(f"\n❌ Ollama test failed: {e}")
        logger.exception("Ollama test failed")
        return False


def test_vector_search_step_by_step():
    """Test vector search step by step to find hanging point"""
    print("\n=== STEP-BY-STEP VECTOR SEARCH TEST ===\n")
    
    try:
        from src.core.vector_store import ChromaDBInterface
        
        print("1. Creating ChromaDBInterface...")
        vector_store = ChromaDBInterface()
        print("   ✅ Created")
        
        print("\n2. Getting vectorstore for chunk_size=3000...")
        vs = vector_store.get_vectorstore(3000)
        print("   ✅ Got vectorstore")
        
        print("\n3. Preparing search...")
        query = "Test"
        k = 2
        
        print(f"   Query: '{query}'")
        print(f"   k: {k}")
        
        print("\n4. Calling similarity_search_with_relevance_scores...")
        print("   ⚠️  This is likely where it hangs...")
        
        # Monitor in separate thread
        stop_event = threading.Event()
        monitor = threading.Thread(target=monitor_thread, args=(stop_event,))
        monitor.start()
        
        start_time = time.time()
        
        # The actual search call that might hang
        results = vs.similarity_search_with_relevance_scores(
            query, 
            k=k, 
            filter=None
        )
        
        elapsed = time.time() - start_time
        stop_event.set()
        monitor.join()
        
        print(f"\n   ✅ Search completed in {elapsed:.2f}s!")
        print(f"   Found {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Step-by-step test failed: {e}")
        logger.exception("Step-by-step test failed")
        return False


def test_with_threading_info():
    """Show active threads to debug hanging"""
    print("\n=== THREAD ANALYSIS ===\n")
    
    import threading
    
    print("Active threads before test:")
    for thread in threading.enumerate():
        print(f"  - {thread.name} (daemon: {thread.daemon})")
    
    # Try a simple operation
    try:
        from src.core.vector_store import ChromaDBInterface
        
        print("\nCreating ChromaDBInterface...")
        
        # Create in a thread to monitor
        result = [None]
        exception = [None]
        
        def create_interface():
            try:
                result[0] = ChromaDBInterface()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=create_interface)
        thread.start()
        thread.join(timeout=5)  # 5 second timeout
        
        if thread.is_alive():
            print("❌ ChromaDBInterface creation is hanging!")
            print("\nActive threads during hang:")
            for t in threading.enumerate():
                print(f"  - {t.name} (daemon: {t.daemon})")
        elif exception[0]:
            print(f"❌ Creation failed with: {exception[0]}")
        else:
            print("✅ ChromaDBInterface created successfully")
            
    except Exception as e:
        print(f"❌ Thread test failed: {e}")


def main():
    """Run focused debugging tests"""
    print("DEBUGGING HANGING ISSUE IN SPIEGEL RAG")
    print("=" * 50)
    
    # Run tests in order of likely failure points
    tests = [
        ("Thread Analysis", test_with_threading_info),
        ("ChromaDB Connection", test_chromadb_connection_detailed),
        ("Ollama Embeddings", test_ollama_embeddings),
        ("Vector Search", test_vector_search_step_by_step),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            if not success:
                print(f"\n⚠️  Found issue in: {test_name}")
                print("\nPOSSIBLE SOLUTIONS:")
                
                if "Ollama" in test_name:
                    print("1. Check if Ollama service is accessible")
                    print("2. Try accessing the Ollama URL directly:")
                    print(f"   curl {os.getenv('OLLAMA_BASE_URL', 'https://dighist.geschichte.hu-berlin.de:11434')}/api/tags")
                    print("3. Check VPN connection")
                    print("4. Try using a local Ollama instance for testing")
                    
                elif "ChromaDB" in test_name:
                    print("1. Check VPN/network connection")
                    print("2. Verify ChromaDB is running on the server")
                    print("3. Check firewall settings")
                    print("4. Try using a local ChromaDB for testing")
                    
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            logger.exception(f"Test {test_name} crashed")


if __name__ == "__main__":
    main()