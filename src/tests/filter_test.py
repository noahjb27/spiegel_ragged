#!/usr/bin/env python3
"""
Robust filter test that won't hang indefinitely.
Uses proper timeout handling and continues even if individual tests hang.
"""

import os
import sys
import time
import threading
import signal

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.vector_store import ChromaDBInterface

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Operation timed out")

def run_test_with_timeout(test_func, timeout_seconds=30):
    """Run a test function with timeout using threading."""
    
    result = {"completed": False, "data": None, "error": None}
    
    def run_test():
        try:
            result["data"] = test_func()
            result["completed"] = True
        except Exception as e:
            result["error"] = str(e)
            result["completed"] = True
    
    # Start test in thread
    test_thread = threading.Thread(target=run_test, daemon=True)
    test_thread.start()
    
    # Wait with timeout
    for i in range(timeout_seconds):
        if result["completed"]:
            break
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"      ⏳ Still waiting... {i+1}s")
    
    if not result["completed"]:
        print(f"      🔴 TIMEOUT after {timeout_seconds}s")
        return {"success": False, "error": "TIMEOUT", "time": None}
    
    if result["error"]:
        print(f"      ❌ Error: {result['error']}")
        return {"success": False, "error": result["error"], "time": None}
    
    return result["data"]

def test_single_filter(vector_store, filter_name, filter_dict):
    """Test a single filter - this is what gets called with timeout."""
    print(f"   Testing: {filter_name}")
    if filter_dict:
        print(f"   Filter: {str(filter_dict)[:100]}...")
    else:
        print(f"   Filter: None")
    
    start = time.time()
    
    results = vector_store.similarity_search(
        query="Berlin",
        chunk_size=2000,
        k=5,
        filter_dict=filter_dict,
        min_relevance_score=0.3
    )
    
    elapsed = time.time() - start
    
    print(f"   ✅ Success: {elapsed:.2f}s, {len(results)} results")
    
    # Show sample result
    if results:
        doc, score = results[0]
        title = doc.metadata.get('Artikeltitel', 'No title')[:40]
        year = doc.metadata.get('Jahrgang', 'Unknown')
        print(f"   📄 Sample: {title}... ({year}) - Score: {score:.3f}")
    
    return {
        "success": True,
        "time": elapsed,
        "results": len(results),
        "error": None
    }

def test_keyword_filter(vector_store, keywords, search_in=None):
    """Test keyword filtering."""
    print(f"   Testing keywords: '{keywords}'")
    print(f"   Search in: {search_in or ['Text', 'Artikeltitel']}")
    
    start = time.time()
    
    results = vector_store.similarity_search(
        query="Berlin",
        chunk_size=2000,
        k=5,
        filter_dict=None,  # No server-side filter
        keywords=keywords,
        search_in=search_in or ["Text", "Artikeltitel"],
        enforce_keywords=True,
        min_relevance_score=0.3
    )
    
    elapsed = time.time() - start
    
    print(f"   ✅ Success: {elapsed:.2f}s, {len(results)} results")
    
    # Verify keyword matching
    if results and keywords:
        doc, score = results[0]
        content_lower = doc.page_content.lower()
        title_lower = doc.metadata.get('Artikeltitel', '').lower()
        
        # Simple check for first keyword
        first_keyword = keywords.split()[0].lower()
        if first_keyword in content_lower or first_keyword in title_lower:
            print(f"   ✅ Keyword '{first_keyword}' found in result")
        else:
            print(f"   ⚠️ Keyword '{first_keyword}' not found in result")
    
    return {
        "success": True,
        "time": elapsed,
        "results": len(results),
        "error": None
    }

def main():
    """Main test function with robust timeout handling."""
    print("🔍 ROBUST FILTER TEST")
    print("=" * 50)
    print("Testing filters with robust timeout handling that won't hang.\n")
    
    try:
        # Initialize vector store
        print("📡 Connecting to ChromaDB...")
        vector_store = ChromaDBInterface()
        print("✅ Connected successfully\n")
        
        # Define test cases
        test_cases = [
            # Server-side filters
            ("No Filter (Baseline)", None),
            ("Single Year Filter", {"Jahrgang": 1965}),
            ("Year Range Filter (Problematic)", {"$and": [{"Jahrgang": {"$gte": 1960}}, {"Jahrgang": {"$lte": 1970}}]}),
            ("Issue Number Filter", {"Ausgabe": 1}),
            ("Issue Range Filter", {"Ausgabe": {"$gte": 1, "$lte": 5}}),
        ]
        
        results = {}
        
        print("🔧 TESTING SERVER-SIDE FILTERS:")
        print("-" * 40)
        
        for i, (name, filter_dict) in enumerate(test_cases, 1):
            print(f"\n{i}. {name}")
            
            def test_func():
                return test_single_filter(vector_store, name, filter_dict)
            
            result = run_test_with_timeout(test_func, timeout_seconds=30)
            results[name] = result
            
            print(f"   ➡️ Continuing to next test...")
        
        # Test keyword filtering
        print(f"\n\n🔤 TESTING KEYWORD FILTERING:")
        print("-" * 40)
        
        keyword_tests = [
            ("Simple Keyword", "Mauer", None),
            ("Boolean AND", "Berlin AND Mauer", None),
            ("Boolean OR", "Mauer OR Wall", None),
            ("Title Search", "Berlin", ["Artikeltitel"]),
            ("Text Search", "Politik", ["Text"]),
        ]
        
        for i, (name, keywords, search_in) in enumerate(keyword_tests, 1):
            print(f"\n{i}. {name}")
            
            def test_func():
                return test_keyword_filter(vector_store, keywords, search_in)
            
            result = run_test_with_timeout(test_func, timeout_seconds=15)
            results[name] = result
            
            print(f"   ➡️ Continuing to next test...")
        
        # Summary
        print(f"\n\n📊 TEST SUMMARY:")
        print("=" * 40)
        
        server_side_working = 0
        server_side_total = len(test_cases)
        keyword_working = 0
        keyword_total = len(keyword_tests)
        
        print("\n🔧 Server-side filters:")
        for name, filter_dict in test_cases:
            result = results.get(name, {})
            if result.get("success"):
                status = f"✅ {result['time']:.2f}s"
                server_side_working += 1
            elif result.get("error") == "TIMEOUT":
                status = "🔴 TIMEOUT"
            else:
                status = f"❌ {result.get('error', 'Unknown error')}"
            
            print(f"   {name:<30} {status}")
        
        print(f"\n🔤 Keyword filters:")
        for name, keywords, search_in in keyword_tests:
            result = results.get(name, {})
            if result.get("success"):
                status = f"✅ {result['time']:.2f}s"
                keyword_working += 1
            elif result.get("error") == "TIMEOUT":
                status = "🔴 TIMEOUT"
            else:
                status = f"❌ {result.get('error', 'Unknown error')}"
            
            print(f"   {name:<30} {status}")
        
        # Conclusions
        print(f"\n🎯 CONCLUSIONS:")
        print(f"   📊 Server-side filters: {server_side_working}/{server_side_total} working")
        print(f"   📊 Keyword filters: {keyword_working}/{keyword_total} working")
        
        if keyword_working > server_side_working:
            print(f"\n✅ RECOMMENDATION: Use keyword filtering instead of server-side filters")
            print(f"   → Keyword filtering is more reliable")
            print(f"   → Avoids server timeout issues")
        elif server_side_working > 0:
            print(f"\n🟡 RECOMMENDATION: Use working server-side filters, avoid problematic ones")
            print(f"   → Avoid year range filters")
            print(f"   → Use simple equality filters when possible")
        else:
            print(f"\n🔴 RECOMMENDATION: Use post-retrieval filtering only")
            print(f"   → Server-side filtering appears unreliable")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test setup failed: {e}")

if __name__ == "__main__":
    main()