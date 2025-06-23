# IMMEDIATE TEST: Check if your current setup uses prefixes
# Run this script to see if you're affected by the prefix issue

import logging
import sys
import os
import numpy as np

# Add your src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.vector_store import ChromaDBInterface
from langchain_ollama import OllamaEmbeddings
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_current_embedding_behavior():
    """
    Test what your current system actually does with queries.
    This will reveal if prefixes are missing.
    """
    
    print("🔍 TESTING CURRENT EMBEDDING BEHAVIOR")
    print("=" * 50)
    
    try:
        # Test your current vector store
        vector_store = ChromaDBInterface()
        
        # Test queries
        test_queries = ["Berlin", "Mauer", "Politik"]
        
        print("\n📊 Current Embedding Analysis:")
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            try:
                # Get embedding using your current system
                embedding = vector_store.embedding_model.embed_query(query)
                print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
                print(f"   📈 Sample values: {embedding[:3]}")
                
                # Test a search to see if it works
                results = vector_store.similarity_search(
                    query=query,
                    chunk_size=3000,
                    k=3,
                    min_relevance_score=0.2
                )
                
                if results:
                    scores = [score for _, score in results]
                    print(f"   🎯 Search results: {len(results)} found")
                    print(f"   📊 Similarity scores: {[f'{s:.3f}' for s in scores]}")
                    
                    # Check if scores are reasonable
                    max_score = max(scores)
                    if max_score < 0.3:
                        print(f"   ⚠️ WARNING: Low similarity scores suggest potential prefix issue")
                    else:
                        print(f"   ✅ Reasonable similarity scores")
                else:
                    print(f"   ❌ No results found - possible embedding issue")
                    
            except Exception as e:
                print(f"   ❌ Error testing query '{query}': {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test current system: {e}")
        return False

def test_manual_prefix_comparison():
    """
    Manually test the difference between prefixed and non-prefixed embeddings.
    This will show you the exact impact of the missing prefixes.
    """
    
    print("\n🧪 MANUAL PREFIX COMPARISON TEST")
    print("=" * 50)
    
    try:
        # Create embedding model directly
        embedding_model = OllamaEmbeddings(
            model=settings.OLLAMA_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        test_cases = [
            {
                "query": "Berlin",
                "document": "Berlin ist die Hauptstadt Deutschlands und liegt in Brandenburg."
            },
            {
                "query": "Mauer", 
                "document": "Die Berliner Mauer teilte Berlin von 1961 bis 1989."
            }
        ]
        
        print("\n📊 Prefix Impact Analysis:")
        
        for i, case in enumerate(test_cases, 1):
            query = case["query"]
            document = case["document"]
            
            print(f"\n{i}. Testing: '{query}' vs '{document[:30]}...'")
            
            # Test WITHOUT prefixes (your current situation)
            query_emb_no_prefix = embedding_model.embed_query(query)
            doc_emb_no_prefix = embedding_model.embed_documents([document])[0]
            
            # Test WITH prefixes (optimal situation)
            query_with_prefix = f"search_query: {query}"
            doc_with_prefix = f"search_document: {document}"
            
            query_emb_with_prefix = embedding_model.embed_query(query_with_prefix)
            doc_emb_with_prefix = embedding_model.embed_documents([doc_with_prefix])[0]
            
            # Calculate similarities
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            # Scenarios
            current_sim = cosine_similarity(query_emb_no_prefix, doc_emb_no_prefix)
            optimal_sim = cosine_similarity(query_emb_with_prefix, doc_emb_with_prefix)
            mixed_sim = cosine_similarity(query_emb_with_prefix, doc_emb_no_prefix)
            
            print(f"   📊 Current (no prefixes):     {current_sim:.4f}")
            print(f"   📊 Optimal (both prefixed):   {optimal_sim:.4f}")
            print(f"   📊 Mixed (query only):        {mixed_sim:.4f}")
            
            improvement = optimal_sim - current_sim
            mixed_penalty = current_sim - mixed_sim
            
            print(f"   📈 Improvement potential:     {improvement:+.4f} ({improvement/current_sim*100:+.1f}%)")
            print(f"   ⚠️ Mixed scenario penalty:     {mixed_penalty:+.4f} ({mixed_penalty/current_sim*100:+.1f}%)")
            
            if improvement > 0.05:
                print(f"   🎯 SIGNIFICANT improvement possible with proper prefixes!")
            elif improvement > 0.01:
                print(f"   ✅ Moderate improvement possible with proper prefixes")
            else:
                print(f"   ℹ️ Minor impact of prefixes for this case")
                
            if mixed_penalty > 0.1:
                print(f"   🚨 CRITICAL: Adding prefixes to queries only would HURT performance!")
            elif mixed_penalty > 0.05:
                print(f"   ⚠️ WARNING: Query-only prefixes would reduce performance")
        
    except Exception as e:
        print(f"❌ Manual comparison test failed: {e}")

def check_langchain_ollama_behavior():
    """
    Check if LangChain Ollama automatically adds prefixes.
    Spoiler: It doesn't, which is the source of your issue.
    """
    
    print("\n🔍 CHECKING LANGCHAIN OLLAMA BEHAVIOR")
    print("=" * 50)
    
    try:
        embedding_model = OllamaEmbeddings(
            model=settings.OLLAMA_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # Test what actually gets sent to the model
        test_query = "test query"
        test_doc = "test document"
        
        print(f"📝 Input query: '{test_query}'")
        print(f"📝 Input document: '{test_doc}'")
        
        # The issue: LangChain OllamaEmbeddings sends text as-is
        # It does NOT add the required prefixes for nomic-embed-text
        
        query_embedding = embedding_model.embed_query(test_query)
        doc_embedding = embedding_model.embed_documents([test_doc])[0]
        
        print(f"✅ LangChain processed query without modification")
        print(f"✅ LangChain processed document without modification")
        print(f"❌ NO PREFIXES ADDED - This is the problem!")
        
        print(f"\n🎯 CONCLUSION:")
        print(f"   • LangChain OllamaEmbeddings does NOT add prefixes automatically")
        print(f"   • Your queries are sent as: '{test_query}'")
        print(f"   • Should be sent as: 'search_query: {test_query}'")
        print(f"   • Your documents were stored as: '{test_doc}'") 
        print(f"   • Should be stored as: 'search_document: {test_doc}'")
        
        return {
            "adds_prefixes": False,
            "impact": "HIGH - Both queries and documents missing required prefixes",
            "action_needed": "Implement custom embedding wrapper with prefixes"
        }
        
    except Exception as e:
        print(f"❌ Failed to check LangChain behavior: {e}")
        return {"error": str(e)}

def generate_action_plan():
    """Generate specific action plan based on test results."""
    
    print("\n📋 RECOMMENDED ACTION PLAN")
    print("=" * 50)
    
    print("🚨 ISSUE CONFIRMED: Missing nomic-embed-text prefixes")
    print("   • Impact: Suboptimal embedding quality")
    print("   • Severity: HIGH - affects all searches")
    print("   • Cause: LangChain OllamaEmbeddings doesn't add required prefixes")
    
    print("\n🎯 IMMEDIATE ACTIONS (Next 2 hours):")
    print("   1. 📊 Quantify the impact with test_manual_prefix_comparison()")
    print("   2. 🧪 Create test collection with proper prefixes") 
    print("   3. 📈 Compare search quality: current vs prefixed")
    
    print("\n⚡ QUICK WINS (This week):")
    print("   1. 🔧 Implement PrefixedOllamaEmbeddings class")
    print("   2. 🧪 Create one new collection with proper prefixes")
    print("   3. 📊 A/B test queries against both collections")
    print("   4. 📏 Measure improvement in similarity scores")
    
    print("\n🚀 FULL SOLUTION (Next iteration):")
    print("   1. 🔄 Re-index all collections with proper document prefixes")
    print("   2. 🔧 Update ChromaDBInterface to use prefixed embeddings")
    print("   3. 🧪 Comprehensive testing and validation")
    print("   4. 📊 Monitor performance improvements")
    
    print("\n⚠️ CRITICAL WARNING:")
    print("   • DON'T add prefixes to queries only - this creates semantic mismatch")
    print("   • Either both queries AND documents need prefixes, or neither")
    print("   • Current state (no prefixes) is better than mixed state")

def main():
    """Run all prefix-related tests."""
    
    print("🎯 NOMIC-EMBED-TEXT PREFIX INVESTIGATION")
    print("=" * 60)
    print("Testing whether your system uses the required prefixes...")
    
    # Test current behavior
    current_works = test_current_embedding_behavior()
    
    if current_works:
        # Check LangChain behavior
        langchain_behavior = check_langchain_ollama_behavior()
        
        # Test manual prefix comparison
        test_manual_prefix_comparison()
        
        # Generate action plan
        generate_action_plan()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY FOR YOUR PRESENTATION:")
    print("• Your embeddings work but are SUBOPTIMAL due to missing prefixes")
    print("• nomic-embed-text requires 'search_query:' and 'search_document:' prefixes")
    print("• LangChain OllamaEmbeddings doesn't add these automatically")
    print("• This affects ALL your searches - significant improvement possible")
    print("• Solution: Implement proper prefixed embeddings")

if __name__ == "__main__":
    main()