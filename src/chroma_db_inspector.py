#!/usr/bin/env python3
"""
ChromaDB Data Inspector - Inspect the structure and content of your ChromaDB collections
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from collections import Counter

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.vector_store import ChromaDBInterface
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBInspector:
    """Inspector for ChromaDB collections and data structure."""
    
    def __init__(self):
        """Initialize the inspector."""
        self.vector_store = ChromaDBInterface()
        self.client = self.vector_store.client
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            return collection_names
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def inspect_collection(self, collection_name: str, sample_size: int = 10) -> Dict[str, Any]:
        """Inspect a specific collection in detail."""
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get basic collection info
            count = collection.count()
            logger.info(f"Collection '{collection_name}' has {count} documents")
            
            # Get a sample of documents
            sample_result = collection.get(
                limit=sample_size,
                include=["metadatas", "documents"]
            )
            
            inspection_result = {
                "name": collection_name,
                "document_count": count,
                "sample_size": len(sample_result["ids"]) if sample_result["ids"] else 0,
                "sample_documents": sample_result,
                "metadata_analysis": self._analyze_metadata(sample_result.get("metadatas", [])),
                "document_analysis": self._analyze_documents(sample_result.get("documents", []))
            }
            
            return inspection_result
            
        except Exception as e:
            logger.error(f"Error inspecting collection '{collection_name}': {e}")
            return {"error": str(e)}
    
    def _analyze_metadata(self, metadatas: List[Dict]) -> Dict[str, Any]:
        """Analyze metadata structure and values."""
        if not metadatas:
            return {"error": "No metadata available"}
        
        # Collect all unique keys
        all_keys = set()
        key_types = {}
        key_examples = {}
        
        for metadata in metadatas:
            if metadata:  # Check if metadata is not None
                for key, value in metadata.items():
                    all_keys.add(key)
                    
                    # Track data types
                    value_type = type(value).__name__
                    if key not in key_types:
                        key_types[key] = set()
                    key_types[key].add(value_type)
                    
                    # Store examples
                    if key not in key_examples:
                        key_examples[key] = []
                    if len(key_examples[key]) < 5:  # Store up to 5 examples
                        key_examples[key].append(value)
        
        # Convert sets to lists for JSON serialization
        for key in key_types:
            key_types[key] = list(key_types[key])
        
        return {
            "total_unique_keys": len(all_keys),
            "keys": sorted(list(all_keys)),
            "key_types": key_types,
            "key_examples": key_examples
        }
    
    def _analyze_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Analyze document content."""
        if not documents:
            return {"error": "No documents available"}
        
        # Calculate document length statistics
        lengths = [len(doc) for doc in documents if doc]
        
        return {
            "document_count": len(documents),
            "average_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "first_document_preview": documents[0][:300] + "..." if documents and documents[0] else "No content"
        }
    
    def analyze_year_distribution(self, collection_name: str, batch_size: int = 100) -> Dict[str, Any]:
        """Analyze the distribution of years in the collection."""
        try:
            collection = self.client.get_collection(collection_name)
            total_count = collection.count()
            
            year_counter = Counter()
            jahrgang_examples = {}
            
            # Process in batches
            offset = 0
            while offset < total_count:
                batch = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
                )
                
                if not batch["metadatas"]:
                    break
                
                for metadata in batch["metadatas"]:
                    if metadata and "Jahrgang" in metadata:
                        jahrgang = metadata["Jahrgang"]
                        year_counter[jahrgang] += 1
                        
                        # Store examples of different data types
                        jahrgang_type = type(jahrgang).__name__
                        if jahrgang_type not in jahrgang_examples:
                            jahrgang_examples[jahrgang_type] = []
                        if len(jahrgang_examples[jahrgang_type]) < 3:
                            jahrgang_examples[jahrgang_type].append(jahrgang)
                
                offset += batch_size
                if offset % 500 == 0:
                    logger.info(f"Processed {offset}/{total_count} documents...")
            
            return {
                "total_documents": total_count,
                "documents_with_jahrgang": sum(year_counter.values()),
                "unique_years": len(year_counter),
                "year_distribution": dict(year_counter.most_common()),
                "jahrgang_data_types": jahrgang_examples,
                "year_range": {
                    "min": min(year_counter.keys()) if year_counter else None,
                    "max": max(year_counter.keys()) if year_counter else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing year distribution: {e}")
            return {"error": str(e)}
    
    def test_filters(self, collection_name: str) -> Dict[str, Any]:
        """Test different filter formats using the PROPER embedding service."""
        try:
            # FIXED: Use the same embedding service as the application
            vector_store = ChromaDBInterface()
            
            test_results = {}
            
            # Test different filter formats using the vector store (not raw collection)
            filter_tests = [
                ("No filter", None),
                ("Simple equality", {"Jahrgang": 1965}),
                ("String equality", {"Jahrgang": "1965"}),
                ("$eq operator", {"Jahrgang": {"$eq": 1965}}),
                ("$gt operator", {"Jahrgang": {"$gt": 1960}}),
                ("$gte operator", {"Jahrgang": {"$gte": 1960}}),
                ("$lt operator", {"Jahrgang": {"$lt": 1970}}),
                ("$lte operator", {"Jahrgang": {"$lte": 1970}}),
                ("$and with separate conditions", {
                    "$and": [
                        {"Jahrgang": {"$gte": 1960}},
                        {"Jahrgang": {"$lte": 1970}}
                    ]
                }),
            ]
            
            # Extract chunk size from collection name
            if "3000" in collection_name:
                chunk_size = 3000
            elif "2000" in collection_name:
                chunk_size = 2000
            else:
                chunk_size = 3000  # default
            
            for test_name, filter_dict in filter_tests:
                try:
                    logger.info(f"Testing filter: {test_name}")
                    
                    # Use the vector store's similarity_search method (uses proper Ollama embeddings)
                    results = vector_store.similarity_search(
                        query="Berlin",
                        chunk_size=chunk_size,
                        k=3,
                        filter_dict=filter_dict,
                        min_relevance_score=0.3
                    )
                    
                    test_results[test_name] = {
                        "status": "SUCCESS",
                        "filter": filter_dict,
                        "result_count": len(results)
                    }
                    logger.info(f"  ✅ Success: {len(results)} results")
                    
                    if results:
                        doc, score = results[0]
                        logger.info(f"    Sample: {doc.metadata.get('Artikeltitel', 'No title')[:50]}... (Year: {doc.metadata.get('Jahrgang', 'Unknown')})")
                
                except Exception as e:
                    test_results[test_name] = {
                        "status": "FAILED",
                        "filter": filter_dict,
                        "error": str(e)
                    }
                    logger.info(f"  ❌ Failed: {str(e)[:100]}...")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error testing filters: {e}")
            return {"error": str(e)}

def main():
    """Main inspection function."""
    print("🔍 ChromaDB Data Inspector")
    print("=" * 50)
    
    inspector = ChromaDBInspector()
    
    # List all collections
    print("\n📊 Available Collections:")
    collections = inspector.list_collections()
    for i, collection in enumerate(collections, 1):
        print(f"  {i}. {collection}")
    
    if not collections:
        print("  No collections found!")
        return
    
    # Inspect each collection that matches our pattern
    for collection in collections:
        if "recursive_chunks" in collection and "spiegel" in collection.lower() or "TH_cosine" in collection:
            print(f"\n🔬 Inspecting Collection: {collection}")
            print("-" * 40)
            
            # Basic inspection
            inspection = inspector.inspect_collection(collection, sample_size=5)
            
            if "error" not in inspection:
                print(f"Documents: {inspection['document_count']}")
                print(f"Sample size: {inspection['sample_size']}")
                
                # Metadata analysis
                metadata_analysis = inspection["metadata_analysis"]
                if "error" not in metadata_analysis:
                    print(f"\nMetadata Keys ({metadata_analysis['total_unique_keys']}):")
                    for key in metadata_analysis["keys"]:
                        types = metadata_analysis["key_types"].get(key, ["unknown"])
                        examples = metadata_analysis["key_examples"].get(key, [])
                        print(f"  • {key} ({', '.join(types)}): {examples[:3]}")
                
                # Document analysis
                doc_analysis = inspection["document_analysis"]
                if "error" not in doc_analysis:
                    print(f"\nDocument Analysis:")
                    print(f"  • Average length: {doc_analysis['average_length']:.0f} chars")
                    print(f"  • Length range: {doc_analysis['min_length']} - {doc_analysis['max_length']}")
                
                
                # Test filters
                print(f"\n🧪 Filter Testing:")
                filter_tests = inspector.test_filters(collection)
                if "error" not in filter_tests:
                    working_filters = []
                    failed_filters = []
                    
                    for test_name, result in filter_tests.items():
                        if result["status"] == "SUCCESS":
                            working_filters.append(test_name)
                        else:
                            failed_filters.append((test_name, result["error"]))
                    
                    print(f"  ✅ Working filters ({len(working_filters)}):")
                    for filter_name in working_filters:
                        print(f"    • {filter_name}")
                    
                    print(f"  ❌ Failed filters ({len(failed_filters)}):")
                    for filter_name, error in failed_filters:
                        print(f"    • {filter_name}: {error[:60]}...")
            
            else:
                print(f"Error: {inspection['error']}")
    
    print("\n🎯 Recommendations:")
    print("1. Use keyword filtering instead of server-side year filters")
    print("2. If server-side filtering is needed, use the working filter formats identified above")
    print("3. Consider post-retrieval filtering for complex queries")

if __name__ == "__main__":
    main()