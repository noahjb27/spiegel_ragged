# In a test script

import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from src.core.rag_engine import SpiegelRAGEngine

rag = SpiegelRAGEngine()
results = rag.retrieve(
    content_description="Berlin",
    keywords=None,  # No keyword filtering
    top_k=5  # Limited results
)
print(f"Retrieved {len(results.get('chunks', []))} chunks")