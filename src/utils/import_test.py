# src/utils/import_test.py
def test_imports():
    """Test that all required packages are available."""
    try:
        from langchain_chroma import Chroma
        print("✅ langchain_chroma import successful")
    except ImportError:
        print("❌ Failed to import Chroma from langchain_chroma")
        
    try:
        from langchain_ollama import OllamaEmbeddings
        print("✅ langchain_ollama import successful")
    except ImportError:
        print("❌ Failed to import OllamaEmbeddings from langchain_ollama")

if __name__ == "__main__":
    test_imports()