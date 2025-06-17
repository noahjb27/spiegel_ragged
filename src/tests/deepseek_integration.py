# test_deepseek_integration.py
"""
Test script to verify DeepSeek R1 integration with the Spiegel RAG system.
Run this to check if the DeepSeek R1 model is properly accessible.
"""
import os
import sys
import json
import requests
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.llm_service import LLMService
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """Test basic Ollama connection and model availability."""
    print("🔗 Testing Ollama Connection...")
    print(f"   Endpoint: {settings.OLLAMA_BASE_URL}")
    
    try:
        # Test /api/tags endpoint
        models_url = f"{settings.OLLAMA_BASE_URL}/api/tags"
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        
        models_data = response.json()
        available_models = [model["name"] for model in models_data.get("models", [])]
        
        print(f"✅ Ollama connection successful")
        print(f"   Available models: {len(available_models)}")
        
        # Check if DeepSeek R1 is available
        if settings.DEEPSEEK_R1_MODEL_NAME in available_models:
            print(f"✅ DeepSeek R1 model found: {settings.DEEPSEEK_R1_MODEL_NAME}")
            
            # Get model details
            for model in models_data.get("models", []):
                if model["name"] == settings.DEEPSEEK_R1_MODEL_NAME:
                    print(f"   Model details:")
                    print(f"     Size: {model.get('size', 0) / (1024**3):.1f} GB")
                    print(f"     Modified: {model.get('modified_at', 'Unknown')}")
                    print(f"     Family: {model.get('details', {}).get('family', 'Unknown')}")
                    print(f"     Parameters: {model.get('details', {}).get('parameter_size', 'Unknown')}")
                    break
            
            return True
        else:
            print(f"❌ DeepSeek R1 model not found: {settings.DEEPSEEK_R1_MODEL_NAME}")
            print(f"   Available models: {available_models}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}")
        print("   Make sure you're connected to HU-Eduroam or VPN")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False

def test_llm_service_initialization():
    """Test LLM service initialization with DeepSeek R1."""
    print("\n🤖 Testing LLM Service Initialization...")
    
    try:
        llm_service = LLMService()
        available_models = llm_service.get_available_models()
        
        print(f"✅ LLM Service initialized successfully")
        print(f"   Available models: {available_models}")
        
        if "deepseek-r1" in available_models:
            print(f"✅ DeepSeek R1 successfully registered in LLM service")
            
            # Get model info
            model_info = llm_service.get_model_info("deepseek-r1")
            print(f"   Model info: {model_info}")
            
            return True
        else:
            print(f"❌ DeepSeek R1 not found in LLM service")
            print(f"   This might indicate an initialization issue")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing LLM service: {e}")
        return False

def test_deepseek_generation():
    """Test actual text generation with DeepSeek R1."""
    print("\n📝 Testing DeepSeek R1 Text Generation...")
    
    try:
        llm_service = LLMService()
        
        if "deepseek-r1" not in llm_service.get_available_models():
            print("❌ DeepSeek R1 not available, skipping generation test")
            return False
        
        # Simple test prompt
        test_question = "Was ist die Hauptstadt von Deutschland und warum?"
        test_context = "Berlin ist seit 1990 wieder die Hauptstadt Deutschlands."
        
        print(f"   Test question: {test_question}")
        print(f"   Generating response... (this may take a moment)")
        
        response = llm_service.generate_response(
            question=test_question,
            context=test_context,
            model="deepseek-r1",
            temperature=0.3,
            max_tokens=200
        )
        
        print(f"✅ DeepSeek R1 generation successful!")
        print(f"   Response length: {len(response['text'])} characters")
        print(f"   Model used: {response.get('model_id', 'Unknown')}")
        print(f"   Provider: {response.get('provider', 'Unknown')}")
        
        # Show first part of response
        response_preview = response['text'][:200]
        if len(response['text']) > 200:
            response_preview += "..."
        print(f"   Response preview: {response_preview}")
        
        # Show metadata if available
        if 'metadata' in response and response['metadata']:
            metadata = response['metadata']
            if 'eval_duration' in metadata:
                eval_time = metadata['eval_duration'] / 1_000_000  # Convert to seconds
                print(f"   Generation time: {eval_time:.2f} seconds")
            if 'eval_count' in metadata:
                tokens_generated = metadata['eval_count']
                print(f"   Tokens generated: {tokens_generated}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DeepSeek R1 generation: {e}")
        return False

def test_health_check():
    """Test the LLM service health check functionality."""
    print("\n🏥 Testing LLM Service Health Check...")
    
    try:
        llm_service = LLMService()
        health_status = llm_service.health_check()
        
        print(f"   Overall status: {health_status.get('overall', 'unknown')}")
        
        # Check DeepSeek R1 specifically
        deepseek_status = health_status.get('providers', {}).get('deepseek-r1', {})
        if deepseek_status:
            status = deepseek_status.get('status', 'unknown')
            if status == 'healthy':
                print(f"✅ DeepSeek R1 health check: {status}")
            else:
                print(f"❌ DeepSeek R1 health check: {status}")
                error = deepseek_status.get('error', 'No error details')
                print(f"   Error: {error}")
        else:
            print(f"❌ DeepSeek R1 not found in health check results")
        
        return deepseek_status.get('status') == 'healthy'
        
    except Exception as e:
        print(f"❌ Error during health check: {e}")
        return False

def main():
    """Run all DeepSeek R1 integration tests."""
    print("🧪 DeepSeek R1 Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("LLM Service Initialization", test_llm_service_initialization),
        ("Health Check", test_health_check),
        ("Text Generation", test_deepseek_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📊 Test Results Summary:")
    print(f"{'=' * 50}")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'=' * 50}")
    if all_passed:
        print("🎉 All tests passed! DeepSeek R1 is ready to use.")
        print("\nYou can now use DeepSeek R1 in the application by:")
        print("   1. Selecting 'deepseek-r1' in the model dropdown")
        print("   2. Using it for complex analytical tasks")
        print("   3. Setting lower temperature (0.1-0.4) for best results")
    else:
        print("⚠️  Some tests failed. Please check the following:")
        print("   1. Are you connected to HU-Eduroam or VPN?")
        print("   2. Is the Ollama service running?")
        print("   3. Is the DeepSeek R1 model properly loaded in Ollama?")
        
        if not results.get("Ollama Connection", False):
            print("\n💡 Connection issues:")
            print("   - Check network connection to HU servers")
            print("   - Verify VPN is working if off-campus")
        
        if not results.get("Text Generation", False):
            print("\n💡 Generation issues:")
            print("   - Model might still be loading")
            print("   - Try again in a few minutes")

if __name__ == "__main__":
    main()