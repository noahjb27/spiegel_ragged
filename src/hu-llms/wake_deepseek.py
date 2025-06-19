# wake_deepseek.py
"""
Script to wake up DeepSeek R1 model and verify it's working.
This loads the model into memory and tests basic functionality.
"""
import requests
import json
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings

def wake_up_deepseek():
    """Wake up the DeepSeek R1 model by making a simple request."""
    print("🚀 Waking up DeepSeek R1 model...")
    print("⏳ This may take 30-90 seconds as the model loads into memory...")
    
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": settings.DEEPSEEK_R1_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please confirm you are ready to help with historical analysis. Respond briefly."
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 150  # Short response
        }
    }
    
    try:
        start_time = time.time()
        
        response = requests.post(
            url,
            json=payload,
            timeout=300,  # 5 minutes for model loading
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        
        response_data = response.json()
        generated_text = response_data.get("message", {}).get("content", "").strip()
        
        print(f"✅ DeepSeek R1 is now loaded and ready!")
        print(f"   Loading time: {elapsed:.1f} seconds")
        print(f"   Response: {generated_text[:200]}...")
        
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Timeout - the model is taking longer than expected to load")
        print("   This might happen if the server is busy. Try again in a few minutes.")
        return False
    except Exception as e:
        print(f"❌ Error waking up model: {e}")
        return False

def check_model_status():
    """Check if DeepSeek R1 is now loaded."""
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/ps", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        running_models = [model["name"] for model in data.get("models", [])]
        
        if settings.DEEPSEEK_R1_MODEL_NAME in running_models:
            print(f"✅ DeepSeek R1 is now running in memory")
            
            # Show memory usage
            for model in data.get("models", []):
                if model["name"] == settings.DEEPSEEK_R1_MODEL_NAME:
                    vram_gb = model.get("size_vram", 0) / (1024**3)
                    print(f"   VRAM usage: {vram_gb:.1f} GB")
                    break
                    
            return True
        else:
            print(f"❌ DeepSeek R1 is still not loaded")
            print(f"   Currently running: {running_models}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking model status: {e}")
        return False

def test_application_integration():
    """Test DeepSeek R1 through our application's LLM service."""
    print("\n🧪 Testing application integration...")
    
    try:
        from src.core.llm_service import LLMService
        
        llm_service = LLMService()
        
        if "deepseek-r1" not in llm_service.get_available_models():
            print("❌ DeepSeek R1 not available in LLM service")
            return False
        
        print("   Testing simple question through LLM service...")
        
        response = llm_service.generate_response(
            question="What is the capital of Germany? Please answer briefly.",
            context="Germany is a European country.",
            model="deepseek-r1",
            temperature=0.1
        )
        
        print(f"✅ Application integration test successful!")
        print(f"   Response: {response['text'].strip()}")
        print(f"   Provider: {response.get('provider')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Application integration failed: {e}")
        return False

def main():
    """Main function to wake up and test DeepSeek R1."""
    print("🎯 DeepSeek R1 Wake-Up and Test")
    print("=" * 40)
    
    # Step 1: Wake up the model
    print("\n1️⃣ Waking up DeepSeek R1...")
    if not wake_up_deepseek():
        print("⚠️  Could not wake up DeepSeek R1. Check the error above.")
        return
    
    # Step 2: Verify it's loaded
    print("\n2️⃣ Checking model status...")
    if not check_model_status():
        print("⚠️  Model doesn't appear to be loaded. Something went wrong.")
        return
    
    # Step 3: Test through application
    print("\n3️⃣ Testing application integration...")
    if test_application_integration():
        print("\n🎉 Success! DeepSeek R1 is ready to use in your application.")
        print("\n💡 Usage tips:")
        print("   • Select 'DeepSeek R1 32B (Ollama)' in the model dropdown")
        print("   • Use temperature 0.1-0.4 for analytical tasks")
        print("   • Allow 1500-3000 tokens for detailed analysis")
        print("   • Best for complex historical questions")
        print("   • First request may be slower (model loading)")
    else:
        print("\n⚠️  Application integration has issues. Check the logs above.")

if __name__ == "__main__":
    main()