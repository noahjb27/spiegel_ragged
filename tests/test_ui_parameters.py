# test_ui_parameters.py - Test the UI parameter handling fix
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.handlers.search_handlers import perform_retrieval

def test_ui_parameter_scenarios():
    """Test various UI parameter scenarios that could cause issues."""
    
    print("🧪 Testing UI parameter handling scenarios...")
    
    # Test scenarios that mimic what Gradio might send
    test_cases = [
        {
            "name": "Empty string keywords (common UI case)",
            "keywords": "",
            "expected_to_work": True
        },
        {
            "name": "String 'None' keywords (problematic case from logs)",
            "keywords": "None", 
            "expected_to_work": True
        },
        {
            "name": "Actual None keywords",
            "keywords": None,
            "expected_to_work": True  
        },
        {
            "name": "Whitespace only keywords",
            "keywords": "   ",
            "expected_to_work": True
        },
        {
            "name": "String 'null' keywords",
            "keywords": "null",
            "expected_to_work": True
        },
        {
            "name": "Valid keywords",
            "keywords": "berlin AND mauer",
            "expected_to_work": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Keywords parameter: {repr(test_case['keywords'])}")
        
        try:
            # Call perform_retrieval with the test parameters
            info_text, results = perform_retrieval(
                content_description="Berliner Mauer Berichte",
                chunk_size=3000,
                year_start=1960,
                year_end=1970,
                keywords=test_case['keywords'],  # This is the parameter we're testing
                search_in=["Text"],
                use_semantic_expansion=False,
                semantic_expansion_factor=3,
                expanded_words_json="",
                enforce_keywords=True,  # This was True in the problematic logs
                use_time_windows=False,
                time_window_size=5,
                top_k=10
            )
            
            if results and results.get('chunks'):
                num_results = len(results['chunks'])
                print(f"   ✅ SUCCESS: Retrieved {num_results} documents")
            else:
                print(f"   ⚠️  No results found (but no error - could be normal)")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            if test_case['expected_to_work']:
                print(f"   This was expected to work!")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_ui_parameter_scenarios()