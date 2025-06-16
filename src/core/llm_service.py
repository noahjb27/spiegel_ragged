"""
Enhanced LLM service supporting multiple providers: HU-LLM (multiple endpoints), OpenAI, and Gemini.
"""
import logging
from typing import Dict, List, Optional, Any

import openai
from openai import OpenAI
import google.generativeai as genai

from src.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Enhanced service for interacting with multiple language model providers."""
    
    def __init__(self):
        """Initialize LLM clients for all providers."""
        self.clients = {}
        self.available_models = []
        
        # Initialize HU-LLM clients
        self._init_hu_llm_clients()
        
        # Initialize OpenAI client if API key is available
        self._init_openai_client()
        
        # Initialize Gemini client if API key is available
        self._init_gemini_client()
        
        logger.info(f"LLM Service initialized with {len(self.available_models)} available models")

    def _init_hu_llm_clients(self):
        """Initialize HU-LLM clients for both endpoints."""
        hu_llm_configs = [
            ("hu-llm1", settings.HU_LLM1_API_URL),
            ("hu-llm3", settings.HU_LLM3_API_URL)
        ]
        
        for model_name, api_url in hu_llm_configs:
            try:
                client = OpenAI(
                    base_url=api_url,
                    api_key="required-but-not-used"  # HU-LLM doesn't use API key
                )
                
                # Test connection by listing models
                models = client.models.list()
                if models and models.data:
                    self.clients[model_name] = {
                        "client": client,
                        "type": "hu-llm",
                        "model_id": models.data[0].id,
                        "endpoint": api_url
                    }
                    self.available_models.append(model_name)
                    logger.info(f"✅ {model_name} initialized successfully at {api_url}")
                else:
                    logger.warning(f"⚠️ {model_name} connected but no models found")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize {model_name} at {api_url}: {e}")

    def _init_openai_client(self):
        """Initialize OpenAI client if API key is available."""
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            logger.info("OpenAI API key not configured, skipping OpenAI initialization")
            return
            
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Test connection with a simple models list call
            client.models.list()
            
            self.clients["openai-gpt4o"] = {
                "client": client,
                "type": "openai",
                "model_id": "gpt-4o",
                "endpoint": "https://api.openai.com/v1/"
            }
            self.available_models.append("openai-gpt4o")
            logger.info("✅ OpenAI GPT-4o initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")

    def _init_gemini_client(self):
        """Initialize Gemini client if API key is available."""
        if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your_gemini_api_key_here":
            logger.info("Gemini API key not configured, skipping Gemini initialization")
            return
            
        try:
            # For Gemini, we'll use the REST API through requests or a compatible client
            # For now, we'll implement a basic structure and mark it as available
            # You may need to install google-generativeai: pip install google-generativeai
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Test connection by listing models
            models = genai.list_models()
            
            self.clients["gemini-pro"] = {
                "client": genai,
                "type": "gemini",
                "model_id": "gemini-pro",
                "endpoint": "https://generativelanguage.googleapis.com/"
            }
            self.available_models.append("gemini-pro")
            logger.info("✅ Gemini Pro initialized successfully")
            
        except ImportError:
            logger.warning("⚠️ google-generativeai not installed. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return self.available_models.copy()
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model in self.clients:
            client_info = self.clients[model]
            return {
                "name": model,
                "display_name": settings.LLM_DISPLAY_NAMES.get(model, model),
                "type": client_info["type"],
                "endpoint": client_info["endpoint"],
                "available": True
            }
        return {
            "name": model,
            "display_name": settings.LLM_DISPLAY_NAMES.get(model, model),
            "available": False
        }
            
    def generate_response(
        self,
        question: str,
        context: str,
        model: str = "hu-llm3",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,  
        temperature: float = 0.3,
        preprompt: str = "",
        postprompt: str = "",
        stream: bool = False,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the selected LLM.
        
        Args:
            question: User question
            context: Context for the question
            model: Model to use (must be in available_models)
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            preprompt: Text to prepend to the question
            postprompt: Text to append to the question
            stream: Whether to stream the response
            response_format: Response format specification
            
        Returns:
            Dict containing response text, model info, and metadata
        """
        
        if system_prompt is None:
            system_prompt = settings.SYSTEM_PROMPTS["default"]
            
        # Construct the prompt
        prompt = f"""{preprompt}
        {question}

        Textauszüge:
        {context}
        {postprompt}
        """
        
        # Validate model availability
        if model not in self.clients:
            available_models = ", ".join(self.available_models)
            raise ValueError(f"Model '{model}' not available. Available models: {available_models}")
        
        client_info = self.clients[model]
        client_type = client_info["type"]
        
        try:
            if client_type == "hu-llm":
                return self._generate_hu_llm_response(
                    client_info, prompt, system_prompt, temperature, max_tokens, model
                )
            elif client_type == "openai":
                return self._generate_openai_response(
                    client_info, prompt, system_prompt, temperature, max_tokens, model, response_format
                )
            elif client_type == "gemini":
                return self._generate_gemini_response(
                    client_info, prompt, system_prompt, temperature, max_tokens, model
                )
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
                
        except Exception as e:
            logger.error(f"Error generating response with {model}: {e}")
            raise

    def _generate_hu_llm_response(
        self, 
        client_info: Dict, 
        prompt: str, 
        system_prompt: str, 
        temperature: float, 
        max_tokens: Optional[int],
        model: str
    ) -> Dict[str, Any]:
        """Generate response using HU-LLM."""
        client = client_info["client"]
        model_id = client_info["model_id"]
        
        # Build request parameters
        request_params = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "model": model_id,
            "temperature": temperature
        }
        
        # Add max_tokens if specified
        if max_tokens:
            request_params["max_tokens"] = max_tokens
        
        chat_completion = client.chat.completions.create(**request_params)
        
        return {
            "text": chat_completion.choices[0].message.content,
            "model": model,
            "model_id": model_id,
            "provider": "hu-llm",
            "endpoint": client_info["endpoint"],
            "metadata": chat_completion.model_dump()
        }

    def _generate_openai_response(
        self, 
        client_info: Dict, 
        prompt: str, 
        system_prompt: str, 
        temperature: float, 
        max_tokens: Optional[int],
        model: str,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI."""
        client = client_info["client"]
        model_id = client_info["model_id"]
        
        # Build request parameters
        request_params = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "model": model_id,
            "temperature": temperature
        }
        
        # Add max_tokens if specified
        if max_tokens:
            request_params["max_tokens"] = max_tokens
            
        # Add response_format if specified
        if response_format:
            request_params["response_format"] = response_format
        
        chat_completion = client.chat.completions.create(**request_params)
        
        return {
            "text": chat_completion.choices[0].message.content,
            "model": model,
            "model_id": model_id,
            "provider": "openai",
            "endpoint": client_info["endpoint"],
            "metadata": chat_completion.model_dump()
        }

    def _generate_gemini_response(
        self, 
        client_info: Dict, 
        prompt: str, 
        system_prompt: str, 
        temperature: float, 
        max_tokens: Optional[int],
        model: str
    ) -> Dict[str, Any]:
        """Generate response using Gemini."""
        genai = client_info["client"]
        model_id = client_info["model_id"]
        
        # Combine system prompt and user prompt for Gemini
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Create model instance
        model_instance = genai.GenerativeModel(model_id)
        
        # Generate response
        response = model_instance.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return {
            "text": response.text,
            "model": model,
            "model_id": model_id,
            "provider": "gemini",
            "endpoint": client_info["endpoint"],
            "metadata": {
                "usage": getattr(response, 'usage_metadata', {}),
                "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all configured LLM providers."""
        health_status = {
            "overall": "healthy",
            "providers": {},
            "available_models": self.available_models
        }
        
        failed_providers = []
        
        for model_name, client_info in self.clients.items():
            try:
                if client_info["type"] == "hu-llm":
                    # Test with a simple models list call
                    client_info["client"].models.list()
                    health_status["providers"][model_name] = {
                        "status": "healthy",
                        "endpoint": client_info["endpoint"]
                    }
                elif client_info["type"] == "openai":
                    # Test with a simple models list call
                    client_info["client"].models.list()
                    health_status["providers"][model_name] = {
                        "status": "healthy",
                        "endpoint": client_info["endpoint"]
                    }
                elif client_info["type"] == "gemini":
                    # Test with list models call
                    list(client_info["client"].list_models())
                    health_status["providers"][model_name] = {
                        "status": "healthy",
                        "endpoint": client_info["endpoint"]
                    }
                    
            except Exception as e:
                health_status["providers"][model_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "endpoint": client_info["endpoint"]
                }
                failed_providers.append(model_name)
        
        if failed_providers:
            health_status["overall"] = "degraded"
            if len(failed_providers) == len(self.clients):
                health_status["overall"] = "unhealthy"
        
        return health_status