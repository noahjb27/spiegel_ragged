"""
LLM service for interfacing with HU-LLM service.
"""
import logging
from typing import Dict, List, Optional, Any

import openai
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with language models."""
    
    def __init__(self):
        """Initialize LLM clients."""
        # Initialize HU-LLM client
        self.hu_llm_client = OpenAI(
            base_url=settings.HU_LLM_API_URL,
            api_key="required-but-not-used"  # HU-LLM doesn't use API key
        )
        
        # Get available HU-LLM models
        try:
            self.hu_llm_models = self.hu_llm_client.models.list()
            logger.info(f"Found {len(self.hu_llm_models.data)} HU-LLM models")
        except Exception as e:
            logger.error(f"Failed to retrieve HU-LLM models: {e}")
            self.hu_llm_models = None
            
        # Initialize OpenAI client with None - will be set when key is provided
        self.openai_client = None

    def set_openai_api_key(self, api_key: str) -> bool:
        """
        Set or update the OpenAI API key.
        
        Args:
            api_key: OpenAI API key
            
        Returns:
            bool: True if key was set successfully
        """
        if not api_key.strip():
            logger.warning("Empty OpenAI API key provided")
            self.openai_client = None
            return False
            
        try:
            # Create new client with provided key
            self.openai_client = OpenAI(api_key=api_key)
            
            # Test connection with a simple models list call
            self.openai_client.models.list()
            logger.info(f"Successfully set OpenAI API key")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client with provided key: {e}")
            self.openai_client = None
            return False
            
    def generate_response(
        self,
        question: str,
        context: str,
        model: str = "hu-llm",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,  
        temperature: float = 0.3,
        preprompt: str = "",
        postprompt: str = "",
        stream: bool = False,
        openai_api_key: Optional[str] = None,
        response_format: Optional[Dict] = None  # Add this parameter
    ) -> Dict[str, Any]:
        """
        Generate a response from the selected LLM with optional response format.
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
        
        # Process request based on model
        if model.startswith("gpt-") or model == "openai":
            if openai_api_key and not self.openai_client:
                self.set_openai_api_key(openai_api_key)
                
            if not self.openai_client:
                raise ValueError("OpenAI API key not set or invalid")
                
            if model == "openai":
                model = "gpt-4o"
                
            try:
                # Build request parameters
                request_params = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "model": model,
                    "temperature": temperature
                }
                
                # Add max_tokens if specified
                if max_tokens:
                    request_params["max_tokens"] = max_tokens
                    
                # Add response_format for OpenAI if specified
                if response_format:
                    request_params["response_format"] = response_format
                
                chat_completion = self.openai_client.chat.completions.create(**request_params)
                
                return {
                    "text": chat_completion.choices[0].message.content,
                    "model": model,
                    "provider": "openai",
                    "metadata": chat_completion.model_dump()
                }
            except Exception as e:
                logger.error(f"Error generating OpenAI response: {e}")
                raise
        else:
            # Use HU-LLM - don't use format parameter as it's not supported
            if not self.hu_llm_models or not self.hu_llm_models.data:
                raise ValueError("No HU-LLM models available")
            
            model_id = self.hu_llm_models.data[0].id
            
            try:
                # Build request parameters without format
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
                
                # Note: HU-LLM doesn't support format parameter
                chat_completion = self.hu_llm_client.chat.completions.create(**request_params)
                
                return {
                    "text": chat_completion.choices[0].message.content,
                    "model": model_id,
                    "provider": "hu-llm",
                    "metadata": chat_completion.model_dump()
                }
            except Exception as e:
                logger.error(f"Error generating HU-LLM response: {e}")
                raise
