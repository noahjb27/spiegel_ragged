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
        openai_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the selected LLM.
        
        Args:
            question: User question
            context: Retrieved context for RAG
            model: Model to use (hu-llm, gpt-4o, gpt-3.5-turbo, etc.)
            system_prompt: Optional system prompt, otherwise uses default
            temperature: Model temperature (0-1)
            preprompt: Optional text to add before the main prompt
            postprompt: Optional text to add after the main prompt
            stream: Whether to stream the response (not implemented yet)
            openai_api_key: Optional OpenAI API key for this request
            
        Returns:
            Dict with response text and metadata
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
            # Use OpenAI
            if openai_api_key and not self.openai_client:
                # Try to set API key if provided
                self.set_openai_api_key(openai_api_key)
                
            if not self.openai_client:
                raise ValueError("OpenAI API key not set or invalid")
                
            # Set default model if just "openai" was specified
            if model == "openai":
                model = "gpt-4o"
                
            try:
                chat_completion = self.openai_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
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
            # Use HU-LLM
            if not self.hu_llm_models or not self.hu_llm_models.data:
                raise ValueError("No HU-LLM models available")
            
            model_id = self.hu_llm_models.data[0].id
            
            try:
                chat_completion = self.hu_llm_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "text": chat_completion.choices[0].message.content,
                    "model": model_id,
                    "provider": "hu-llm",
                    "metadata": chat_completion.model_dump()
                }
            except Exception as e:
                logger.error(f"Error generating HU-LLM response: {e}")
                raise

    def generate_streaming_response(
        self,
        question: str,
        context: str,
        model: str = "hu-llm",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        preprompt: str = "",
        postprompt: str = "",
        openai_api_key: Optional[str] = None
    ):
        """
        Generate a streaming response from the selected LLM.
        
        Args:
            question: User question
            context: Retrieved context for RAG
            model: Model to use (hu-llm, gpt-4o, gpt-3.5-turbo, etc.)
            system_prompt: Optional system prompt
            temperature: Model temperature (0-1)
            preprompt: Optional text to add before the main prompt
            postprompt: Optional text to add after the main prompt
            openai_api_key: Optional OpenAI API key for this request
            
        Returns:
            Generator yielding response chunks
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
            # Use OpenAI
            if openai_api_key and not self.openai_client:
                # Try to set API key if provided
                self.set_openai_api_key(openai_api_key)
                
            if not self.openai_client:
                yield {"error": "OpenAI API key not set or invalid"}
                return
                
            # Set default model if just "openai" was specified
            if model == "openai":
                model = "gpt-4o"
                
            try:
                chat_completion = self.openai_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=temperature,
                    stream=True
                )
                
                response_text = ""
                
                # Process streaming response
                for chunk in chat_completion:
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        if hasattr(delta, "content") and delta.content:
                            content = delta.content
                            response_text += content
                            yield {"text": content, "complete_text": response_text, "provider": "openai"}
                
            except Exception as e:
                logger.error(f"Error generating streaming OpenAI response: {e}")
                yield {"error": str(e)}
        else:
            # Use HU-LLM
            if not self.hu_llm_models or not self.hu_llm_models.data:
                yield {"error": "No HU-LLM models available"}
                return
            
            model_id = self.hu_llm_models.data[0].id
            
            try:
                chat_completion = self.hu_llm_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=model_id,
                    temperature=temperature,
                    stream=True
                )
                
                response_text = ""
                
                # Process streaming response
                for chunk in chat_completion:
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        if hasattr(delta, "content") and delta.content:
                            content = delta.content
                            response_text += content
                            yield {"text": content, "complete_text": response_text, "provider": "hu-llm"}
                
            except Exception as e:
                logger.error(f"Error generating streaming HU-LLM response: {e}")
                yield {"error": str(e)}