"""
LLM service for interfacing with HU-LLM service.
"""
import logging
import os
from typing import Dict, List, Optional, Any

import openai
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with language models."""
    
    def __init__(self):
        """Initialize LLM client."""
        # Initialize HU-LLM client
        self.client = OpenAI(
            base_url=settings.HU_LLM_API_URL,
            api_key="required-but-not-used"  # HU-LLM doesn't use API key
        )
        
        # Get available HU-LLM models
        try:
            self.models = self.client.models.list()
            logger.info(f"Found {len(self.models.data)} HU-LLM models")
        except Exception as e:
            logger.error(f"Failed to retrieve HU-LLM models: {e}")
            self.models = None
    
    def generate_response(
        self,
        question: str,
        context: str,
        model: str = "hu-llm",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        preprompt: str = "",
        postprompt: str = "",
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from the HU-LLM.
        
        Args:
            question: User question
            context: Retrieved context for RAG
            model: Model to use (ignored, always uses HU-LLM)
            system_prompt: Optional system prompt, otherwise uses default
            temperature: Model temperature (0-1)
            preprompt: Optional text to add before the main prompt
            postprompt: Optional text to add after the main prompt
            stream: Whether to stream the response (not implemented yet)
            
        Returns:
            Dict with response text and metadata
        """
        if system_prompt is None:
            system_prompt = settings.SYSTEM_PROMPTS["default"]
            
        prompt = f"""{preprompt}
        {question}

        Textauszüge:
        {context}
        {postprompt}
        """
        
        if not self.models or not self.models.data:
            raise ValueError("No HU-LLM models available")
        
        model_id = self.models.data[0].id
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=model_id,
                temperature=temperature
            )
            
            return {
                "text": chat_completion.choices[0].message.content,
                "model": model_id,
                "metadata": chat_completion.model_dump()
            }
        except Exception as e:
            logger.error(f"Error generating HU-LLM response: {e}")
            raise
            
    def generate_streaming_response(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        preprompt: str = "",
        postprompt: str = ""
    ):
        """
        Generate a streaming response from the HU-LLM.
        
        Args:
            question: User question
            context: Retrieved context for RAG
            system_prompt: Optional system prompt
            temperature: Model temperature (0-1)
            preprompt: Optional text to add before the main prompt
            postprompt: Optional text to add after the main prompt
            
        Returns:
            Generator yielding response chunks
        """
        if system_prompt is None:
            system_prompt = settings.SYSTEM_PROMPTS["default"]
            
        prompt = f"""{preprompt}
        {question}

        Textauszüge:
        {context}
        {postprompt}
        """
        
        if not self.models or not self.models.data:
            raise ValueError("No HU-LLM models available")
        
        model_id = self.models.data[0].id
        
        try:
            chat_completion = self.client.chat.completions.create(
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
                        yield {"text": content, "complete_text": response_text}
            
        except Exception as e:
            logger.error(f"Error generating streaming HU-LLM response: {e}")
            yield {"error": str(e)}