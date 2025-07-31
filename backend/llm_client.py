"""
LLM Client for Ollama Integration
Supports Qwen3, OpenHermes, and other Ollama models
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with local Ollama models"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "openhermes",
                 fallback_model: str = "qwen2.5:4b"):
        self.base_url = base_url
        self.model_name = model_name
        self.fallback_model = fallback_model
        self._ready = False
        self.session: Optional[aiohttp.ClientSession] = None
        
        # System prompt for the AI assistant
        self.system_prompt = """You are a helpful AI voice assistant. You provide clear, concise, and friendly responses. Keep your answers conversational and natural since they will be spoken aloud. Avoid overly long responses unless specifically asked for detailed information."""
    
    async def _initialize(self):
        """Initialize the LLM client and check model availability"""
        try:
            self.session = aiohttp.ClientSession()
            
            logger.info(f"Checking Ollama server at {self.base_url}...")
            
            # Check if Ollama is running
            if await self._check_ollama_health():
                # Check if primary model is available
                if await self._check_model_available(self.model_name):
                    self._ready = True
                    logger.info(f"LLM client initialized with model: {self.model_name}")
                elif await self._check_model_available(self.fallback_model):
                    self.model_name = self.fallback_model
                    self._ready = True
                    logger.info(f"LLM client initialized with fallback model: {self.model_name}")
                else:
                    logger.error(f"Neither {self.model_name} nor {self.fallback_model} are available")
            else:
                logger.error("Ollama server is not running or not accessible")
                
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
    
    async def is_ready(self) -> bool:
        """Check if the LLM client is ready"""
        return self._ready
    
    async def get_response(self, user_input: str, 
                          context: Optional[str] = None,
                          max_tokens: int = 150) -> str:
        """
        Get a response from the LLM
        
        Args:
            user_input: User's input text
            context: Optional context for the conversation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            AI response text
        """
        if not self._ready or not self.session:
            return "Sorry, the AI is not available right now."
        
        try:
            # Prepare the prompt
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            if context:
                messages.append({"role": "assistant", "content": context})
            
            messages.append({"role": "user", "content": user_input})
            
            # Make request to Ollama
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    if "message" in result and "content" in result["message"]:
                        ai_response = result["message"]["content"].strip()
                        logger.info(f"Generated response: {ai_response[:50]}...")
                        return ai_response
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        return "Sorry, I couldn't generate a response."
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    return "Sorry, there was an error processing your request."
                    
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for LLM response")
            return "Sorry, the response took too long to generate."
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return "Sorry, there was an error processing your request."
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama server is healthy"""
        try:
            if not self.session:
                return False
                
            async with self.session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error checking Ollama health: {e}")
            return False
    
    async def _check_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available in Ollama"""
        try:
            if not self.session:
                return False
                
            async with self.session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    if "models" in result:
                        available_models = [model["name"] for model in result["models"]]
                        
                        # Check for exact match or partial match
                        for available_model in available_models:
                            if model_name in available_model or available_model.startswith(model_name):
                                logger.info(f"Found model: {available_model}")
                                return True
                    
                    logger.warning(f"Model {model_name} not found. Available models: {available_models}")
                    return False
                else:
                    logger.error(f"Error fetching models: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    async def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            if not self.session:
                return []
                
            async with self.session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    if "models" in result:
                        return [model["name"] for model in result["models"]]
                    
                return []
                
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            if await self._check_model_available(model_name):
                self.model_name = model_name
                logger.info(f"Switched to model: {model_name}")
                return True
            else:
                logger.error(f"Model {model_name} is not available")
                return False
                
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.session and not self.session.closed:
            asyncio.create_task(self.session.close())
