"""
LLM Client module for interfacing with various LLM providers.

This module provides a unified interface for making API calls to different
LLM providers (OpenRouter, OpenAI, Anthropic, etc.) using the requests library.
"""

from typing import List, Dict, Any, Optional
import requests
import json
from dataclasses import dataclass


@dataclass
class Message:
    """
    @brief Represents a chat message.

    @param role The role of the message sender ('system', 'user', or 'assistant')
    @param content The content of the message
    """
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """
        @brief Convert message to dictionary format.

        @return Dictionary representation of the message
        """
        return {"role": self.role, "content": self.content}


class LLMClient:
    """
    @brief Client for making LLM API calls using requests library.

    This client supports multiple LLM providers through a unified interface.
    It uses the requests library for HTTP calls, making it easy to extend
    and debug.

    @param provider The LLM provider name (e.g., 'openrouter', 'openai')
    @param base_url The base URL for the API endpoint
    @param api_key The API key for authentication
    @param model The model identifier to use
    @param default_temperature Default temperature for sampling
    @param default_max_tokens Default maximum tokens to generate
    """

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        default_temperature: float = 0.7,
        default_max_tokens: int = 2048
    ) -> None:
        """
        @brief Initialize the LLM client.

        @param provider The LLM provider name
        @param base_url The base URL for the API
        @param api_key The API key
        @param model The model identifier
        @param default_temperature Default temperature setting
        @param default_max_tokens Default max tokens setting
        """
        self.provider = provider
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def _get_headers(self) -> Dict[str, str]:
        """
        @brief Get HTTP headers for API requests.

        @return Dictionary of headers
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        @brief Build the request body for the API call.

        @param messages List of messages to send
        @param temperature Temperature for sampling
        @param max_tokens Maximum tokens to generate
        @param kwargs Additional parameters
        @return Request body dictionary
        """
        body = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }

        # Add any additional parameters
        body.update(kwargs)

        return body

    def chat_completion(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        @brief Make a chat completion API call.

        @param messages List of messages for the conversation
        @param temperature Temperature for sampling (optional)
        @param max_tokens Maximum tokens to generate (optional)
        @param kwargs Additional parameters for the API
        @return API response dictionary
        @throws requests.HTTPError If the API call fails
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        body = self._build_request_body(messages, temperature, max_tokens, **kwargs)

        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()

        return response.json()

    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        @brief Extract the content from an API response.

        @param response The API response dictionary
        @return The extracted content string
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract content from response: {e}")

    def simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        @brief Make a simple completion call with a single prompt.

        @param prompt The user prompt
        @param system_prompt Optional system prompt
        @param temperature Temperature for sampling (optional)
        @param max_tokens Maximum tokens to generate (optional)
        @param kwargs Additional parameters for the API
        @return The generated completion text
        """
        messages = []

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        messages.append(Message(role="user", content=prompt))

        response = self.chat_completion(messages, temperature, max_tokens, **kwargs)
        return self.extract_content(response)

    def multi_turn_completion(
        self,
        conversation: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        @brief Make a multi-turn conversation completion.

        @param conversation List of message dictionaries with 'role' and 'content'
        @param temperature Temperature for sampling (optional)
        @param max_tokens Maximum tokens to generate (optional)
        @param kwargs Additional parameters for the API
        @return The generated completion text
        """
        messages = [Message(role=msg["role"], content=msg["content"]) for msg in conversation]

        response = self.chat_completion(messages, temperature, max_tokens, **kwargs)
        return self.extract_content(response)

    def get_usage_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        """
        @brief Extract token usage information from API response.

        @param response The API response dictionary
        @return Dictionary with usage statistics
        """
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def __repr__(self) -> str:
        """
        @brief Get string representation of the client.

        @return String representation
        """
        return f"LLMClient(provider='{self.provider}', model='{self.model}')"
