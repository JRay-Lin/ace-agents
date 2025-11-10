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
    """Represents a chat message.

    Attributes:
        role: The role of the message sender ('system', 'user', or 'assistant').
        content: The content of the message.
    """
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format.

        Returns:
            Dictionary with 'role' and 'content' keys.
        """
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Client for making LLM API calls using requests library.

    This client supports multiple LLM providers through a unified interface.
    It uses the requests library for HTTP calls, making it easy to extend
    and debug.

    Attributes:
        provider: The LLM provider name (e.g., 'openrouter', 'openai').
        base_url: The base URL for the API endpoint.
        api_key: The API key for authentication.
        model: The model identifier to use.
        default_temperature: Default temperature for sampling.
        default_max_tokens: Default maximum tokens to generate.
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
        """Initialize the LLM client.

        Args:
            provider: The LLM provider name.
            base_url: The base URL for the API.
            api_key: The API key for authentication.
            model: The model identifier.
            default_temperature: Default temperature setting. Defaults to 0.7.
            default_max_tokens: Default max tokens setting. Defaults to 2048.
        """
        self.provider = provider
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers including Authorization and Content-Type.
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
        """Build the request body for the API call.

        Args:
            messages: List of messages to send.
            temperature: Temperature for sampling. Defaults to None (uses default_temperature).
            max_tokens: Maximum tokens to generate. Defaults to None (uses default_max_tokens).
            **kwargs: Additional parameters to include in the request.

        Returns:
            Request body dictionary with model, messages, temperature, and max_tokens.
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
        """Make a chat completion API call.

        Args:
            messages: List of messages for the conversation.
            temperature: Temperature for sampling. Defaults to None.
            max_tokens: Maximum tokens to generate. Defaults to None.
            **kwargs: Additional parameters for the API.

        Returns:
            API response dictionary from the LLM provider.

        Raises:
            requests.HTTPError: If the API call fails.
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        body = self._build_request_body(messages, temperature, max_tokens, **kwargs)

        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()

        return response.json()

    def extract_content(self, response: Dict[str, Any]) -> str:
        """Extract the content from an API response.

        Args:
            response: The API response dictionary.

        Returns:
            The extracted content string from the first choice.

        Raises:
            ValueError: If content extraction fails.
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
        """Make a simple completion call with a single prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt. Defaults to None.
            temperature: Temperature for sampling. Defaults to None.
            max_tokens: Maximum tokens to generate. Defaults to None.
            **kwargs: Additional parameters for the API.

        Returns:
            The generated completion text.
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
        """Make a multi-turn conversation completion.

        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys.
            temperature: Temperature for sampling. Defaults to None.
            max_tokens: Maximum tokens to generate. Defaults to None.
            **kwargs: Additional parameters for the API.

        Returns:
            The generated completion text.
        """
        messages = [Message(role=msg["role"], content=msg["content"]) for msg in conversation]

        response = self.chat_completion(messages, temperature, max_tokens, **kwargs)
        return self.extract_content(response)

    def get_usage_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage information from API response.

        Args:
            response: The API response dictionary.

        Returns:
            Dictionary with usage statistics including prompt_tokens, completion_tokens,
            and total_tokens.
        """
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def __repr__(self) -> str:
        """Get string representation of the client.

        Returns:
            String representation showing provider and model.
        """
        return f"LLMClient(provider='{self.provider}', model='{self.model}')"
