"""
Example showing how to use environment variables with ACE-Agents.

This demonstrates the recommended approach for loading API keys
from environment variables using python-dotenv.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ace_agents import AceFramework

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    """
    Example using environment variables for configuration.
    """
    # Load configuration from environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("ACE_MODEL", "anthropic/claude-3.5-sonnet")
    temperature = float(os.getenv("ACE_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("ACE_MAX_TOKENS", "2048"))
    playbook_dir = os.getenv("ACE_PLAYBOOK_DIR", "data/playbook")

    # Validate API key
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables")
        print("\nPlease create a .env file with the following content:")
        print("OPENROUTER_API_KEY=your-api-key-here")
        return

    # Initialize ACE framework with environment variables
    ace = AceFramework(
        provider="openrouter",
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        playbook_dir=playbook_dir
    )

    print("ACE Framework initialized successfully!")
    print(f"Model: {model}")
    print(f"Playbook directory: {playbook_dir}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")

    # Example query
    query = "How do I validate an email address in Python?"
    print(f"\nQuery: {query}")

    try:
        response = ace.generate(query)
        print(f"\nResponse: {response[:300]}...")
    except Exception as e:
        print(f"\nError: {e}")

    # Show playbook stats
    stats = ace.get_playbook_stats()
    print(f"\nPlaybook stats: {stats}")


if __name__ == "__main__":
    main()