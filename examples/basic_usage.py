"""
Basic usage example for ACE-Agents framework.

This example demonstrates how to use the ACE framework for adaptive
context optimization.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ace_agents import AceFramework, ContextPlaybook, Bullet
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_offline_adaptation() -> None:
    """
    Example: Offline adaptation with training data.

    If you want to use environment variables, you can load them yourself:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    """
    print("=" * 60)
    print("Example: Offline Adaptation")
    print("=" * 60)

    # Initialize ACE framework with explicit API key
    # Users should pass the API key directly
    ace = AceFramework(
        provider="openrouter",
        api_key="your-api-key-here",  # Replace with your actual API key
        model="anthropic/claude-3.5-sonnet",
        temperature=0.7,
    )

    # Sample training data
    training_data = [
        {
            "query": "How do I validate an email address?",
            "ground_truth": "Use a regex pattern like ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
        },
        {
            "query": "What's the best way to hash passwords?",
            "ground_truth": "Use bcrypt or argon2 with a sufficient work factor",
        },
        {
            "query": "How do I prevent SQL injection?",
            "ground_truth": "Use parameterized queries or prepared statements",
        },
    ]

    print(f"\nTraining with {len(training_data)} examples...")

    # Perform offline adaptation
    stats = ace.offline_adapt(
        training_data=training_data, epochs=2, save_path="playbook_trained.json"
    )

    print("\nTraining complete!")
    print(f"Statistics: {stats}")

    # Show playbook stats
    playbook_stats = ace.get_playbook_stats()
    print(f"\nPlaybook statistics: {playbook_stats}")


def example_online_adaptation() -> None:
    """
    Example: Online adaptation with single queries.
    """
    print("\n" + "=" * 60)
    print("Example: Online Adaptation")
    print("=" * 60)

    # Initialize with existing playbook
    ace = AceFramework(
        provider="openrouter",
        api_key="your-api-key-here",  # Replace with your actual API key
        model="anthropic/claude-3.5-sonnet",
        # Playbook will be automatically loaded from data/playbook/playbook.json
    )

    # Generate with adaptation
    query = "How do I securely store API keys?"
    print(f"\nQuery: {query}")

    response = ace.online_adapt(
        query=query,
        ground_truth="Store API keys in environment variables or a secure vault service",
    )

    print(f"Response: {response[:200]}...")

    # Save updated playbook
    ace.save_playbook("playbook_updated.json")
    print("\nPlaybook updated and saved")


def example_simple_generation() -> None:
    """
    Example: Simple generation without adaptation.
    """
    print("\n" + "=" * 60)
    print("Example: Simple Generation")
    print("=" * 60)

    # Create a playbook manually
    playbook = ContextPlaybook()

    bullet1 = Bullet(
        id=Bullet.generate_id(),
        content="Always validate user input before processing",
        section="strategies_and_hard_rules",
    )
    bullet2 = Bullet(
        id=Bullet.generate_id(),
        content="Use parameterized queries to prevent SQL injection",
        section="strategies_and_hard_rules",
    )

    playbook.add_bullet(bullet1)
    playbook.add_bullet(bullet2)

    # Save playbook
    playbook.save("playbook_manual.json")
    print(f"\nCreated playbook with {len(playbook)} bullets")

    # Initialize ACE with custom playbook location
    ace = AceFramework(
        provider="openrouter",
        api_key="your-api-key-here",  # Replace with your actual API key
        model="anthropic/claude-3.5-sonnet",
        playbook_dir=".",
        playbook_name="playbook_manual.json",
    )

    # Generate without adaptation
    query = "How do I build a secure login form?"
    print(f"\nQuery: {query}")

    response = ace.generate(query)
    print(f"Response: {response[:200]}...")


def example_playbook_management() -> None:
    """
    Example: Managing playbooks programmatically.
    """
    print("\n" + "=" * 60)
    print("Example: Playbook Management")
    print("=" * 60)

    # Create a new playbook
    playbook = ContextPlaybook()

    # Add bullets
    for i in range(5):
        bullet = Bullet(
            id=Bullet.generate_id(),
            content=f"Strategy {i + 1}: Important security rule",
            section="strategies_and_hard_rules",
        )
        playbook.add_bullet(bullet)

    print(f"\nCreated playbook with {len(playbook)} bullets")

    # Mark some bullets as helpful/harmful
    bullets = playbook.bullets
    bullets[0].mark_helpful()
    bullets[0].mark_helpful()
    bullets[1].mark_harmful()
    bullets[2].mark_helpful()

    # Show bullet scores
    print("\nBullet scores:")
    for bullet in bullets:
        print(
            f"  {bullet.id}: {bullet.get_score()} (H:{bullet.helpful_count}, X:{bullet.harmful_count})"
        )

    # Convert to prompt
    print("\nPlaybook as prompt:")
    print(playbook.to_prompt(include_scores=True))

    # Save and load
    playbook.save("playbook_example.json")
    loaded = ContextPlaybook.load("playbook_example.json")
    print(f"\nLoaded playbook: {len(loaded)} bullets")


def main() -> None:
    """
    Main function to run all examples.

    Note: To use environment variables, you can load them like this:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
    """
    print("\n" + "=" * 60)
    print("ACE-Agents Framework Examples")
    print("=" * 60)

    # Example 1: Playbook Management (doesn't require API key)
    example_playbook_management()

    # Note: The following examples require a valid API key
    # You need to replace "your-api-key-here" in the example functions
    # with your actual API key
    print("\n" + "=" * 60)
    print("INFO: LLM examples require a valid API key")
    print("Please replace 'your-api-key-here' in the code with your actual API key")
    print("Or load from environment variables using python-dotenv")
    print("=" * 60)


if __name__ == "__main__":
    main()
