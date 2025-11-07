# ACE-Agents: Agentic Context Engineering Framework

A Python implementation of the ACE (Agentic Context Engineering) framework for adaptive context optimization using LLM agents.

## Overview

ACE-Agents is a framework that enables LLMs to learn and maintain context through an adaptive playbook system. Instead of using static prompts or brief instructions, ACE builds a growing collection of strategies, rules, and insights that evolve through experience.

### Key Features

-   **Three Specialized Agents**:

    -   **Generator**: Produces reasoning trajectories using the context playbook
    -   **Reflector**: Analyzes outputs and extracts insights from successes and failures
    -   **Curator**: Manages playbook updates with semantic deduplication

-   **Adaptive Learning**:

    -   **Offline Adaptation**: Learn from labeled training data across multiple epochs
    -   **Online Adaptation**: Update context in real-time during inference

-   **Flexible LLM Integration**:

    -   Uses direct HTTP requests for maximum flexibility
    -   Easy integration with OpenRouter, OpenAI, Anthropic, and other providers

-   **Semantic Deduplication**:
    -   Automatic removal of redundant context using sentence embeddings
    -   Preserves high-value insights while preventing context bloat

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ace-agents.git
cd ace-agents

# Install dependencies
uv sync

# Or install with development dependencies
uv sync --extra dev
```

## Quick Start

### Basic Usage

```python
from ace_agents import AceFramework

# Initialize the framework
# Playbook will be automatically saved to data/playbook/playbook.json
ace = AceFramework(
    provider="openrouter",
    api_key="your-api-key",
    model="anthropic/claude-3.5-sonnet"
)

# Generate a response
response = ace.generate("How do I validate an email address?")
print(response)
```

### Custom Playbook Location

```python
# Use a custom directory for playbooks
ace = AceFramework(
    provider="openrouter",
    api_key="your-api-key",
    model="anthropic/claude-3.5-sonnet",
    playbook_dir="my_project/playbooks",
    playbook_name="security_playbook.json"
)
```

### Offline Adaptation

```python
# Prepare training data
training_data = [
    {
        "query": "How do I validate an email?",
        "ground_truth": "Use regex pattern ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    },
    {
        "query": "How do I hash passwords securely?",
        "ground_truth": "Use bcrypt or argon2 with sufficient work factor"
    }
]

# Train the playbook (auto-saved to data/playbook/playbook.json)
stats = ace.offline_adapt(
    training_data=training_data,
    epochs=3
)

print(f"Training complete: {stats}")
```

### Online Adaptation

```python
# The framework automatically loads existing playbook from data/playbook/playbook.json
ace = AceFramework(
    provider="openrouter",
    api_key="your-api-key",
    model="anthropic/claude-3.5-sonnet"
)

# Generate with real-time adaptation (auto-saved after update)
response = ace.online_adapt(
    query="How do I secure API keys?",
    ground_truth="Store in environment variables or vault service"
)
```

### Manual Playbook Management

```python
from ace_agents import ContextPlaybook, Bullet

# Create a playbook
playbook = ContextPlaybook()

# Add bullets
bullet = Bullet(
    id=Bullet.generate_id(),
    content="Always validate user input before processing",
    section="strategies_and_hard_rules"
)
playbook.add_bullet(bullet)

# Save playbook
playbook.save("my_playbook.json")

# Load playbook
loaded = ContextPlaybook.load("my_playbook.json")
```

## Architecture

### Context Playbook Structure

The playbook consists of "bullets" - individual pieces of context organized into sections:

```json
{
    "bullets": [
        {
            "id": "ctx-a1b2c3d4",
            "content": "Always validate email with regex before processing",
            "section": "strategies_and_hard_rules",
            "helpful_count": 5,
            "harmful_count": 0,
            "created_at": "2025-01-15T10:30:00",
            "updated_at": "2025-01-15T10:30:00",
            "metadata": {}
        }
    ]
}
```

### Agent Workflow

1. **Generator** receives a query and uses the playbook to generate a response
2. **Reflector** analyzes the response against ground truth and extracts insights
3. **Curator** converts insights into playbook updates (ADD/UPDATE/REMOVE operations)
4. Semantic deduplication removes redundant bullets

## Configuration

### API Keys

All API keys and configuration must be passed directly to the `AceFramework` constructor. The framework does **not** automatically load environment variables.

If you prefer to use environment variables, you can load them yourself:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Pass to AceFramework
ace = AceFramework(
    provider="openrouter",
    api_key=os.getenv("API_KEY"),
    model=os.getenv("ACE_MODEL", "anthropic/claude-3.5-sonnet"),
    temperature=float(os.getenv("ACE_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("ACE_MAX_TOKENS", "2048"))
)
```

### Example .env file

```bash
OPENROUTER_API_KEY=your-key-here
ACE_MODEL=anthropic/claude-3.5-sonnet
ACE_TEMPERATURE=0.7
ACE_MAX_TOKENS=2048
```

### Playbook Sections

-   `strategies_and_hard_rules`: Core strategies and mandatory rules
-   `troubleshooting`: Common issues and solutions
-   `general`: General tips and guidelines

## Examples

See the `examples/` directory for more detailed usage examples:

-   `basic_usage.py`: Complete examples of all framework features
-   `with_env_vars.py`: How to use environment variables with python-dotenv

### Using Environment Variables

1. Copy `.env.example` to `.env`:

    ```bash
    cp .env.example .env
    ```

2. Edit `.env` and fill in your API keys:

    ```bash
    API_KEY=your-actual-api-key
    ```

3. Run the example:
    ```bash
    python examples/with_env_vars.py
    ```

## Project Structure

```
ace-agents/
├── src/ace_agents/
│   ├── __init__.py
│   ├── ace_framework.py      # Main framework orchestrator
│   ├── agents.py              # Generator, Reflector, Curator agents
│   ├── context.py             # Bullet and ContextPlaybook classes
│   ├── llm_client.py          # LLM API client
│   └── utils.py               # Utility functions (semantic similarity)
├── test/
│   └── test_context.py        # Unit tests
├── examples/
│   ├── basic_usage.py         # Usage examples
│   └── with_env_vars.py       # Environment variables example
├── docs/
│   └── ticket/                # Implementation tickets
├── .env.example               # Environment variables template
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Performance

Based on the original ACE paper:

-   **10.6% average improvement** on AppWorld agent tasks
-   **86.9% reduction** in adaptation latency
-   **83.6% reduction** in token costs
-   **75.1% fewer rollouts** needed for adaptation

## References

This implementation is based on the paper:

**"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"**
arXiv:2510.04618v1

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{ace2025,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Model},
  journal={arXiv preprint arXiv:2510.04618},
  year={2025}
}
```

## Support

For issues, questions, or contributions, please open an issue on GitHub.
