"""
Main ACE Framework module.

This module implements the AceFramework class which orchestrates the
three agents (Generator, Reflector, Curator) for adaptive context engineering.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from .llm_client import LLMClient
from .agents import GeneratorAgent, ReflectorAgent, CuratorAgent
from .context import ContextPlaybook
import logging


class AceFramework:
    """Main framework for Agentic Context Engineering.

    The AceFramework orchestrates the three core agents to enable
    adaptive context optimization through offline and online adaptation.

    Attributes:
        llm_client: LLM client instance for API communication.
        generator: Generator agent for response generation.
        reflector: Reflector agent for quality assessment.
        curator: Curator agent for playbook management.
        playbook_dir: Directory path for storing playbooks.
        playbook_name: Name of the playbook file.
        playbook_path: Full path to the playbook file.
        playbook: Context playbook instance.
        logger: Logger instance for framework operations.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        playbook_dir: str = "data/playbook",
        playbook_name: str = "playbook.json",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the ACE framework.

        Args:
            provider: LLM provider name (e.g., 'openrouter', 'openai').
            api_key: API key for authentication.
            model: Model identifier to use.
            playbook_dir: Directory path for playbooks. Defaults to 'data/playbook'.
            playbook_name: Playbook filename. Defaults to 'playbook.json'.
            temperature: Default temperature for generation. Defaults to 0.7.
            max_tokens: Default maximum tokens for generation. Defaults to 2048.
        """
        # Initialize LLM client
        base_url_dict = {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1/responses",
        }
        self.llm_client = LLMClient(
            provider=provider,
            base_url=base_url_dict[provider],
            api_key=api_key,
            model=model,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

        # Initialize agents
        self.generator = GeneratorAgent(self.llm_client)
        self.reflector = ReflectorAgent(self.llm_client)
        self.curator = CuratorAgent(self.llm_client)

        # Setup playbook path
        self.playbook_dir = Path(playbook_dir)
        self.playbook_name = playbook_name
        self.playbook_path = self.playbook_dir / playbook_name

        # Create playbook directory if it doesn't exist
        self.playbook_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load playbook
        if self.playbook_path.exists():
            self.playbook = ContextPlaybook.load(str(self.playbook_path))
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Loaded existing playbook from {self.playbook_path}")
        else:
            self.playbook = ContextPlaybook()
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Initialized new playbook (will save to {self.playbook_path})")

    def offline_adapt(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 3,
        auto_save: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Perform offline adaptation on training data.

        This method iteratively improves the context playbook using
        labeled training examples. It runs multiple epochs of generation,
        reflection, and curation.

        Args:
            training_data: List of training examples with 'query' and 'ground_truth' keys.
            epochs: Number of training epochs. Defaults to 3.
            auto_save: Whether to automatically save playbook after training. Defaults to True.
            **kwargs: Additional parameters passed to agent methods.

        Returns:
            Dictionary with adaptation statistics including:
                - total_examples: Number of training examples processed
                - epochs: Number of epochs completed
                - bullets_added: Total bullets added to playbook
                - bullets_updated: Total bullets updated in playbook
                - bullets_removed: Total bullets removed from playbook
        """
        stats = {
            "total_examples": len(training_data),
            "epochs": epochs,
            "bullets_added": 0,
            "bullets_updated": 0,
            "bullets_removed": 0,
        }

        self.logger.info(
            f"Starting offline adaptation: {len(training_data)} examples, {epochs} epochs"
        )

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_stats = self._run_adaptation_epoch(training_data, **kwargs)

            # Update stats
            stats["bullets_added"] += epoch_stats.get("bullets_added", 0)
            stats["bullets_updated"] += epoch_stats.get("bullets_updated", 0)
            stats["bullets_removed"] += epoch_stats.get("bullets_removed", 0)

            # Deduplicate after each epoch
            removed = self.curator.deduplicate(self.playbook)
            stats["bullets_removed"] += removed

            self.logger.info(f"Epoch {epoch + 1} complete. Playbook size: {len(self.playbook)}")

        # Auto-save adapted playbook to default path
        if auto_save:
            self.playbook.save(str(self.playbook_path))
            self.logger.info(f"Playbook auto-saved to {self.playbook_path}")

        return stats

    def _run_adaptation_epoch(
        self, training_data: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Run a single adaptation epoch.

        Args:
            training_data: List of training examples with 'query' and optional 'ground_truth'.
            **kwargs: Additional parameters passed to agent methods.

        Returns:
            Dictionary with epoch statistics including bullets_added, bullets_updated,
            and bullets_removed counts.
        """
        stats = {
            "bullets_added": 0,
            "bullets_updated": 0,
            "bullets_removed": 0,
        }

        for i, example in enumerate(training_data):
            query = example["query"]
            ground_truth = example.get("ground_truth")

            # Generate response
            try:
                response, bullet_feedback = self.generator.generate(
                    query=query, playbook=self.playbook, **kwargs
                )

                # Reflect on the generation
                insights = self.reflector.reflect(
                    query=query,
                    trajectory=response,
                    ground_truth=ground_truth,
                    playbook=self.playbook,
                    **kwargs,
                )

                # Curate updates
                deltas = self.curator.curate(insights=insights, playbook=self.playbook, **kwargs)

                # Apply deltas
                self.curator.apply_deltas(deltas, self.playbook)

                # Update stats
                for delta in deltas:
                    op = delta.get("operation")
                    if op == "ADD":
                        stats["bullets_added"] += 1
                    elif op == "UPDATE":
                        stats["bullets_updated"] += 1
                    elif op == "REMOVE":
                        stats["bullets_removed"] += 1

            except Exception as e:
                self.logger.error(f"Error processing example {i}: {e}")
                continue

        return stats

    def online_adapt(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        auto_save: bool = True,
        **kwargs: Any,
    ) -> str:
        """Perform online adaptation with a single query.

        This method generates a response and optionally updates the playbook
        based on execution feedback or ground truth.

        Args:
            query: The user query to process.
            ground_truth: Ground truth answer for evaluation. Defaults to None.
            execution_result: Execution feedback from running the response. Defaults to None.
            auto_save: Whether to automatically save playbook after update. Defaults to True.
            **kwargs: Additional parameters passed to agent methods.

        Returns:
            Generated response string.
        """
        # Generate response
        response, bullet_feedback = self.generator.generate(
            query=query, playbook=self.playbook, **kwargs
        )

        # If feedback is available, update playbook
        if ground_truth or execution_result:
            insights = self.reflector.reflect(
                query=query,
                trajectory=response,
                ground_truth=ground_truth,
                execution_result=execution_result,
                playbook=self.playbook,
                **kwargs,
            )

            deltas = self.curator.curate(insights=insights, playbook=self.playbook, **kwargs)

            self.curator.apply_deltas(deltas, self.playbook)

            # Auto-save updated playbook
            if auto_save:
                self.playbook.save(str(self.playbook_path))
                self.logger.info(f"Playbook auto-saved to {self.playbook_path}")

        return response

    def generate(self, query: str, **kwargs: Any) -> str:
        """Generate a response without adaptation.

        Args:
            query: The user query to process.
            **kwargs: Additional parameters passed to the generator agent.

        Returns:
            Generated response string.
        """
        response, _ = self.generator.generate(query=query, playbook=self.playbook, **kwargs)
        return response

    def save_playbook(self, path: str, format: str = "json") -> None:
        """Save the current playbook.

        Args:
            path: File path to save the playbook to.
            format: File format for serialization ('json' or 'yaml'). Defaults to 'json'.
        """
        self.playbook.save(path, format)

    def load_playbook(self, path: str) -> None:
        """Load a playbook from file.

        Args:
            path: File path to load the playbook from.
        """
        self.playbook = ContextPlaybook.load(path)

    def get_playbook_stats(self) -> Dict[str, Any]:
        """Get statistics about the current playbook.

        Returns:
            Dictionary with playbook statistics including:
                - total_bullets: Total number of bullets across all sections
                - sections: Number of sections in the playbook
                - section_details: Dictionary mapping section names to bullet counts
        """
        return {
            "total_bullets": len(self.playbook),
            "sections": len(self.playbook.sections),
            "section_details": {
                section: len(bullets) for section, bullets in self.playbook.sections.items()
            },
        }

    def __repr__(self) -> str:
        """Get string representation of the framework.

        Returns:
            String representation showing provider, model, and playbook size.
        """
        return (
            f"AceFramework(provider='{self.llm_client.provider}', "
            f"model='{self.llm_client.model}', "
            f"playbook_bullets={len(self.playbook)})"
        )
