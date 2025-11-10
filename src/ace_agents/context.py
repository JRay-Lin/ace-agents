"""
Context management module for ACE framework.

This module provides data structures for managing context playbooks,
including bullets (context items) and the overall playbook structure.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import yaml
import uuid


@dataclass
class Bullet:
    """A single context item (bullet) in the playbook.

    A Bullet represents a strategy, insight, or rule that has been learned
    through the ACE adaptation process. Each bullet tracks its effectiveness
    through helpful and harmful counters.

    Attributes:
        id: Unique identifier for the bullet (e.g., "ctx-00001").
        content: The actual strategy, insight, or rule text.
        section: Category/section this bullet belongs to.
        helpful_count: Number of times this bullet was marked as helpful.
        harmful_count: Number of times this bullet was marked as harmful.
        created_at: Timestamp when this bullet was created.
        updated_at: Timestamp when this bullet was last updated.
        metadata: Additional metadata for the bullet.
    """

    id: str
    content: str
    section: str
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique bullet ID.

        Returns:
            A unique bullet identifier string in format 'ctx-XXXXXXXX'.
        """
        return f"ctx-{uuid.uuid4().hex[:8]}"

    def mark_helpful(self) -> None:
        """Increment the helpful counter for this bullet."""
        self.helpful_count += 1
        self.updated_at = datetime.now()

    def mark_harmful(self) -> None:
        """Increment the harmful counter for this bullet."""
        self.harmful_count += 1
        self.updated_at = datetime.now()

    def get_score(self) -> int:
        """Calculate the net score of this bullet.

        Returns:
            The score computed as (helpful_count - harmful_count).
        """
        return self.helpful_count - self.harmful_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert bullet to dictionary format.

        Returns:
            Dictionary representation of the bullet with ISO-formatted timestamps.
        """
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bullet":
        """Create a Bullet instance from dictionary data.

        Args:
            data: Dictionary containing bullet data with ISO-formatted timestamps.

        Returns:
            A new Bullet instance.
        """
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class ContextPlaybook:
    """Manages a collection of bullets organized by sections.

    The ContextPlaybook is the core data structure for storing and managing
    context learned through the ACE framework. It supports operations like
    adding, updating, removing bullets, and converting the playbook to
    prompt format for LLM consumption.

    Attributes:
        bullets: List of all bullets in the playbook.
        sections: Dictionary mapping section names to lists of bullets.
    """

    def __init__(self) -> None:
        """Initialize an empty context playbook."""
        self.bullets: List[Bullet] = []
        self.sections: Dict[str, List[Bullet]] = {}

    def add_bullet(self, bullet: Bullet) -> None:
        """Add a new bullet to the playbook.

        Args:
            bullet: The bullet to add.
        """
        self.bullets.append(bullet)

        if bullet.section not in self.sections:
            self.sections[bullet.section] = []
        self.sections[bullet.section].append(bullet)

    def update_bullet(
        self,
        bullet_id: str,
        content: Optional[str] = None,
        section: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing bullet.

        Args:
            bullet_id: The ID of the bullet to update.
            content: New content for the bullet. Defaults to None.
            section: New section for the bullet. Defaults to None.
            metadata: New metadata for the bullet. Defaults to None.

        Returns:
            True if bullet was found and updated, False otherwise.
        """
        bullet = self.get_bullet(bullet_id)
        if bullet is None:
            return False

        # Update fields if provided
        if content is not None:
            bullet.content = content
        if metadata is not None:
            bullet.metadata = metadata

        # Handle section change
        if section is not None and section != bullet.section:
            # Remove from old section
            if bullet.section in self.sections:
                self.sections[bullet.section].remove(bullet)

            # Update section
            bullet.section = section

            # Add to new section
            if section not in self.sections:
                self.sections[section] = []
            self.sections[section].append(bullet)

        bullet.updated_at = datetime.now()
        return True

    def remove_bullet(self, bullet_id: str) -> bool:
        """Remove a bullet from the playbook.

        Args:
            bullet_id: The ID of the bullet to remove.

        Returns:
            True if bullet was found and removed, False otherwise.
        """
        bullet = self.get_bullet(bullet_id)
        if bullet is None:
            return False

        # Remove from bullets list
        self.bullets.remove(bullet)

        # Remove from section
        if bullet.section in self.sections:
            self.sections[bullet.section].remove(bullet)
            if len(self.sections[bullet.section]) == 0:
                del self.sections[bullet.section]

        return True

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        """Retrieve a bullet by its ID.

        Args:
            bullet_id: The ID of the bullet to retrieve.

        Returns:
            The bullet if found, None otherwise.
        """
        for bullet in self.bullets:
            if bullet.id == bullet_id:
                return bullet
        return None

    def get_bullets_by_section(self, section: str) -> List[Bullet]:
        """Get all bullets in a specific section.

        Args:
            section: The section name.

        Returns:
            List of bullets in the section, or empty list if section doesn't exist.
        """
        return self.sections.get(section, [])

    def to_prompt(self, include_scores: bool = False) -> str:
        """Convert the playbook to a formatted prompt string for LLM.

        Args:
            include_scores: Whether to include helpful/harmful scores. Defaults to False.

        Returns:
            Formatted markdown string representation of the playbook.
        """
        if not self.bullets:
            return "No context available."

        lines = ["# Context Playbook\n"]

        for section, bullets in self.sections.items():
            if not bullets:
                continue

            lines.append(f"\n## {section.replace('_', ' ').title()}\n")

            for bullet in bullets:
                score_text = ""
                if include_scores:
                    score = bullet.get_score()
                    score_text = f" [Score: {score}]"

                lines.append(f"- [{bullet.id}] {bullet.content}{score_text}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert playbook to dictionary format.

        Returns:
            Dictionary representation with 'bullets' and 'sections' keys.
        """
        return {
            "bullets": [bullet.to_dict() for bullet in self.bullets],
            "sections": list(self.sections.keys())
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextPlaybook":
        """Create a ContextPlaybook from dictionary data.

        Args:
            data: Dictionary containing playbook data with 'bullets' list.

        Returns:
            A new ContextPlaybook instance.
        """
        playbook = cls()
        for bullet_data in data.get("bullets", []):
            bullet = Bullet.from_dict(bullet_data)
            playbook.add_bullet(bullet)
        return playbook

    def save(self, path: str, format: str = "json") -> None:
        """Save the playbook to a file.

        Args:
            path: File path to save to.
            format: File format ('json' or 'yaml'). Defaults to 'json'.

        Raises:
            ValueError: If format is not 'json' or 'yaml'.
        """
        data = self.to_dict()

        with open(path, 'w', encoding='utf-8') as f:
            if format.lower() == "json":
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: str) -> "ContextPlaybook":
        """Load a playbook from a file.

        Args:
            path: File path to load from (must end with .json, .yaml, or .yml).

        Returns:
            A new ContextPlaybook instance.

        Raises:
            ValueError: If file extension is not supported.
        """
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                data = json.load(f)
            elif path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file extension: {path}")

        return cls.from_dict(data)

    def __len__(self) -> int:
        """Get the number of bullets in the playbook.

        Returns:
            Number of bullets.
        """
        return len(self.bullets)

    def __repr__(self) -> str:
        """Get string representation of the playbook.

        Returns:
            String representation showing bullet and section counts.
        """
        return f"ContextPlaybook(bullets={len(self.bullets)}, sections={len(self.sections)})"
