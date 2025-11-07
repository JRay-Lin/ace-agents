"""
Unit tests for context module.

This module tests the Bullet and ContextPlaybook classes.
"""

import pytest
from datetime import datetime
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ace_agents.context import Bullet, ContextPlaybook


class TestBullet:
    """Test cases for the Bullet class."""

    def test_bullet_creation(self) -> None:
        """Test creating a bullet with all fields."""
        bullet = Bullet(
            id="ctx-00000001",
            content="Always validate input",
            section="strategies_and_hard_rules"
        )

        assert bullet.id == "ctx-00000001"
        assert bullet.content == "Always validate input"
        assert bullet.section == "strategies_and_hard_rules"
        assert bullet.helpful_count == 0
        assert bullet.harmful_count == 0

    def test_bullet_id_generation(self) -> None:
        """Test automatic bullet ID generation."""
        bullet_id = Bullet.generate_id()
        assert bullet_id.startswith("ctx-")
        assert len(bullet_id) == 12  # "ctx-" + 8 hex chars

    def test_mark_helpful(self) -> None:
        """Test marking a bullet as helpful."""
        bullet = Bullet(
            id="ctx-00000001",
            content="Test",
            section="general"
        )

        bullet.mark_helpful()
        assert bullet.helpful_count == 1

        bullet.mark_helpful()
        assert bullet.helpful_count == 2

    def test_mark_harmful(self) -> None:
        """Test marking a bullet as harmful."""
        bullet = Bullet(
            id="ctx-00000001",
            content="Test",
            section="general"
        )

        bullet.mark_harmful()
        assert bullet.harmful_count == 1

    def test_get_score(self) -> None:
        """Test bullet score calculation."""
        bullet = Bullet(
            id="ctx-00000001",
            content="Test",
            section="general"
        )

        assert bullet.get_score() == 0

        bullet.mark_helpful()
        bullet.mark_helpful()
        bullet.mark_harmful()

        assert bullet.get_score() == 1  # 2 - 1

    def test_bullet_serialization(self) -> None:
        """Test bullet to_dict and from_dict."""
        bullet = Bullet(
            id="ctx-00000001",
            content="Test content",
            section="general",
            helpful_count=5,
            harmful_count=2
        )

        # Convert to dict
        bullet_dict = bullet.to_dict()
        assert bullet_dict["id"] == "ctx-00000001"
        assert bullet_dict["content"] == "Test content"
        assert bullet_dict["helpful_count"] == 5

        # Convert back from dict
        bullet2 = Bullet.from_dict(bullet_dict)
        assert bullet2.id == bullet.id
        assert bullet2.content == bullet.content
        assert bullet2.helpful_count == bullet.helpful_count


class TestContextPlaybook:
    """Test cases for the ContextPlaybook class."""

    def test_empty_playbook(self) -> None:
        """Test creating an empty playbook."""
        playbook = ContextPlaybook()
        assert len(playbook) == 0
        assert len(playbook.sections) == 0

    def test_add_bullet(self) -> None:
        """Test adding bullets to playbook."""
        playbook = ContextPlaybook()

        bullet1 = Bullet(
            id="ctx-00000001",
            content="Rule 1",
            section="strategies_and_hard_rules"
        )
        playbook.add_bullet(bullet1)

        assert len(playbook) == 1
        assert "strategies_and_hard_rules" in playbook.sections

        bullet2 = Bullet(
            id="ctx-00000002",
            content="Rule 2",
            section="strategies_and_hard_rules"
        )
        playbook.add_bullet(bullet2)

        assert len(playbook) == 2
        assert len(playbook.sections["strategies_and_hard_rules"]) == 2

    def test_get_bullet(self) -> None:
        """Test retrieving a bullet by ID."""
        playbook = ContextPlaybook()

        bullet = Bullet(
            id="ctx-00000001",
            content="Test",
            section="general"
        )
        playbook.add_bullet(bullet)

        retrieved = playbook.get_bullet("ctx-00000001")
        assert retrieved is not None
        assert retrieved.id == "ctx-00000001"

        not_found = playbook.get_bullet("ctx-99999999")
        assert not_found is None

    def test_update_bullet(self) -> None:
        """Test updating a bullet."""
        playbook = ContextPlaybook()

        bullet = Bullet(
            id="ctx-00000001",
            content="Original",
            section="general"
        )
        playbook.add_bullet(bullet)

        # Update content
        success = playbook.update_bullet(
            "ctx-00000001",
            content="Updated content"
        )
        assert success is True

        updated_bullet = playbook.get_bullet("ctx-00000001")
        assert updated_bullet is not None
        assert updated_bullet.content == "Updated content"

    def test_remove_bullet(self) -> None:
        """Test removing a bullet."""
        playbook = ContextPlaybook()

        bullet = Bullet(
            id="ctx-00000001",
            content="Test",
            section="general"
        )
        playbook.add_bullet(bullet)

        assert len(playbook) == 1

        success = playbook.remove_bullet("ctx-00000001")
        assert success is True
        assert len(playbook) == 0

        # Try removing non-existent bullet
        success = playbook.remove_bullet("ctx-99999999")
        assert success is False

    def test_to_prompt(self) -> None:
        """Test converting playbook to prompt format."""
        playbook = ContextPlaybook()

        bullet1 = Bullet(
            id="ctx-00000001",
            content="Rule 1",
            section="strategies_and_hard_rules"
        )
        bullet2 = Bullet(
            id="ctx-00000002",
            content="Tip 1",
            section="troubleshooting"
        )

        playbook.add_bullet(bullet1)
        playbook.add_bullet(bullet2)

        prompt = playbook.to_prompt()

        assert "# Context Playbook" in prompt
        assert "[ctx-00000001] Rule 1" in prompt
        assert "[ctx-00000002] Tip 1" in prompt

    def test_save_and_load_json(self) -> None:
        """Test saving and loading playbook as JSON."""
        playbook = ContextPlaybook()

        bullet = Bullet(
            id="ctx-00000001",
            content="Test bullet",
            section="general"
        )
        playbook.add_bullet(bullet)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            playbook.save(temp_path, format='json')

            # Load from file
            loaded_playbook = ContextPlaybook.load(temp_path)

            assert len(loaded_playbook) == 1
            loaded_bullet = loaded_playbook.get_bullet("ctx-00000001")
            assert loaded_bullet is not None
            assert loaded_bullet.content == "Test bullet"

        finally:
            os.unlink(temp_path)

    def test_get_bullets_by_section(self) -> None:
        """Test getting bullets by section."""
        playbook = ContextPlaybook()

        bullet1 = Bullet(id="ctx-001", content="A", section="sec1")
        bullet2 = Bullet(id="ctx-002", content="B", section="sec1")
        bullet3 = Bullet(id="ctx-003", content="C", section="sec2")

        playbook.add_bullet(bullet1)
        playbook.add_bullet(bullet2)
        playbook.add_bullet(bullet3)

        sec1_bullets = playbook.get_bullets_by_section("sec1")
        assert len(sec1_bullets) == 2

        sec2_bullets = playbook.get_bullets_by_section("sec2")
        assert len(sec2_bullets) == 1

        empty_bullets = playbook.get_bullets_by_section("nonexistent")
        assert len(empty_bullets) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
