"""
ACE-Agents: Agentic Context Engineering Framework

This package implements the ACE (Agentic Context Engineering) framework
for adaptive context optimization using LLM agents.
"""

from .ace_framework import AceFramework
from .agents import GeneratorAgent, ReflectorAgent, CuratorAgent
from .context import Bullet, ContextPlaybook
from .llm_client import LLMClient, Message

__version__ = "0.1.0"

__all__ = [
    "AceFramework",
    "GeneratorAgent",
    "ReflectorAgent",
    "CuratorAgent",
    "Bullet",
    "ContextPlaybook",
    "LLMClient",
    "Message",
]
