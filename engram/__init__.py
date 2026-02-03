"""
Engram — Neuroscience-grounded memory for AI agents.

Usage:
    from engram import Memory

    # Basic usage (FTS5 only)
    mem = Memory("./agent.db")
    mem.add("the sky is blue", type="factual")
    results = mem.recall("sky color")
    mem.consolidate()
    
    # With embeddings (recommended for semantic search)
    from engram.embeddings import OpenAIAdapter
    
    mem = Memory("./agent.db", embedding=OpenAIAdapter())
    # or simply:
    mem = Memory("./agent.db", embedding="openai")
"""

from engram.memory import Memory
from engram.config import MemoryConfig
from engram.core import MemoryType, MemoryLayer, MemoryEntry, MemoryStore

__all__ = ["Memory", "MemoryConfig", "MemoryType", "MemoryLayer", "MemoryEntry", "MemoryStore"]
__version__ = "0.2.0"  # Bumped for embedding support
