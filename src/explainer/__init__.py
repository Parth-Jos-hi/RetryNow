"""Explainer package: 'why this action' + customer-facing recovery copy.

Uses an LLM when a key is available (``EXPLAINER_PROVIDER`` = auto/openai/groq)
and falls back to deterministic templates otherwise — the pipeline always runs.
"""

from src.explainer.explainer import Explainer

__all__ = ["Explainer"]