"""Factory for BrainCoModel (placeholder)."""

from __future__ import annotations


def create_model(hand_side: str = "left") -> "BrainCoModel":  # type: ignore[name-defined]  # noqa: F821
    """Create a BrainCoModel for the given hand side (placeholder)."""
    from dexim.brainco.model.model import BrainCoModel

    return BrainCoModel()
