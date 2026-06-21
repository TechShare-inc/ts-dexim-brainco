"""Factory for BrainCoModel."""

from __future__ import annotations


def create_model(hand_side: str = "left") -> "BrainCoModel":  # type: ignore[name-defined]  # noqa: F821
    """Create a BrainCoModel for the given hand side.

    Parameters
    ----------
    hand_side :
        ``"left"`` (default) or ``"right"``.
    """
    from dexim.brainco.model.model import BrainCoModel

    return BrainCoModel(hand_side=hand_side)
