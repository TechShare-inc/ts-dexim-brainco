"""Legacy Revo2 SDK seam, now lazy and fail-closed pending qualification."""

from .sdk import SDKBackend


class BrainCoModbusRS485Backend(SDKBackend):
    """Revo2 normalized position API; native activation remains gated."""

    model = "revo2"
