"""Separate Revo3 2.3.5 Manager/Hand fake seam; no Revo2 routing."""

from .sdk import SDKBackend


class Revo3SDKBackend(SDKBackend):
    """Revo3 degrees SDK seam, provisional mapping, hardware unqualified."""

    model = "revo3"
