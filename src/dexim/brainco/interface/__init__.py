"""
dexim.brainco.interface -- RobotInterface implementations for BrainCo Revo2 hand.

Classes:
    MockInterface:       Loopback interface for testing / simulation (no hardware).
    BrainCoInterface:    Multi-process interface via SeqLock shared memory.  This is
                         the interface the node and Brain & Gateway process use for all
                         real hardware interaction.

Note:
    The Modbus RS-485 backend lives in ``dexim.brainco.interface.backends`` --
    it is never imported by the node or the main process.
"""

__version__ = "0.1.0"

# Mock interface has no heavy deps -- always available.
# BrainCoInterface -- requires multiprocessing only, always available.
from dexim.brainco.interface.config import BrainCoHwConfig as BrainCoHwConfig
from dexim.brainco.interface.factory import build_interface as build_interface
from dexim.brainco.interface.brainco_interface import (
    BrainCoInterface as BrainCoInterface,
)
from dexim.brainco.interface.mock_interface import MockInterface as MockInterface

__all__: list[str] = []
__all__.append("MockInterface")
__all__.append("BrainCoInterface")
__all__.append("BrainCoHwConfig")
__all__.append("build_interface")
