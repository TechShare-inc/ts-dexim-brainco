"""
dexim.brainco -- BrainCo Revo2 dexterous hand package.

Submodules:
    dexim.brainco.model      -- kinematics, FK (BrainCoModel)
    dexim.brainco.interface  -- RobotInterface implementations (mock, RS-485)
    dexim.brainco.node       -- control node, feature extraction, retargeting
    dexim.brainco.cli        -- CLI entry point for device management and control
"""

__version__ = "0.1.0"


def register_visualizer_renderer() -> None:
    """Register BrainCo's renderer factory with the session visualizer.

    Called explicitly by visualization clients.
    If ``dexim-visualizer`` is not installed the
    import is silently skipped.
    """
    try:
        from dexim.visualizer.registry import register_renderer_factory
    except ImportError:
        return

    def lazy_renderer(*args, **kwargs):
        from dexim.brainco.viz import create_session_renderer

        return create_session_renderer(*args, **kwargs)

    register_renderer_factory("dexim-brainco", lazy_renderer)
