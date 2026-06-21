"""dexim.brainco.cli - BrainCo Revo2 hand CLI command group.

Exposes:
    brainco_group:  Click group usable standalone or mounted by the umbrella.
    standalone_app: Entry point registered in ``[project.scripts]``.
"""

__version__ = "0.1.0"

import rich_click as click
from dexim.cli.common import configure_logging, print_banner, setup_error_handling

from .commands import check, config_group, model_group, probe, run, status, viz
from .devices import devices_group
from .joint import joint_group

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_COMMANDS_TABLE_COLUMN_WIDTH_RATIO = (1, 3)


@click.group(name="brainco")
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    show_default=True,
    envvar="DEXIM_LOG_LEVEL",
    help="Logging verbosity.",
)
def brainco_group(log_level: str) -> None:
    """BrainCo Revo2 dexterous hand - run, status, model, config."""
    configure_logging(log_level)


brainco_group.add_command(run)
brainco_group.add_command(status)
brainco_group.add_command(probe)
brainco_group.add_command(check)
brainco_group.add_command(viz)
brainco_group.add_command(config_group, name="config")
brainco_group.add_command(model_group, name="model")
brainco_group.add_command(joint_group, name="joint")
brainco_group.add_command(devices_group, name="devices")


def standalone_app() -> None:
    """Entry point for the standalone ``dexim-brainco`` CLI."""
    setup_error_handling()
    print_banner("BrainCo Revo2", "dexim-brainco")
    brainco_group()

