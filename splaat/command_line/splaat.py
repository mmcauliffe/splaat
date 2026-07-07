"""Command line functions for calling the root splaat command"""
from __future__ import annotations

import rich_click as click

from splaat.command_line.gui import gui_cli
from splaat.command_line.plot import plot_cli

__all__ = ["splaat_cli"]


@click.group(
    name="splaat",
    help="Splaat is CLI and GUI tool for visualizing annotated audio files.",
)
def splaat_cli() -> None:
    """
    Main function for the Splaat command line interface
    """
    pass


@click.command(
    name="version",
    short_help="Show version of Splaat",
)
def version_cli():
    try:
        from splaat._version import version
    except (ImportError, ModuleNotFoundError):
        version = None
    click.echo(version)


_commands = [
    plot_cli,
    gui_cli,
    version_cli,
]


for c in _commands:
    splaat_cli.add_command(c)

if __name__ == "__main__":
    splaat_cli()
