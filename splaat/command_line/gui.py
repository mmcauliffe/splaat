"""Command line functions for calling the splaat GUI command"""
from __future__ import annotations

import os
import sys

import rich_click as click

from splaat.desktop.main import Application, MainWindow
from splaat.utils import configure_logger, get_temporary_directory

if sys.platform == "darwin":
    os.environ["QT_MEDIA_BACKEND"] = "darwin"


@click.command(
    name="gui",
    help="Launch the Splaat GUI.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Launch the Splaat GUI",
)
@click.help_option("-h", "--help")
@click.pass_context
def gui_cli(context, **kwargs) -> None:
    """
    GUI CLI function
    """
    configure_logger("splaat")
    configure_logger("splaat", get_temporary_directory().joinpath("splaat.log"))

    app = Application(sys.argv)
    main_window = MainWindow(**kwargs)

    app.setActiveWindow(main_window)
    main_window.show()
    sys.exit(app.exec())
