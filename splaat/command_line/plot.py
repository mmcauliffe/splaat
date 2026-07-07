"""Command line functions for calling the splaat plot command"""
from __future__ import annotations

import pathlib

import rich_click as click

from splaat.plot.combined import plot_file


@click.command(
    name="plot",
    help="Plot a sound file and optional textgrid.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Plot an audio file",
)
@click.argument(
    "audio_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=pathlib.Path),
)
@click.argument(
    "output_path", type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--textgrid_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=pathlib.Path),
    default=None,
)
@click.option("--begin", "--start", "start", type=click.FLOAT, default=0)
@click.option("--end", type=click.FLOAT, default=-1)
@click.option("--max_frequency", "--max_freq", "max_frequency", type=click.INT, default=8000)
@click.option("--dpi", type=click.INT, default=72)
@click.option("--figure_width", type=click.FLOAT, default=12)
@click.option("--figure_height", type=click.FLOAT, default=4.5)
@click.help_option("-h", "--help")
@click.pass_context
def plot_cli(context, **kwargs) -> None:
    """
    Plotting CLI function
    """
    output_path = kwargs.pop("output_path")
    figure = plot_file(**kwargs)
    figure.savefig(output_path, bbox_inches="tight")
