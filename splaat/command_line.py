"""Command line functions for calling the root splaat command"""
from __future__ import annotations

import rich_click as click
import pathlib
import matplotlib.pyplot as plt

from splaat.audio.load import load_audio
from splaat.plot.waveform import plot_waveform
from splaat.plot.spectrogram import plot_spectrogram

@click.command(
    name="splaat",
    help="Splaat is a utility for generating plots of audio and annotations.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Plot an annotated audio file",
)
@click.argument("audio_path", type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=pathlib.Path))
@click.argument(
    "textgrid_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=pathlib.Path)
)
@click.argument(
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    "--begin",
    "--start",
    "start",
    type=click.FLOAT, default=0
)
@click.option(
    "--end",
    type=click.FLOAT, default=-1
)
@click.option(
    "--max_frequency",
    "--max_freq",
    "max_frequency",
    type=click.INT, default=8000
)
@click.option(
    "--dpi",
    type=click.INT, default=72
)
@click.option(
    "--width",
    type=click.FLOAT, default=12
)
@click.option(
    "--height",
    type=click.FLOAT, default=4.5
)
@click.help_option("-h", "--help")
def splaat_cli(
        audio_path,
        textgrid_path,
        output_path,
        start,
        end,
        max_frequency,
        dpi,
        width,
        height,
) -> None:
    """
    Main function for the splaat command line interface
    """
    audio, sr = load_audio(audio_path)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax1 = fig.add_subplot(211)
    plot_waveform(audio, sr, start=start, end=end, axis=ax1)
    ax1.set_axis_off()
    ax2 = fig.add_subplot(212, sharex=ax1)
    plot_spectrogram(audio, sr, start=start, end=end, max_frequency=max_frequency, axis=ax2)
    plt.subplots_adjust(bottom=0.15, hspace=0)
    fig.savefig(output_path)

