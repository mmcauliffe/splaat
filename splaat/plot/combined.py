from __future__ import annotations

import pathlib
import typing

import matplotlib.pyplot as plt
from praatio import textgrid as tgio

from splaat.audio.load import load_audio
from splaat.plot.spectrogram import plot_spectrogram
from splaat.plot.textgrid import plot_textgrid
from splaat.plot.waveform import plot_waveform


def plot_file(
    file_path: typing.Union[str, pathlib.Path],
    textgrid_path: typing.Union[str, pathlib.Path] = None,
    channel: int = 0,
    start: float = 0,
    end: float = -1,
    max_frequency: int = 8000,
    window_size: typing.Union[typing.Literal["wide_band", "narrow_band"], float] = "wide_band",
    preemph: float = 0.94,
    font_size=14,
    min_prop=0.2,
    time_steps: int = 1000,
    cmap="Greys",
    figure_height=9,
    figure_width=12,
    dpi=72,
    background_color="white",
    foreground_color="black",
):
    file_path = pathlib.Path(file_path)
    if textgrid_path is None:
        textgrid_path = file_path.with_suffix(".TextGrid")
    else:
        textgrid_path = pathlib.Path(textgrid_path)
    if textgrid_path.exists():
        rows = 3
    else:
        rows = 2
    figure, axes = plt.subplots(
        rows, figsize=(figure_width, figure_height), dpi=dpi, facecolor=background_color
    )
    audio, sample_rate = load_audio(file_path)
    plot_waveform(
        audio,
        sample_rate,
        channel=channel,
        start=start,
        end=end,
        font_size=font_size,
        ax=axes[0],
        background_color=background_color,
        foreground_color=foreground_color,
    )
    axes[0].get_yaxis().set_visible(False)
    plot_spectrogram(
        audio,
        sample_rate,
        channel=channel,
        start=start,
        end=end,
        max_frequency=max_frequency,
        window_size=window_size,
        preemph=preemph,
        time_steps=time_steps,
        font_size=font_size,
        min_prop=min_prop,
        cmap=cmap,
        ax=axes[1],
        background_color=background_color,
        foreground_color=foreground_color,
    )
    axes[0].get_xaxis().set_visible(False)
    if textgrid_path.exists():
        tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=False)
        plot_textgrid(
            tg,
            start=start,
            end=end,
            font_size=font_size,
            ax=axes[2],
            background_color=background_color,
            foreground_color=foreground_color,
        )
        axes[1].get_xaxis().set_visible(False)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    return figure
