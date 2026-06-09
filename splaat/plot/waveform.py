from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    channel: int = 0,
    start: float = 0,
    end: float = -1,
    font_size=14,
    figure=None,
    ax=None,
    figure_height=4.5,
    figure_width=12,
    dpi=72,
    background_color="white",
    foreground_color="black",
):
    if channel is not None and audio.ndim > 1:
        audio = audio[channel, :]
    start_sample = int(start * sample_rate)  # index of starting time: seconds to samples
    end_sample = int(end * sample_rate)  # index of ending time
    if end_sample < 0 or end_sample > len(audio):  # stop at the end of the waveform
        end_sample = len(audio)
    if start_sample > end_sample:  # don't let start follow end
        start_sample = 0
    audio_window = audio[start_sample:end_sample]
    time_axis = (
        np.arange(start_sample, end_sample) / sample_rate
    )  # this is a list of time values in seconds, from 0, to len(y)/fs
    if ax is None:
        if figure is None:
            figure = plt.figure(
                figsize=(figure_width, figure_height), dpi=dpi, facecolor=background_color
            )
        ax = figure.add_subplot(211)
    ax.set_facecolor(background_color)
    ax.plot(time_axis, audio_window, color=foreground_color)
    ax.set_xlabel("Time (sec)", size=font_size)
    ax.set_ylabel("Amplitude", size=font_size)
    ax.tick_params(labelsize=font_size, colors=foreground_color)
    ax.xaxis.label.set_color(foreground_color)
    ax.yaxis.label.set_color(foreground_color)
    ax.set_xlim(time_axis.min(), time_axis.max())
    ax.set_ylim(audio_window.min(), audio_window.max())
    for s in ax.spines.values():
        s.set_color(foreground_color)

    return figure, ax
