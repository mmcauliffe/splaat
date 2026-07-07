import matplotlib.pyplot as plt
import pandas as pd
from praatio.data_classes.textgrid import Textgrid


def plot_textgrid(
    textgrid: Textgrid,
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
    if end < 0:
        end = textgrid.maxTimestamp
    if ax is None:
        if figure is None:
            figure = plt.figure(
                figsize=(figure_width, figure_height), dpi=dpi, facecolor=background_color
            )
        ax = figure.add_subplot(313)
    ax.set_facecolor(background_color)
    ax.hlines(y=-1000, xmin=start, xmax=end, color=foreground_color, linestyle="-")
    for tier_name in textgrid.tierNames:
        if "word" in tier_name:
            ypos = -500
        else:
            ypos = -1500
        for wi in textgrid._tierDict[tier_name].entries:
            if wi.end <= start:
                continue
            if wi.start >= end and end > 0:
                break
            midpoint = (wi.start + wi.end) / 2
            if midpoint < start:
                midpoint = (start + wi.end) / 2
            else:
                ax.vlines(wi.start, ypos + 500, ypos - 500, color=foreground_color, linestyle="-")
            if midpoint > end:
                midpoint = (wi.start + end) / 2
            else:
                ax.vlines(wi.end, ypos + 500, ypos - 500, color=foreground_color, linestyle="-")

            ax.text(
                midpoint,
                ypos,
                wi.label,
                ha="center",
                verticalalignment="center",
                fontsize="large",
                color=foreground_color,
            )
    ax.set_xlabel("Time (sec)", size=font_size)
    ax.set_yticks([-500, -1500], labels=["words", "phones"])
    ax.tick_params(labelsize=font_size, colors=foreground_color)
    ax.xaxis.label.set_color(foreground_color)
    ax.yaxis.label.set_color(foreground_color)
    ax.set_xlim(start, end)
    ax.set_ylim(-2000, 0)
    for s in ax.spines.values():
        s.set_color(foreground_color)

    return figure, ax
