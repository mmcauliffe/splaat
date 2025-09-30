import pandas as pd


def add_textgrid_labels(textgrid_df, spec_plot, t1=None, t2=None):
    if t1 or t2:
        textgrid_df = textgrid_df[
            ((textgrid_df["t1"] >= t1) & (textgrid_df["t1"] <= t2))
            | ((textgrid_df["t2"] >= t1) & (textgrid_df["t2"] <= t2))
        ].copy()
        min_sec = min(textgrid_df["t1_wd"])
        max_sec = max(textgrid_df["t2_wd"])

    textgrid_df["midpoint"] = (textgrid_df["t2"] + textgrid_df["t1"]) / 2
    textgrid_df["midpoint"] = pd.to_numeric(textgrid_df["midpoint"], errors="coerce")
    textgrid_df["word_mid"] = (textgrid_df["t1_wd"] + textgrid_df["t2_wd"]) / 2

    ax = spec_plot[0]
    words = []
    for index, row in textgrid_df.iterrows():
        ax.vlines(row["t1"], -1, -1000, color="red")
        if t1 or t2:
            ax.hlines(y=-1000, xmin=min_sec, xmax=max_sec, color="r", linestyle="-")
        else:
            ax.axhline(y=-1000, color="r", linestyle="-")
        ax.vlines(row["t1_wd"], -1, -2000, color="red")
        ax.annotate(
            row["phones"],
            (row["midpoint"], -500),
            textcoords="data",
            ha="center",
            fontsize="large",
        )
        if row["words"] not in words:
            ax.annotate(
                row["words"],
                (row["word_mid"], -1500),
                textcoords="data",
                ha="center",
                fontsize="large",
            )
        words.append(row["words"])
        ax.margins(x=0, y=0)
        ax.set_yticks([0, 2000, 4000, 6000])
