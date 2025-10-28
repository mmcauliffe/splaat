import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(audio, sample_rate, start=0, end=-1, channel=0, axis=None, font_size=14, figure_width=12, figure_height=4.5, dpi =72,show_time=True,**kwargs):


    if audio.ndim > 1:
        audio = audio[channel, :]
    start_sample = int(start * sample_rate)  # index of starting time: seconds to samples
    end_sample = int(end * sample_rate)  # index of ending time
    if end_sample < 0 or end_sample > len(audio):  # stop at the end of the waveform
        end_sample = len(audio)
    if start_sample > end_sample:  # don't let start follow end
        start_sample = 0
    audio = audio[start_sample:end_sample]
    times = np.arange(len(audio)) / sample_rate
    times = np.add(times, start)
    if axis is None:
        fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
        axis = fig.add_subplot(111)
    axis.plot(times, audio, **kwargs)

    return axis