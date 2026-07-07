from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Queue
from threading import Lock

import librosa
import numpy as np
import scipy
import scipy.signal
import soundfile
import sqlalchemy
from PySide6 import QtCore
from sqlalchemy.orm import joinedload, selectinload

from splaat.db import File, SoundFile, SqlBase, Utterance, WordInterval
from splaat.desktop.settings import SplaatSettings
from splaat.textgrid import parse_file_to_db

logger = logging.getLogger("splaat")


class WorkerSignals(QtCore.QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    """

    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)
    stream_result = QtCore.Signal(object)
    progress = QtCore.Signal(int, str)
    total = QtCore.Signal(int)

    def __init__(self, name):
        super().__init__()
        self.name = name


class ProgressCallback:
    """
    Class for sending progress indications back to the main process
    """

    def __init__(self, callback=None, total_callback=None):
        self._total = 0
        self.callback = callback
        self.total_callback = total_callback
        self._progress = 0
        self.callback_interval = 1
        self.lock = threading.Lock()
        self.start_time = None

    @property
    def total(self) -> int:
        """Total entries to process"""
        with self.lock:
            return self._total

    @property
    def progress(self) -> int:
        """Current number of entries processed"""
        with self.lock:
            return self._progress

    @property
    def progress_percent(self) -> float:
        """Current progress as percetage"""
        with self.lock:
            if not self._total:
                return 0.0
            return self._progress / self._total

    def update_total(self, total: int) -> None:
        """
        Update the total for the callback

        Parameters
        ----------
        total: int
            New total
        """
        with self.lock:
            if self._total == 0 and total != 0:
                self.start_time = time.time()
            self._total = total
            if self.total_callback is not None:
                self.total_callback(self._total)

    def set_progress(self, progress: int) -> None:
        """
        Update the number of entries processed for the callback

        Parameters
        ----------
        progress: int
            New progress
        """
        with self.lock:
            self._progress = progress

    def increment_progress(self, increment: int) -> None:
        """
        Increment the number of entries processed for the callback

        Parameters
        ----------
        increment: int
            Update the progress by this amount
        """
        with self.lock:
            self._progress += increment
            if self.callback is not None:
                current_time = time.time()
                current_duration = current_time - self.start_time
                time_per_iteration = current_duration / self._progress
                remaining_iterations = self._total - self._progress
                remaining_time = datetime.timedelta(
                    seconds=int(time_per_iteration * remaining_iterations)
                )
                self.callback(self._progress, str(remaining_time))


class Worker(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    name = "N/A"

    def __init__(self, *args, use_mp=False, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.args = args
        self.kwargs = kwargs
        self.stopped = threading.Event()
        self.settings = SplaatSettings()
        self.signals = WorkerSignals(self.name)
        self.use_mp = use_mp

        # Add the callback to our kwargs
        self.progress_callback = ProgressCallback(
            callback=self.signals.progress.emit, total_callback=self.signals.total.emit
        )
        if not use_mp:
            self.kwargs["progress_callback"] = self.progress_callback
        self.kwargs["stopped"] = self.stopped

    def _run(self):
        pass

    def cancel(self):
        self.stopped.set()

    @QtCore.Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            if self.use_mp:
                queue = Queue()
                kwargs = {}
                kwargs.update(self.kwargs)
                kwargs["queue"] = queue
                p = threading.Thread(target=self._run)
                p.start()
                result = queue.get()
                p.join()
                if isinstance(result, Exception):
                    raise result
            else:
                result = self._run()
        except Exception:
            exctype, value = sys.exc_info()[:2]

            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class QueryUtterancesWorker(Worker):
    def __init__(self, session, use_mp=False, **kwargs):
        super().__init__(use_mp=use_mp, **kwargs)
        self.session = session

    def _run(self):
        with self.session() as session:
            count_only = self.kwargs.get("count", False)
            columns = [
                Utterance.id,
                File.id,
                File.relative_path,
                File.name,
                SoundFile.duration,
                Utterance.text,
                Utterance.phone_text,
            ]
            sort_index = self.kwargs.get("sort_index", None)
            files = session.query(*columns).join(Utterance.file).join(File.sound_file)

            if count_only:
                return files.count()
            if self.progress_callback is not None:
                self.progress_callback.update_total(self.kwargs.get("limit", 100))
            if sort_index is not None and sort_index + 2 <= len(columns) - 1:
                sort_column = columns[sort_index + 2]
                if self.kwargs.get("sort_desc", False):
                    sort_column = sort_column.desc()
                utterances = files.order_by(sort_column, Utterance.id)
            else:
                utterances = files.order_by(File.name, Utterance.start)
            utterances = utterances.limit(self.kwargs.get("limit", 100)).offset(
                self.kwargs.get("current_offset", 0)
            )
            data = []
            indices = []
            file_indices = []
            reversed_indices = {}
            for i, u in enumerate(utterances):
                if self.stopped is not None and self.stopped.is_set():
                    return
                data.append(list(u[2:]))
                indices.append(u[0])
                file_indices.append(u[1])
                reversed_indices[u[0]] = i
                if self.progress_callback is not None:
                    self.progress_callback.increment_progress(1)
        return data, indices, file_indices, reversed_indices


class FileUtterancesWorker(Worker):
    def __init__(self, session, file_id, use_mp=False, **kwargs):
        super().__init__(use_mp=use_mp, **kwargs)
        self.session = session
        self.file_id = file_id

    def _run(self):
        utterances = (
            self.session.query(Utterance)
            .options(
                selectinload(Utterance.phone_intervals),
                selectinload(Utterance.word_intervals),
                selectinload(Utterance.other_intervals),
            )
            .filter(Utterance.file_id == self.file_id)
            .order_by(Utterance.start)
            .all()
        )
        return utterances, self.file_id


class LoadPhonesWorker(Worker):
    def __init__(self, session, use_mp=False, **kwargs):
        super().__init__(use_mp=use_mp, **kwargs)
        self.session = session

    def _run(self):
        begin = time.time()
        conn = self.session.bind.raw_connection()
        phones = []
        try:
            cursor = conn.cursor()
            cursor.execute(
                "select distinct phone_interval.phone from phone_interval order by phone_interval.phone"
            )
            query = cursor.fetchall()
            for p in query:
                phones.append(p)
            cursor.close()
        finally:
            conn.close()
        logger.debug(f"Loading all phones took {time.time() - begin:.3f} seconds.")
        return phones


class LoadWordsWorker(Worker):
    def __init__(self, session, use_mp=False, **kwargs):
        super().__init__(use_mp=use_mp, **kwargs)
        self.session = session

    def _run(self):
        begin = time.time()
        conn = self.session.bind.raw_connection()
        words = []
        try:
            cursor = conn.cursor()
            cursor.execute(
                "select distinct word_interval.word from word_interval order by word_interval.word"
            )
            query = cursor.fetchall()
            for w in query:
                words.append(w)
            cursor.close()
        finally:
            conn.close()
        logger.debug(f"Loading all words took {time.time() - begin:.3f} seconds.")
        return words


class FunctionWorker(QtCore.QThread):  # pragma: no cover
    def __init__(self, name, *args):
        super().__init__(*args)
        self.settings = SplaatSettings()
        self.signals = WorkerSignals(name)
        self.stopped = threading.Event()
        self.lock = Lock()

    def setParams(self, kwargs):
        self.kwargs = kwargs
        self.kwargs["progress_callback"] = self.signals.progress
        self.kwargs["stopped"] = self.stopped
        self.total = None

    def stop(self):
        self.stopped.set()


class AutoWaveformWorker(Worker):  # pragma: no cover
    def __init__(self, y, normalized_min, normalized_max, begin, end, channel, *args):
        super().__init__("Scaling waveform", *args)
        self.y = y
        self.normalized_min = normalized_min
        self.normalized_max = normalized_max
        self.begin = begin
        self.end = end
        self.channel = channel

    def run(self):
        self.stopped.clear()
        if self.y.shape[0] == 0:
            return
        max_val = np.max(np.abs(self.y), axis=0)
        if np.isnan(max_val):
            return
        normalized = self.y / max_val
        normalized[np.isnan(normalized)] = 0

        height = self.normalized_max - self.normalized_min

        new_height = height / 2
        mid_point = self.normalized_min + new_height
        normalized = normalized * 0.5 + mid_point
        if self.stopped.is_set():
            return
        self.signals.result.emit((normalized, self.begin, self.end, self.channel))


class WaveformWorker(Worker):  # pragma: no cover
    def __init__(self, file_path, *args):
        super().__init__("Loading waveform", *args)
        self.file_path = file_path

    def run(self):
        try:
            y, _ = soundfile.read(self.file_path)
        except soundfile.LibsndfileError:
            logger.warning(f"Could not read {self.file_path}")
            y = None
        self.signals.result.emit((y, self.file_path))


class AnnotationTierWorker(Worker):  # pragma: no cover
    def __init__(
        self,
        session,
        file_id,
        *args,
        utterance_id=None,
        start=None,
        end=None,
    ):
        super().__init__("Generating annotation tier", *args)
        self.session = session
        self.file_id = file_id
        self.utterance_id = utterance_id
        self.start = start
        self.end = end
        self.settings = SplaatSettings()

    def run(self):
        if self.session is None:
            return
        with self.session() as session:
            file = session.get(File, self.file_id)
            utterances = (
                self.session.query(Utterance)
                .options(
                    selectinload(Utterance.word_intervals),
                    selectinload(Utterance.phone_intervals),
                )
                .join(WordInterval.utterance)
                .filter(Utterance.file_id == self.file_id)
                .order_by(Utterance.start)
            )

            if self.utterance_id is not None:
                utterances = utterances.filter(WordInterval.id == self.word_interval_id)
            if file.duration > 500 and self.start is not None and self.end is not None:
                cached_begin = self.start - 30
                cached_end = self.end + 30
                utterances = utterances.filter(
                    WordInterval.end >= cached_begin,
                    WordInterval.start <= cached_end,
                )
            else:
                cached_begin = None
                cached_end = None
            utterances = utterances.all()
            if (
                file.duration > 500
                and self.start is not None
                and self.end is not None
                and utterances
            ):
                cached_begin = min(cached_begin, utterances[0].begin)
                cached_end = max(cached_end, utterances[-1].end)
            self.signals.result.emit((utterances, self.file_id, cached_begin, cached_end))


class SpectrogramWorker(Worker):  # pragma: no cover
    def __init__(self, y, sample_rate, begin, end, channel, *args):
        super().__init__("Generating spectrogram", *args)
        self.y = y
        self.sample_rate = sample_rate
        self.begin = begin
        self.end = end
        self.channel = channel

    def run(self):
        dynamic_range = self.settings.value(self.settings.SPEC_DYNAMIC_RANGE)
        n_fft = self.settings.value(self.settings.SPEC_N_FFT)
        time_steps = self.settings.value(self.settings.SPEC_N_TIME_STEPS)
        window_size = self.settings.value(self.settings.SPEC_WINDOW_SIZE)
        pre_emph_coeff = self.settings.value(self.settings.SPEC_PREEMPH)
        max_freq = self.settings.value(self.settings.SPEC_MAX_FREQ)
        if self.y.shape[0] == 0:
            self.signals.result.emit(None)
            return
        duration = self.y.shape[0] / self.sample_rate
        if duration > self.settings.value(self.settings.SPEC_MAX_TIME):
            self.signals.result.emit(None)
            return
        max_sr = 2 * max_freq
        if self.sample_rate > max_sr:
            self.y = scipy.signal.resample(
                self.y, int(self.y.shape[0] * max_sr / self.sample_rate)
            )
            self.sample_rate = max_sr
        self.y = librosa.effects.preemphasis(self.y, coef=pre_emph_coeff)
        begin_samp = int(self.begin * self.sample_rate)
        end_samp = int(self.end * self.sample_rate)
        window_size = round(window_size, 6)
        window_size_samp = int(window_size * self.sample_rate)
        duration_samp = end_samp - begin_samp
        if time_steps >= duration_samp:
            step_size_samples = 1
        else:
            step_size_samples = int(duration_samp / time_steps)
        stft = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(
                    self.y,
                    n_fft=n_fft,
                    win_length=window_size_samp,
                    hop_length=step_size_samples,
                    center=True,
                )
            ),
            top_db=dynamic_range,
        )
        min_db, max_db = np.min(stft), np.max(stft)
        self.signals.result.emit((stft, self.channel, self.begin, self.end, min_db, max_db))


class ImportCorpusWorker(FunctionWorker):  # pragma: no cover
    def __init__(self, *args):
        super().__init__("Importing corpus", *args)
        self.corpus_path: Path = None
        self.reset = None

    def stop(self):
        self.stopped.set()

    def set_params(self, corpus_path: Path, reset=False):
        self.corpus_path = corpus_path
        self.reset = reset

    def run(self):
        try:
            corpus_name = os.path.basename(self.corpus_path)
            corpus_temp_dir = self.settings.temp_directory.joinpath(corpus_name)
            corpus_temp_dir.mkdir(exist_ok=True, parents=True)
            db_file = corpus_temp_dir.joinpath(f"{corpus_name}.db")
            db_string = f"sqlite:///{db_file}"
            if not db_file.exists():
                db_engine = sqlalchemy.create_engine(db_string)
                SqlBase.metadata.create_all(db_engine)
                with sqlalchemy.orm.Session(db_engine) as session:
                    for f in self.corpus_path.rglob("*"):
                        if self.stopped.is_set():
                            break
                        if not f.is_file():
                            continue
                        parse_file_to_db(session, f, root_directory=self.corpus_path)
                    if not self.stopped.is_set():
                        session.commit()
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
            self.signals.result.emit(None)
        else:
            self.signals.result.emit(corpus_name)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
