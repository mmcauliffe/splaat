from __future__ import annotations

import logging
import re
import typing
from dataclasses import dataclass
from threading import Lock

import numpy as np
import sqlalchemy
from PySide6 import QtCore
from sqlalchemy.orm import joinedload

from splaat.db import File, PhoneInterval, Utterance, WordInterval
from splaat.desktop import undo, workers
from splaat.desktop.settings import SplaatSettings

logger = logging.getLogger("splaat")


@dataclass(slots=True)
class TextFilterQuery:
    text: str
    regex: bool = False
    word: bool = False
    case_sensitive: bool = False
    graphemes: typing.Collection[str] = None

    @property
    def search_text(self):
        if not self.case_sensitive:
            return self.text.lower()
        return self.text

    def generate_expression(self, posix=False):
        word_symbols = r"\w"
        if self.graphemes:
            dash_prefix = "-" if "-" in self.graphemes else ""
            graphemes = "".join([x for x in self.graphemes if x != "-"])
            word_symbols = rf"[{dash_prefix}\w{graphemes}]"
        text = self.text
        if not self.case_sensitive:
            text = text.lower()
        if not text:
            return text
        if not self.regex:
            text = re.escape(text)
        word_break_set = r"\b"
        if self.word:
            if not text.startswith(word_break_set):
                text = word_break_set + text
            if not text.endswith(word_break_set):
                text += word_break_set
        if posix:
            if text.startswith(r"\b"):
                text = rf"((?<!{word_symbols})|(?<=^))" + text[2:]
            if text.endswith(r"\b"):
                text = text[:-2] + rf"((?!{word_symbols})|(?=$))"
        if not self.case_sensitive:
            text = "(?i)" + text
        return text


class TableModel(QtCore.QAbstractTableModel):
    runFunction = QtCore.Signal(object, object, object)  # Function plus finished processor
    resultCountChanged = QtCore.Signal(int)
    newResults = QtCore.Signal()

    def __init__(self, header_data, parent=None):
        super().__init__(parent)
        self._header_data = header_data
        self._data = []
        self.result_count = None
        self.sort_index = None
        self.sort_order = None
        self.current_offset = 0
        self.limit = 1

    def set_limit(self, limit: int):
        self.limit = limit

    def set_offset(self, offset):
        self.current_offset = offset
        self.update_data()
        self.update_result_count()

    def update_sort(self, column, order):
        self.sort_index = column
        self.sort_order = order
        self.update_data()
        self.update_result_count()

    def query_count(self, **kwargs):
        pass

    def query_data(self, **kwargs):
        pass

    def finalize_result_count(self, result_count=None):
        if isinstance(result_count, int):
            self.result_count = result_count
        self.resultCountChanged.emit(self.result_count)

    def update_result_count(self):
        self.result_count = None
        self.runFunction.emit(self.query_count, self.finalize_result_count, [])

    def update_data(self):
        self.runFunction.emit(self.query_data, self.finish_update_data, [])

    def finish_update_data(self, *args, **kwargs):
        self.layoutAboutToBeChanged.emit()
        self._data = []
        self.layoutChanged.emit()

    def headerData(self, index, orientation, role=None, *args, **kwargs):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self._header_data[index]

    def data(self, index, role=None):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._header_data)


class FileModel(QtCore.QAbstractListModel):
    addCommand = QtCore.Signal(object)
    selectionRequested = QtCore.Signal(object)
    changeCommandFired = QtCore.Signal()
    refreshTiers = QtCore.Signal()

    waveformReady = QtCore.Signal()
    utterancesReady = QtCore.Signal()
    speakersChanged = QtCore.Signal()
    phoneTierChanged = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.utterances = []
        self.file = None
        self.y = None
        self.speakers = []
        self._indices = []
        self._speaker_indices = []
        self.reversed_indices = {}
        self.speaker_channel_mapping = {}
        self.corpus_model: typing.Optional[CorpusModel] = None
        self.closing = False
        self.cached_begin = None
        self.cached_end = None

        self.thread_pool = QtCore.QThreadPool()
        self.thread_pool.setMaxThreadCount(4)

    def get_utterance(self, utterance_id: int) -> Utterance:
        try:
            return self.utterances[self.reversed_indices[utterance_id]]
        except KeyError:
            return None

    def set_corpus_model(self, corpus_model: CorpusModel):
        self.corpus_model = corpus_model

    def clean_up_for_close(self):
        self.closing = True

    def set_file(self, file_id, utterance_id=None, begin=None, end=None):
        self.file = (
            self.corpus_model.session.query(File).options(joinedload(File.sound_file)).get(file_id)
        )
        self.y = None
        self.get_utterances(utterance_id, begin, end)
        waveform_worker = workers.WaveformWorker(self.file.sound_file.sound_file_path)
        waveform_worker.signals.result.connect(self.finalize_loading_wave_form)
        self.thread_pool.start(waveform_worker)

    def finalize_loading_utterances(self, results):
        if self.closing:
            return
        utterances, file_id, self.cached_begin, self.cached_end = results
        if file_id != self.file.id:
            return
        self.utterances = utterances
        for i, u in enumerate(utterances):
            self.reversed_indices[u.id] = i
            self._indices.append(u.id)
        self.utterancesReady.emit()

    def finalize_loading_wave_form(self, results):
        if self.closing:
            return
        y, file_path = results
        if self.file is None or file_path != self.file.sound_file.sound_file_path:
            return
        self.y = y
        self.waveformReady.emit()

    def get_utterances(self, utterance_id=None, start=None, end=None):
        parent_index = self.index(0, 0)
        self.beginRemoveRows(parent_index, 0, len(self.utterances))
        self.utterances = []
        self.speakers = []
        self._indices = []
        self._speaker_indices = []
        self.speaker_channel_mapping = {}
        self.reversed_indices = {}
        self.endRemoveRows()
        if self.file is None:
            return
        speaker_tier_worker = workers.AnnotationTierWorker(
            self.corpus_model.session,
            self.file.id,
            utterance_id=utterance_id,
            start=start,
            end=end,
        )
        speaker_tier_worker.signals.result.connect(self.finalize_loading_utterances)
        self.thread_pool.start(speaker_tier_worker)

    def update_phone_boundaries(
        self,
        utterance: Utterance,
        first_phone_interval: PhoneInterval,
        second_phone_interval: PhoneInterval,
        new_time: float,
    ):
        if first_phone_interval.end == new_time and second_phone_interval.start == new_time:
            return
        self.addCommand.emit(
            undo.UpdatePhoneBoundariesCommand(
                utterance, first_phone_interval, second_phone_interval, new_time, self
            )
        )
        self.corpus_model.set_file_modified(self.file.id)

    def update_phone_interval(
        self, utterance: Utterance, phone_interval: PhoneInterval, phone: str
    ):
        if phone_interval.phone == phone:
            return
        self.addCommand.emit(
            undo.UpdatePhoneIntervalCommand(utterance, phone_interval, phone, self)
        )
        self.corpus_model.set_file_modified(self.file.id)

    def insert_phone_interval(
        self,
        utterance: Utterance,
        phone_interval,
        previous_interval: PhoneInterval,
        following_interval: PhoneInterval,
        word_interval: WordInterval,
    ):
        self.addCommand.emit(
            undo.InsertPhoneIntervalCommand(
                utterance,
                phone_interval,
                previous_interval,
                following_interval,
                self,
                word_interval,
            )
        )
        self.corpus_model.set_file_modified(self.file.id)

    def delete_phone_interval(
        self,
        utterance: Utterance,
        phone_interval: PhoneInterval,
        previous_interval: PhoneInterval,
        following_interval: PhoneInterval,
        time_point: float,
    ):
        self.addCommand.emit(
            undo.DeletePhoneIntervalCommand(
                utterance, phone_interval, previous_interval, following_interval, time_point, self
            )
        )
        self.corpus_model.set_file_modified(self.file.id)

    def refresh_utterances(self):
        for utterance in self.utterances:
            self.corpus_model.session.refresh(utterance)

    def rowCount(self, parent=None):
        return len(self.utterances)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.utterances[index.row()]


class FileSelectionModel(QtCore.QItemSelectionModel):
    fileAboutToChange = QtCore.Signal()
    fileChanged = QtCore.Signal()
    channelChanged = QtCore.Signal()
    resetView = QtCore.Signal()
    viewChanged = QtCore.Signal(object, object)
    selectionAudioChanged = QtCore.Signal(object)
    currentUtteranceChanged = QtCore.Signal(object)
    speakerRequested = QtCore.Signal(object)
    searchTermChanged = QtCore.Signal(object)

    spectrogramReady = QtCore.Signal()
    waveformReady = QtCore.Signal()
    pitchTrackReady = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = SplaatSettings()
        self.min_time = 0
        self.max_time = 10
        self.selected_min_time = None
        self.selected_max_time = None
        self.x = None
        self.y = None
        self.top_point = 2
        self.bottom_point = 0
        self.separator_point = 1
        self.selected_channel = 0
        self.spectrogram = None
        self.min_db = None
        self.max_db = None
        self.pitch_track_x = None
        self.pitch_track_y = None
        self.waveform_x = None
        self.waveform_y = None
        self.requested_word_interval_id = None
        self.closing = False

        self.thread_pool = QtCore.QThreadPool()
        self.thread_pool.setMaxThreadCount(self.settings.value(self.settings.PLOT_THREAD_COUNT))
        self.model().waveformReady.connect(self.load_audio_selection)
        self.model().utterancesReady.connect(self.finalize_set_new_file)
        self.viewChanged.connect(self.load_audio_selection)
        self.view_change_timer = QtCore.QTimer()
        self.view_change_timer.setSingleShot(True)
        self.view_change_timer.setInterval(10)
        self.view_change_timer.timeout.connect(self.send_selection_update)

    def selected_utterances(self):
        utterances = []
        m = self.model()
        for index in self.selectedRows(0):
            u = m.utterances[index.row()]
            utterances.append(u)
        return utterances

    def load_audio_selection(self):
        if self.model().y is None:
            return
        begin_samp = int(self.min_time * self.model().file.sample_rate)
        end_samp = int(self.max_time * self.model().file.sample_rate)
        if self.model().cached_begin is not None and (
            self.min_time < self.model().cached_begin + 5
            or self.max_time > self.model().cached_end - 5
        ):
            self.model().get_utterances(start=self.min_time, end=self.max_time)
        if len(self.model().y.shape) > 1:
            y = self.model().y[begin_samp:end_samp, self.selected_channel]
        else:
            y = self.model().y[begin_samp:end_samp]
        spectrogram_worker = workers.SpectrogramWorker(
            y,
            self.model().file.sound_file.sample_rate,
            self.min_time,
            self.max_time,
            self.selected_channel,
        )
        spectrogram_worker.signals.result.connect(self.finalize_loading_spectrogram)
        self.thread_pool.start(spectrogram_worker)

        auto_waveform_worker = workers.AutoWaveformWorker(
            y,
            self.separator_point,
            self.top_point,
            self.min_time,
            self.max_time,
            self.selected_channel,
        )
        auto_waveform_worker.signals.result.connect(self.finalize_loading_auto_wave_form)
        self.thread_pool.start(auto_waveform_worker)

    def clean_up_for_close(self):
        self.closing = True

    @property
    def plot_min(self):
        if self.settings.right_to_left:
            return -self.max_time
        return self.min_time

    @property
    def plot_max(self):
        if self.settings.right_to_left:
            return -self.min_time
        return self.max_time

    def finalize_loading_spectrogram(self, results):
        if self.closing:
            return
        if results is None:
            self.spectrogram = None
            self.min_db = None
            self.max_db = None
            self.spectrogramReady.emit()
            return
        stft, channel, begin, end, min_db, max_db = results
        if begin != self.min_time or end != self.max_time:
            return
        if self.settings.right_to_left:
            stft = np.flip(stft, 1)
        self.spectrogram = stft
        self.min_db = self.min_db
        self.max_db = self.max_db
        self.spectrogramReady.emit()

    def finalize_loading_pitch_track(self, results):
        if self.closing:
            return
        if results is None:
            self.pitch_track_y = None
            self.pitch_track_x = None
            self.pitchTrackReady.emit()
            return
        pitch_track, voicing_track, channel, begin, end, min_f0, max_f0 = results
        if begin != self.min_time or end != self.max_time:
            return
        if self.settings.right_to_left:
            pitch_track = np.flip(pitch_track, 0)
        self.pitch_track_y = pitch_track
        if pitch_track is None:
            return
        x = np.linspace(
            start=self.plot_min,
            stop=self.plot_max,
            num=pitch_track.shape[0],
        )
        self.pitch_track_x = x
        self.pitchTrackReady.emit()

    def finalize_loading_auto_wave_form(self, results):
        if self.closing:
            return
        y, begin, end, channel = results
        if begin != self.min_time or end != self.max_time:
            return
        if self.settings.right_to_left:
            y = np.flip(y, 0)
        x = np.linspace(start=self.plot_min, stop=self.plot_max, num=y.shape[0])
        self.waveform_x = x
        self.waveform_y = y
        self.waveformReady.emit()

    def select_audio(self, begin, end):
        if end is not None and end - begin < 0.025:
            end = None
        self.selected_min_time = begin
        self.selected_max_time = end
        if self.selected_min_time != self.min_time or end is not None:
            self.selectionAudioChanged.emit(False)

    def request_start_time(self, start_time, update=False):
        if start_time >= self.max_time:
            return
        if start_time < self.min_time:
            return
        self.selected_min_time = start_time
        self.selected_max_time = None
        self.selectionAudioChanged.emit(update)

    def set_current_channel(self, channel):
        if channel == self.selected_channel:
            return
        self.selected_channel = channel
        self.load_audio_selection()

    def get_selected_wave_form(self):
        if self.y is None:
            return None, None
        if len(self.y.shape) > 1 and self.y.shape[0] == 2:
            return self.x, self.y[self.selected_channel, :]
        return self.x, self.y

    def zoom(self, factor, mid_point=None):
        if factor == 0 or self.min_time is None:
            return
        cur_duration = self.max_time - self.min_time
        if mid_point is None:
            mid_point = self.min_time + (cur_duration / 2)
        new_duration = cur_duration / factor
        new_start = mid_point - (mid_point - self.min_time) / factor
        new_start = max(new_start, 0)
        new_end = min(new_start + new_duration, self.model().file.duration)
        if new_end - new_start <= 0.025:
            return
        self.set_view_times(new_start, new_end)

    def pan(self, factor):
        if self.min_time is None:
            return
        if factor < 1:
            factor = 1 - factor
            right = True
        else:
            right = False
            factor = factor - 1
        if right and self.max_time == self.model().file.duration:
            return
        if not right and self.min_time == 0:
            return
        cur_duration = self.max_time - self.min_time
        shift = factor * cur_duration
        if right:
            new_start = self.min_time + shift
            new_end = self.max_time + shift
        else:
            new_start = self.min_time - shift
            new_end = self.max_time - shift
        if new_start < 0:
            new_end = new_end + abs(new_start)
            new_start = 0
        if new_end > self.model().file.duration:
            new_start -= self.model().file.duration - new_end
            new_end = self.model().file.duration
        self.set_view_times(new_start, new_end)

    def zoom_in(self):
        if self.model().file is None:
            return
        self.zoom(1.5)

    def zoom_out(self):
        if self.model().file is None:
            return
        self.zoom(0.5)

    def zoom_to_selection(self):
        if self.selected_min_time is not None and self.selected_max_time is not None:
            start = self.selected_min_time
            end = self.selected_max_time
        elif len(self.selectedRows(0)) > 0:
            m = self.model()
            start = 100000
            end = 0
            for index in self.selectedRows(0):
                u = m.utterances[index.row()]
                if u.start < start:
                    start = u.start
                if u.end > end:
                    end = u.end
        else:
            return
        self.set_view_times(start, end)

    @property
    def step_size(self):
        return (self.max_time - self.min_time) * 0.5

    def pan_right(self):
        new_begin = self.min_time + self.step_size
        self.update_from_slider(new_begin)

    def pan_left(self):
        new_begin = self.min_time - self.step_size
        self.update_from_slider(new_begin)

    def zoom_all(self):
        self.set_view_times(0, self.model().file.duration)

    def update_from_slider(self, value):
        if not self.max_time:
            return
        if value < 0:
            value = 0
        window_size = self.max_time - self.min_time
        if value + window_size > self.model().file.duration:
            value = self.model().file.duration - window_size
        self.set_view_times(value, value + window_size)

    def visible_utterances(self) -> typing.List[Utterance]:
        file_utterances = []
        if not self.model().file:
            return file_utterances
        if self.model().rowCount() > 1:
            for u in self.model().utterances:
                if u.start >= self.max_time:
                    break
                if u.end <= self.min_time:
                    continue
                file_utterances.append(u)
        else:
            file_utterances.extend(self.model().utterances)
        return file_utterances

    def model(self) -> FileModel:
        return super().model()

    def set_view_times(self, start, end):
        start = max(start, 0)
        end = min(end, self.model().file.duration)
        if (start, end) == (self.min_time, self.max_time):
            return
        self.min_time = start
        self.max_time = end
        if (
            self.selected_max_time is not None
            and not self.min_time <= self.selected_min_time <= self.max_time
        ):
            self.selected_min_time = self.min_time
        if (
            self.selected_max_time is not None
            and not self.min_time <= self.selected_max_time <= self.max_time
        ):
            self.selected_max_time = None
        self.view_change_timer.start()

    def send_selection_update(self):
        self.viewChanged.emit(self.min_time, self.max_time)

    def set_current_file(
        self,
        file_id,
        begin,
        end,
    ):
        self.selected_min_time = None
        self.selected_max_time = None
        self.fileAboutToChange.emit()
        self.model().set_file(file_id, begin=begin, end=end)
        self.fileChanged.emit()
        self.set_view_times(begin, end)

    def finalize_set_new_file(self):
        self.fileChanged.emit()

    def checkSelected(self, utterance_id: int):
        m = self.model()
        for index in self.selectedRows(0):
            if utterance_id == m._indices[index.row()]:
                return True
        return False


class CorpusModel(TableModel):
    corpusLoaded = QtCore.Signal()
    corpusLoading = QtCore.Signal()
    textFilterChanged = QtCore.Signal()
    phoneFilterChanged = QtCore.Signal()

    def __init__(self, parent=None):
        header = [
            "Directory",
            "File",
            "Duration",
            "Text",
        ]
        super().__init__(header, parent=parent)
        self.settings = SplaatSettings()
        self.file_column = header.index("File")
        self.duration_column = header.index("Duration")
        self.text_column = header.index("Text")
        self.default_header_sizes = ["00000000", "0000000000000000", "000000", "00000000"]
        self.sort_index = None
        self.sort_order = None
        self.text_filter = None
        self.phone_filter = None
        self.reversed_indices = {}
        self._indices = []
        self._file_indices = []
        self._speaker_indices = []
        self._data = []
        self.corpus_name = None
        self._db_engine = None
        self.files = None
        self.session: sqlalchemy.orm.scoped_session = None
        self.file_count = 0
        self.phones = []
        self.words = []
        self.edit_lock = Lock()

    def export_files(self, export_directory):
        modified_files = self.session.query(File).filter(File.modified == True)  # noqa
        for f in modified_files:
            f.save(export_directory)

    def set_file_modified(self, file_id: typing.Union[int, typing.List[int]]):
        if isinstance(file_id, int):
            file_id = [file_id]
        data = {File.modified: True}
        self.session.query(File).filter(File.id.in_(file_id)).update(data)
        self.session.commit()

    def update_sort(self, column, order):
        self.sort_index = column
        self.sort_order = order
        self.update_data()

    def utterance_id_at(self, index) -> typing.Optional[int]:
        if not isinstance(index, int):
            if not index.isValid():
                return None
            index = index.row()
        if index > len(self._indices) - 1:
            return None
        if len(self._indices) == 0:
            return None
        return self._indices[index]

    def file_id_at(self, index) -> typing.Optional[int]:
        if not isinstance(index, int):
            if not index.isValid():
                return None
            index = index.row()
        if index > len(self._file_indices) - 1:
            return None
        if len(self._file_indices) == 0:
            return None
        return self._file_indices[index]

    def file_name_at(self, index) -> typing.Optional[str]:
        if not isinstance(index, int):
            if not index.isValid():
                return None
            index = index.row()
        if index > len(self._indices) - 1:
            return None
        if len(self._indices) == 0:
            return None
        return f"{self._data[index][0]}/{self._data[index][1]}"

    @property
    def db_string(self):
        db_path = self.settings.temp_directory.joinpath(self.corpus_name, f"{self.corpus_name}.db")
        return f"sqlite:///{db_path}"

    @property
    def db_engine(self):
        if self._db_engine is None:
            self._db_engine = sqlalchemy.create_engine(self.db_string)
        return self._db_engine

    def setCorpus(self, corpus_name):
        self.corpus_name = corpus_name
        if self.corpus_name is not None:
            self.session = sqlalchemy.orm.scoped_session(
                sqlalchemy.orm.sessionmaker(bind=self.db_engine, expire_on_commit=False)
            )
            self.corpusLoading.emit()
            self.refresh_files()
            self.refresh_phones()
            self.refresh_words()

    def refresh_files(self):
        self.update_data()
        self.update_result_count()

    def data(self, index, role):
        if not index.isValid():
            return None
        try:
            data = self._data[index.row()][index.column()]
        except IndexError:
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole and index.column() == 0:
            if data:
                return str(data)
        elif role == QtCore.Qt.ItemDataRole.DisplayRole:
            return data

    def clear_data(self):
        self.layoutAboutToBeChanged.emit()
        self.reversed_indices = {}
        self._indices = []
        self._file_indices = []
        self._data = []
        self.files = []
        self.layoutChanged.emit()

    def finish_update_data(self, result, *args, **kwargs):
        if not result:
            return
        self.layoutAboutToBeChanged.emit()
        (
            self._data,
            self._indices,
            self._file_indices,
            self.reversed_indices,
        ) = result
        self.layoutChanged.emit()
        self.newResults.emit()

    @property
    def count_kwargs(self) -> typing.Dict[str, typing.Any]:
        kwargs = self.query_kwargs
        kwargs["count"] = True
        return kwargs

    @property
    def query_kwargs(self) -> typing.Dict[str, typing.Any]:
        kwargs = {
            "text_filter": self.text_filter,
            "phone_filter": self.phone_filter,
            "limit": self.limit,
            "current_offset": self.current_offset,
        }
        if self.sort_index is not None:
            kwargs["sort_index"] = self.sort_index
            kwargs["sort_desc"] = self.sort_order == QtCore.Qt.SortOrder.DescendingOrder
        return kwargs

    def search_text(
        self,
        text_filter: TextFilterQuery,
    ):
        self.text_filter = text_filter
        self.textFilterChanged.emit()
        self.refresh_files()

    def search_phones(
        self,
        phone_filter: TextFilterQuery,
    ):
        self.phone_filter = phone_filter
        self.phoneFilterChanged.emit()
        self.refresh_files()

    def finalize_result_count(self, result_count):
        if not isinstance(result_count, int):
            return
        self.result_count = result_count
        self.resultCountChanged.emit(self.result_count)

    def update_data(self):
        self.runFunction.emit("Querying utterances", self.finish_update_data, [self.query_kwargs])

    def update_result_count(self):
        self.runFunction.emit(
            "Counting utterance results", self.finalize_result_count, [self.count_kwargs]
        )

    def refresh_phones(self):
        self.runFunction.emit("Loading phones", self.finish_update_phones, [])

    def refresh_words(self):
        self.runFunction.emit("Loading words", self.finish_update_words, [])

    @property
    def fully_loaded(self):
        if not self.phones:
            return False
        if not self.words:
            return False
        return True

    def finish_update_phones(self, phones):
        self.phones = phones
        if self.fully_loaded:
            self.corpusLoaded.emit()

    def finish_update_words(self, words):
        self.words = words
        if self.fully_loaded:
            self.corpusLoaded.emit()
