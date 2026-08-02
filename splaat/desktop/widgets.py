from __future__ import annotations

import logging
import re
import time
import typing

from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets

import splaat.desktop.resources_rc  # noqa
from splaat.desktop.models import TextFilterQuery
from splaat.desktop.plot import SplaatPlot
from splaat.desktop.settings import SplaatSettings

logger = logging.getLogger("splaat")

if typing.TYPE_CHECKING:
    from splaat.desktop.models import CorpusModel, FileModel, FileSelectionModel
    from splaat.desktop.workers import Worker


class MediaPlayer(QtMultimedia.QMediaPlayer):  # pragma: no cover
    timeChanged = QtCore.Signal(object)
    audioReady = QtCore.Signal(object)

    def __init__(self, *args):
        super(MediaPlayer, self).__init__(*args)
        self.settings = SplaatSettings()
        self.devices = QtMultimedia.QMediaDevices()
        self.devices.audioOutputsChanged.connect(self.update_audio_device)
        self.max_time = None
        self.start_load_time = None
        self.min_time = None
        self.selection_model = None
        self.positionChanged.connect(self.checkStop)
        # self.positionChanged.connect(self.positionDebug)
        self.errorOccurred.connect(self.handle_error)

        self._audio_output = QtMultimedia.QAudioOutput(self.devices.defaultAudioOutput())
        self.setAudioOutput(self._audio_output)
        self.playbackStateChanged.connect(self.reset_position)
        self.fade_in_anim = QtCore.QPropertyAnimation(self._audio_output, b"volume")
        self.fade_in_anim.setDuration(10)
        self.fade_in_anim.setStartValue(0.1)
        self.fade_in_anim.setEndValue(self._audio_output.volume())
        self.fade_in_anim.setEasingCurve(QtCore.QEasingCurve.Type.Linear)
        self.fade_in_anim.setKeyValueAt(0.1, 0.1)

        self.file_path = None
        self.set_volume(self.settings.value(self.settings.VOLUME))

    def setMuted(self, muted: bool):
        self.audioOutput().setMuted(muted)

    def handle_error(self, *args):
        logger.info("ERROR")
        logger.info(args)

    def play(self) -> None:
        if self.startTime() is None:
            return
        if self.mediaStatus() not in {
            QtMultimedia.QMediaPlayer.MediaStatus.BufferedMedia,
            QtMultimedia.QMediaPlayer.MediaStatus.LoadedMedia,
            QtMultimedia.QMediaPlayer.MediaStatus.EndOfMedia,
        }:
            return
        fade_in = self.settings.value(self.settings.ENABLE_FADE)
        if fade_in:
            self._audio_output.setVolume(0.1)
        if (
            self.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.StoppedState
            or self.currentTime() < self.startTime()
            or self.currentTime() >= self.maxTime()
        ):
            self.setCurrentTime(self.startTime())
        super(MediaPlayer, self).play()
        if fade_in:
            self.fade_in_anim.start()

    def startTime(self):
        if (
            self.selection_model.selected_min_time is not None
            and self.selection_model.min_time
            <= self.selection_model.selected_min_time
            <= self.selection_model.max_time
        ):
            return self.selection_model.selected_min_time
        return self.selection_model.min_time

    def maxTime(self):
        if (
            self.selection_model.selected_max_time is not None
            and self.selection_model.min_time
            <= self.selection_model.selected_max_time
            <= self.selection_model.max_time
        ):
            return self.selection_model.selected_max_time
        return self.selection_model.max_time

    def reset_position(self):
        state = self.playbackState()
        if state == QtMultimedia.QMediaPlayer.PlaybackState.StoppedState:
            self.setCurrentTime(self.startTime())

    def update_audio_device(self):
        self._audio_output.setDevice(self.devices.defaultAudioOutput())
        self.setAudioOutput(self._audio_output)

    def refresh_settings(self):
        self.settings.sync()
        self.update_audio_device()

    def set_models(self, selection_model: typing.Optional[FileSelectionModel]):
        if selection_model is None:
            return
        self.selection_model = selection_model
        self.selection_model.fileChanged.connect(self.load_new_file)
        self.selection_model.viewChanged.connect(self.update_times)
        self.selection_model.selectionAudioChanged.connect(self.update_selection_times)

    def set_volume(self, volume: int):
        self.settings.setValue(self.settings.VOLUME, volume)
        if self.audioOutput() is None:
            return
        linearVolume = QtMultimedia.QAudio.convertVolume(
            volume / 100.0,
            QtMultimedia.QAudio.VolumeScale.LogarithmicVolumeScale,
            QtMultimedia.QAudio.VolumeScale.LinearVolumeScale,
        )
        self.audioOutput().setVolume(linearVolume)
        self.fade_in_anim.setEndValue(linearVolume)

    def volume(self) -> int:
        if self.audioOutput() is None:
            return 100
        volume = self.audioOutput().volume()
        volume = int(
            QtMultimedia.QAudio.convertVolume(
                volume,
                QtMultimedia.QAudio.VolumeScale.LinearVolumeScale,
                QtMultimedia.QAudio.VolumeScale.LogarithmicVolumeScale,
            )
            * 100
        )
        return volume

    def update_selection_times(self, update=False):
        if update or self.playbackState() != QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self.setCurrentTime(self.startTime())

    def update_times(self):
        if self.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            return
        if self.currentTime() < self.startTime() or self.currentTime() > self.maxTime():
            self.stop()
        if self.playbackState() != QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self.stop()
            self.setCurrentTime(self.startTime())

    def load_new_file(self, *args):
        if self.playbackState() in {
            QtMultimedia.QMediaPlayer.PlaybackState.PlayingState,
            QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
        }:
            self.stop()
            time.sleep(0.1)
        try:
            new_file = self.selection_model.model().file.sound_file.sound_file_path
        except Exception:
            self.setSource(QtCore.QUrl())
            return
        if (
            self.selection_model.max_time is None
            or self.selection_model.model().file is None
            or self.selection_model.model().file.duration is None
        ):
            self.setSource(QtCore.QUrl())
            return
        self.setSource(f"file:///{new_file}")

    def currentTime(self):
        pos = self.position()
        return pos / 1000

    def setCurrentTime(self, time):
        if time is None:
            time = 0
        pos = int(time * 1000)
        self.setPosition(pos)

    def checkStop(self):
        self.timeChanged.emit(self.currentTime())
        if self.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            if self.maxTime() is None or self.currentTime() > self.maxTime():
                self.stop()


class ErrorButtonBox(QtWidgets.QDialogButtonBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.report_bug_button = QtWidgets.QPushButton("Report bug")
        self.report_bug_button.setIcon(QtGui.QIcon.fromTheme("folder-open"))
        self.addButton(self.report_bug_button, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)


class FontDialog(QtWidgets.QFontDialog):
    def __init__(self, *args):
        super(FontDialog, self).__init__(*args)


class FontEdit(QtWidgets.QPushButton):  # pragma: no cover
    def __init__(self, parent=None):
        super(FontEdit, self).__init__(parent=parent)
        self.font = None
        self.clicked.connect(self.open_dialog)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def set_font(self, font: QtGui.QFont):
        self.font = font
        self.update_icon()

    def update_icon(self):
        self.setFont(self.font)
        self.setText(self.font.key().split(",", maxsplit=1)[0])

    def open_dialog(self):
        ok, font = FontDialog.getFont(self.font, self)
        if ok:
            self.font = font
            self.update_icon()


class HeaderView(QtWidgets.QHeaderView):
    def __init__(self, *args):
        super(HeaderView, self).__init__(*args)
        self.settings = SplaatSettings()
        self.setHighlightSections(False)
        self.setStretchLastSection(True)
        self.setSortIndicatorShown(True)
        self.setSectionsClickable(True)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.generate_context_menu)

    def sectionSizeFromContents(self, logicalIndex: int) -> QtCore.QSize:
        size = super().sectionSizeFromContents(logicalIndex)
        size.setWidth(
            size.width() + self.settings.text_padding + 3 + self.settings.sort_indicator_padding
        )
        return size

    def showHideColumn(self):
        index = self.model()._header_data.index(self.sender().text())
        self.setSectionHidden(index, not self.isSectionHidden(index))

    def generate_context_menu(self, location):
        menu = QtWidgets.QMenu()
        menu.addSeparator()
        m: CorpusModel = self.model()
        for i in range(m.columnCount()):
            column_name = m.headerData(
                i,
                orientation=QtCore.Qt.Orientation.Horizontal,
                role=QtCore.Qt.ItemDataRole.DisplayRole,
            )
            a = QtGui.QAction(column_name, self)

            a.setCheckable(True)
            if not self.isSectionHidden(i):
                a.setChecked(True)
            a.triggered.connect(self.showHideColumn)
            menu.addAction(a)
        # menu.setStyleSheet(self.settings.menu_style_sheet)
        menu.exec_(self.mapToGlobal(location))


class BaseTableView(QtWidgets.QTableView):
    def __init__(self, *args):
        self.settings = SplaatSettings()
        super().__init__(*args)
        self.setCornerButtonEnabled(False)
        # self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setHighlightSections(False)
        self.verticalHeader().setSectionsClickable(False)

        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setDragEnabled(False)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        self.header = HeaderView(QtCore.Qt.Orientation.Horizontal, self)
        self.setHorizontalHeader(self.header)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        copy_combo = QtCore.QKeyCombination(QtCore.Qt.Modifier.CTRL, QtCore.Qt.Key.Key_C)
        if event.keyCombination() == copy_combo:
            clipboard = QtGui.QGuiApplication.clipboard()
            current = self.selectionModel().currentIndex()
            text = self.selectionModel().model().data(current, QtCore.Qt.ItemDataRole.DisplayRole)
            clipboard.setText(str(text))
        elif QtGui.QKeySequence(event.keyCombination()) not in self.settings.all_keybinds:
            super().keyPressEvent(event)

    def setModel(self, model: QtCore.QAbstractItemModel) -> None:
        super().setModel(model)
        self.refresh_settings()

    def refresh_settings(self):
        self.settings.sync()
        # self.horizontalHeader().setFont(self.settings.big_font)
        # self.setFont(self.settings.font)


class SplaatTableView(BaseTableView):
    def setModel(self, model: QtCore.QAbstractItemModel) -> None:
        super().setModel(model)
        # self.model().newResults.connect(self.scrollToTop)
        self.selectionModel().clear()
        self.horizontalHeader().sortIndicatorChanged.connect(self.model().update_sort)

    def refresh_settings(self):
        super().refresh_settings()
        fm = QtGui.QFontMetrics(self.settings.big_font)
        minimum = 100
        default_sizes = getattr(
            self.model(),
            "default_header_sizes",
            [0 for i in range(self.horizontalHeader().count())],
        )
        for i in range(self.horizontalHeader().count()):
            text = self.model().headerData(
                i, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole
            )
            default_size = default_sizes[i]
            if len(default_size):
                text = default_size
            width = fm.boundingRect(text).width() + (3 * self.settings.sort_indicator_padding)
            if width < minimum:
                minimum = width
            self.setColumnWidth(i, width)
        self.horizontalHeader().setMinimumSectionSize(minimum)


class FileListTable(SplaatTableView):
    viewEditRequested = QtCore.Signal(int)

    def __init__(self, *args):
        super().__init__(*args)
        self.doubleClicked.connect(self.view_file)
        self.view_file_action = QtGui.QAction("View and Edit", self)
        self.view_file_action.triggered.connect(self.view_file)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.generate_context_menu)

    def generate_context_menu(self, location):
        menu = QtWidgets.QMenu()
        menu.addAction(self.view_file_action)
        menu.exec_(self.mapToGlobal(location))

    def view_file(self):
        rows = self.selectionModel().selectedRows()
        if rows:
            self.viewEditRequested.emit(rows[0])


class PaginationWidget(QtWidgets.QToolBar):
    offsetRequested = QtCore.Signal(int)
    pageRequested = QtCore.Signal()

    def __init__(self, *args):
        super(PaginationWidget, self).__init__(*args)
        w = QtWidgets.QWidget(self)
        w.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        w2 = QtWidgets.QWidget(self)
        w2.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.current_page = 0
        self.limit = 1
        self.num_pages = 1
        self.result_count = 0
        self.next_page_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("media-seek-forward"), text="Next page"
        )
        self.previous_page_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("media-seek-backward"), text="Previous page"
        )
        self.addWidget(w)
        self.page_label = QtWidgets.QLabel("Page 1 of 1")
        self.addAction(self.previous_page_action)
        self.addWidget(self.page_label)
        self.addAction(self.next_page_action)
        self.addWidget(w2)
        self.next_page_action.triggered.connect(self.next_page)
        self.previous_page_action.triggered.connect(self.previous_page)

    def reset(self):
        self.current_page = 0
        self.num_pages = 0

    def first_page(self):
        self.current_page = 0
        self.offsetRequested.emit(self.current_page * self.limit)

    def next_page(self):
        if self.current_page != self.num_pages - 1:
            self.current_page += 1
            self.offsetRequested.emit(self.current_page * self.limit)
            self.refresh_pages()

    def previous_page(self):
        if self.current_page != 0:
            self.current_page -= 1
            self.offsetRequested.emit(self.current_page * self.limit)
            self.refresh_pages()

    def set_limit(self, limit: int):
        self.limit = limit
        self._recalculate_num_pages()

    def _recalculate_num_pages(self):
        if self.result_count == 0:
            return
        self.num_pages = int(self.result_count / self.limit)
        if self.result_count % self.limit != 0:
            self.num_pages += 1
        self.refresh_pages()

    def update_result_count(self, result_count: int):
        self.result_count = result_count
        self._recalculate_num_pages()
        self.current_page = min(self.current_page, self.num_pages)

    def refresh_pages(self):
        self.previous_page_action.setEnabled(True)
        self.next_page_action.setEnabled(True)
        if self.current_page == 0:
            self.previous_page_action.setEnabled(False)
        if self.current_page == self.num_pages - 1 and self.num_pages > 0:
            self.next_page_action.setEnabled(False)
        self.page_label.setText(f"Page {self.current_page + 1} of {self.num_pages}")
        self.pageRequested.emit()


class InternalToolButtonEdit(QtWidgets.QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)

        self.tool_bar = QtWidgets.QToolBar(self)
        self._internal_layout.insertWidget(self._internal_layout.count(), self.tool_bar)
        self.tool_bar.setFocusProxy(self)

    @property
    def _internal_layout(self):
        if not hasattr(self, "_internal_layout_"):
            self._internal_layout_ = QtWidgets.QHBoxLayout(self)
            self._internal_layout_.addStretch()
        self._internal_layout_.setContentsMargins(1, 1, 1, 1)
        self._internal_layout_.setSpacing(0)
        return self._internal_layout_

    def setError(self):
        if not self.property("error"):
            self.setProperty("error", True)
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()

    def resetError(self):
        if self.property("error"):
            self.setProperty("error", False)
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()

    def _fix_cursor_position(self):
        self.setTextMargins(0, 0, self.tool_bar.geometry().width(), 0)

    def add_internal_action(self, action, name=None):
        self.tool_bar.addAction(action)
        w = self.tool_bar.widgetForAction(action)
        if name is not None:
            w.setObjectName(name)
        w.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        w.setFocusProxy(self)
        self._fix_cursor_position()


class ClearableField(InternalToolButtonEdit):
    def __init__(self, *args):
        super().__init__(*args)

        clear_icon = QtGui.QIcon.fromTheme("edit-clear")
        self.clear_action = QtGui.QAction(icon=clear_icon, parent=self)
        self.clear_action.triggered.connect(self.clear)
        self.clear_action.setVisible(False)
        self.textChanged.connect(self.check_contents)
        self.add_internal_action(self.clear_action, "clear_field")

    def clear(self) -> None:
        super().clear()
        self.returnPressed.emit()

    def add_button(self, button):
        self._internal_layout.insertWidget(self._internal_layout.count(), button)
        button.setFocusProxy(self)

    def check_contents(self):
        self.resetError()
        if super().text():
            self.clear_action.setVisible(True)
        else:
            self.clear_action.setVisible(False)


class ReplaceBox(ClearableField):
    replaceAllActivated = QtCore.Signal(object)

    def __init__(self, *args):
        super().__init__(*args)
        self.returnPressed.connect(self.activate)
        self.setObjectName("replace_box")

    def lock(self):
        self.setDisabled(True)

    def unlock(self):
        self.setEnabled(True)

    def activate(self):
        if not self.isEnabled():
            return
        self.replaceAllActivated.emit(self.text())


class SearchBox(ClearableField):
    searchActivated = QtCore.Signal(object)
    validationError = QtCore.Signal(object)

    def __init__(self, *args):
        super().__init__(*args)
        self.returnPressed.connect(self.activate)
        self.setObjectName("search_box")

        self.clear_action.triggered.connect(self.returnPressed.emit)

        regex_icon = QtGui.QIcon()
        word_icon = QtGui.QIcon()
        case_icon = QtGui.QIcon()
        if QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark:
            regex_icon.addFile(
                ":icons/dark/actions/edit-regex.svg",
                mode=QtGui.QIcon.Mode.Normal,
                state=QtGui.QIcon.State.Off,
            )
            word_icon.addFile(
                ":icons/dark/actions/edit-word.svg",
                mode=QtGui.QIcon.Mode.Normal,
                state=QtGui.QIcon.State.Off,
            )
            case_icon.addFile(
                ":icons/dark/actions/edit-case.svg",
                mode=QtGui.QIcon.Mode.Normal,
                state=QtGui.QIcon.State.Off,
            )
        else:
            regex_icon.addFile(
                ":icons/light/actions/edit-regex.svg",
                mode=QtGui.QIcon.Mode.Normal,
                state=QtGui.QIcon.State.Off,
            )
            word_icon.addFile(
                ":icons/light/actions/edit-word.svg",
                mode=QtGui.QIcon.Mode.Normal,
                state=QtGui.QIcon.State.Off,
            )
            case_icon.addFile(
                ":icons/light/actions/edit-case.svg",
                mode=QtGui.QIcon.Mode.Normal,
                state=QtGui.QIcon.State.Off,
            )

        self.regex_action = QtGui.QAction(icon=regex_icon, parent=self)
        self.regex_action.setCheckable(True)

        self.word_action = QtGui.QAction(icon=word_icon, parent=self)
        self.word_action.setCheckable(True)

        self.case_action = QtGui.QAction(icon=case_icon, parent=self)
        self.case_action.setCheckable(True)

        self.add_internal_action(self.regex_action, "regex_search_field")
        self.add_internal_action(self.word_action, "word_search_field")
        self.add_internal_action(self.case_action, "case_search_field")

    def activate(self):
        if self.regex_action.isChecked():
            try:
                _ = re.compile(self.text())
            except Exception:
                self.setError()
                self.validationError.emit("Search regex not valid")
                return
        self.searchActivated.emit(self.query())

    def setQuery(self, query: TextFilterQuery):
        self.setText(query.text)
        with QtCore.QSignalBlocker(self.regex_action) as _, QtCore.QSignalBlocker(
            self.word_action
        ) as _:
            self.regex_action.setChecked(query.regex)
            self.word_action.setChecked(query.word)
            self.case_action.setChecked(query.case_sensitive)
            self.activate()

    def query(self) -> typing.Optional[TextFilterQuery]:
        if not super().text():
            return None
        filter = TextFilterQuery(
            super().text(),
            self.regex_action.isChecked(),
            self.word_action.isChecked(),
            self.case_action.isChecked(),
        )
        return filter


class FileListWidget(QtWidgets.QWidget):  # pragma: no cover
    fileChanged = QtCore.Signal(object)
    viewEditRequested = QtCore.Signal(object)

    def __init__(self, *args):
        super(FileListWidget, self).__init__(*args)
        self.settings = SplaatSettings()
        self.setMinimumWidth(100)
        self.corpus_model: typing.Optional[CorpusModel] = None
        layout = QtWidgets.QVBoxLayout()

        self.search_text_box = SearchBox(self)
        self.search_phones_box = SearchBox(self)
        search_layout = QtWidgets.QFormLayout()
        self.search_widget = QtWidgets.QWidget()
        search_layout.addRow("Text", self.search_text_box)
        search_layout.addRow("Phones", self.search_phones_box)
        self.search_widget.setLayout(search_layout)
        layout.addWidget(self.search_widget)
        self.search_text_box.searchActivated.connect(self.search_text)
        self.search_phones_box.searchActivated.connect(self.search_phones)

        self.table_widget = FileListTable(self)
        self.table_widget.viewEditRequested.connect(self.viewEditRequested.emit)
        layout.addWidget(self.table_widget)

        self.pagination_toolbar = PaginationWidget()
        self.pagination_toolbar.pageRequested.connect(self.table_widget.scrollToTop)
        layout.addWidget(self.pagination_toolbar)
        self.setLayout(layout)
        self.refresh_settings()

    def search_text(self):
        query = self.search_text_box.query()
        self.pagination_toolbar.reset()
        self.corpus_model.current_offset = 0
        self.corpus_model.search_text(
            query,
        )

    def search_phones(self):
        query = self.search_phones_box.query()
        self.pagination_toolbar.reset()
        self.corpus_model.current_offset = 0
        self.corpus_model.search_phones(
            query,
        )

    def query_started(self):
        self.table_widget.setVisible(True)
        self.pagination_toolbar.setVisible(True)

    def query_finished(self):
        self.table_widget.setVisible(True)
        self.pagination_toolbar.setVisible(True)

    def set_models(
        self,
        corpus_model: CorpusModel,
    ):
        self.corpus_model: CorpusModel = corpus_model
        self.table_widget.setModel(corpus_model)
        self.corpus_model.resultCountChanged.connect(self.pagination_toolbar.update_result_count)
        self.pagination_toolbar.offsetRequested.connect(self.corpus_model.set_offset)
        self.corpus_model.newResults.connect(self.query_finished)

    def refresh_settings(self):
        self.settings.sync()
        self.table_widget.refresh_settings()
        self.pagination_toolbar.set_limit(self.settings.value(self.settings.RESULTS_PER_PAGE))


class DetailView(QtWidgets.QWidget):
    undoRequested = QtCore.Signal()
    redoRequested = QtCore.Signal()
    playRequested = QtCore.Signal()

    def __init__(self, *args):
        super().__init__(*args)

        layout = QtWidgets.QVBoxLayout()
        self.settings = SplaatSettings()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.corpus_model = None
        self.file_model = None
        self.selection_model = None
        self.plot_widget = SplaatPlot(self)

        layout.addWidget(self.plot_widget)

        self.scroll_bar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal, self)

        # self.scroll_bar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.scroll_bar.valueChanged.connect(self.update_from_slider)
        layout.addWidget(self.scroll_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.show_all_speakers = False

    def refresh_settings(self):
        pass

    def set_models(
        self,
        corpus_model: CorpusModel,
        file_model: FileModel,
        selection_model: FileSelectionModel,
    ):
        self.corpus_model = corpus_model
        self.file_model = file_model
        self.selection_model = selection_model
        self.selection_model.viewChanged.connect(self.update_to_slider)
        self.selection_model.fileChanged.connect(self.update_to_slider)
        self.plot_widget.set_models(corpus_model, file_model, selection_model)

    def update_to_slider(self):
        with QtCore.QSignalBlocker(self.scroll_bar):
            if self.selection_model.model().file is None or self.selection_model.min_time is None:
                return
            duration_ms = int(self.selection_model.model().file.duration * 1000)
            begin = self.selection_model.min_time * 1000
            end = self.selection_model.max_time * 1000
            window_size_ms = int(end - begin)
            self.scroll_bar.setMinimum(0)
            self.scroll_bar.setEnabled(True)
            self.scroll_bar.setPageStep(window_size_ms)
            self.scroll_bar.setSingleStep(int(window_size_ms * 0.5))
            self.scroll_bar.setMaximum(duration_ms - window_size_ms)
            self.scroll_bar.setValue(begin)

    def set_search_term(self, text_search_term, phones_search_term):
        self.plot_widget.set_search_term(text_search_term, phones_search_term)

    def update_from_slider(self, value: int):
        new_begin = value / 1000
        window_size = self.scroll_bar.pageStep() / 1000
        if new_begin + window_size > self.selection_model.model().file.duration:
            new_begin = self.selection_model.model().file.duration - window_size
        elif new_begin < 0:
            new_begin = 0
        self.selection_model.update_from_slider(new_begin)

    def pan_left(self):
        self.scroll_bar.triggerAction(self.scroll_bar.SliderAction.SliderSingleStepSub)

    def pan_right(self):
        self.scroll_bar.triggerAction(self.scroll_bar.SliderAction.SliderSingleStepAdd)


class DetailedMessageBox(QtWidgets.QDialog):  # pragma: no cover
    reportBug = QtCore.Signal()

    def __init__(self, detailed_message, *args, **kwargs):
        super(DetailedMessageBox, self).__init__(*args, **kwargs)
        from splaat.desktop.ui_error_dialog import Ui_ErrorDialog

        self.ui = Ui_ErrorDialog()
        self.ui.setupUi(self)
        self.settings = SplaatSettings()
        icon = QtGui.QIcon.fromTheme("emblem-important")
        size = self.ui.iconLabel.size()
        self.ui.iconLabel.setPixmap(icon.pixmap(size))
        self.setWindowIcon(icon)
        self.ui.detailed_message.setText(detailed_message)
        self.ui.buttonBox.report_bug_button.clicked.connect(self.reportBug.emit)
        self.ui.buttonBox.rejected.connect(self.reject)


class StoppableProgressBar(QtWidgets.QWidget):
    finished = QtCore.Signal(object)

    def __init__(self, worker: Worker, id, *args):
        super().__init__(*args)
        self.worker = worker
        self.id = id
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.total.connect(self.update_total)
        self.worker.signals.finished.connect(self.update_finished)
        layout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(self.worker.name)
        layout.addWidget(self.label)
        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)
        self.cancel_button = QtWidgets.QToolButton()
        self.cancel_action = QtGui.QAction("select", self)
        self.cancel_action.setIcon(QtGui.QIcon.fromTheme("edit-clear"))
        self.cancel_action.triggered.connect(worker.cancel)
        self.cancel_button.setDefaultAction(self.cancel_action)
        layout.addWidget(self.cancel_button)
        self.setLayout(layout)

    def cancel(self):
        self.progress_bar.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.worker.stopped.stop()

    def update_finished(self):
        self.finished.emit(self.id)

    def update_total(self, total):
        self.progress_bar.setMaximum(total)

    def update_progress(self, progress, time_remaining):
        self.progress_bar.setFormat(f"%v of %m - %p% ({time_remaining} remaining)")
        self.progress_bar.setValue(progress)


class ProgressMenu(QtWidgets.QMenu):
    allDone = QtCore.Signal()

    def __init__(self, *args):
        super(ProgressMenu, self).__init__(*args)
        self.settings = SplaatSettings()
        layout = QtWidgets.QVBoxLayout()
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_layout = QtWidgets.QVBoxLayout()
        self.scroll_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setLayout(self.scroll_layout)
        layout.addWidget(self.scroll_area)
        self.scroll_area.setFixedWidth(
            500 + self.scroll_area.verticalScrollBar().sizeHint().width()
        )
        self.scroll_area.setFixedHeight(300)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.progress_bars: typing.Dict[int, StoppableProgressBar] = {}
        self.setLayout(layout)
        self.current_id = 0

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        p = self.pos()
        geo = self.parent().geometry()
        self.move(
            p.x() + geo.width() - self.geometry().width(),
            p.y() - geo.height() - self.geometry().height(),
        )

    def track_worker(self, worker: Worker):
        self.progress_bars[self.current_id] = StoppableProgressBar(worker, self.current_id)
        self.scroll_area.layout().addWidget(self.progress_bars[self.current_id])
        self.progress_bars[self.current_id].finished.connect(self.update_finished)
        self.current_id += 1

    def update_finished(self, id):
        self.scroll_layout.removeWidget(self.progress_bars[id])
        self.progress_bars[id].deleteLater()
        del self.progress_bars[id]
        if len(self.progress_bars) == 0:
            self.allDone.emit()


class ProgressWidget(QtWidgets.QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.done_icon = QtGui.QIcon.fromTheme("emblem-default")
        self.animated = QtGui.QMovie(":spinning_blue.svg")
        self.animated.frameChanged.connect(self.update_animation)
        self.setIcon(self.done_icon)
        self.menu = ProgressMenu(self)
        self.setMenu(self.menu)
        self.menu.allDone.connect(self.all_done)

    def add_worker(self, worker):
        self.menu.track_worker(worker)
        if self.animated.state() == QtGui.QMovie.MovieState.NotRunning:
            self.animated.start()

    def update_animation(self):
        self.setIcon(QtGui.QIcon(self.animated.currentPixmap()))

    def all_done(self):
        self.setIcon(self.done_icon)
        if self.animated.state() == QtGui.QMovie.MovieState.Running:
            self.animated.stop()
