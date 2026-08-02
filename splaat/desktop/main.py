from __future__ import annotations

import datetime
import logging
import os

import sqlalchemy
from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets

from splaat.desktop import gui_db, workers
from splaat.desktop.models import CorpusModel, FileModel, FileSelectionModel
from splaat.desktop.settings import SplaatSettings
from splaat.desktop.ui_file_detail import Ui_FileDetailWindow
from splaat.desktop.ui_main_window import Ui_MainWindow
from splaat.desktop.ui_preferences import Ui_PreferencesDialog
from splaat.desktop.widgets import DetailedMessageBox, MediaPlayer, ProgressWidget
from splaat.utils import get_temporary_directory

logger = logging.getLogger("splaat")


class Application(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setActiveWindow(self, act):
        super().setActiveWindow(act)
        act.styleSheetChanged.connect(self.setStyleSheet)


class FileDetailWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workers = []
        self.settings = SplaatSettings()
        self.ui = Ui_FileDetailWindow()
        self.ui.setupUi(self)
        self.corpus_model: CorpusModel = None
        self.file_utterances_model: FileModel = None
        self.file_selection_model: FileSelectionModel = None
        if self.settings.contains(SplaatSettings.DETAIL_GEOMETRY):
            self.restoreGeometry(self.settings.value(SplaatSettings.DETAIL_GEOMETRY))
        self.media_player = MediaPlayer(self)
        self.media_player.playbackStateChanged.connect(self.handleAudioState)
        self.media_player.audioReady.connect(self.file_loaded)
        self.media_player.playingChanged.connect(self.update_play_button)
        self.media_player.timeChanged.connect(
            self.ui.centralwidget.plot_widget.audio_plot.update_play_line
        )
        if self.settings.contains(SplaatSettings.VOLUME):
            self.media_player.set_volume(self.settings.value(SplaatSettings.VOLUME))
        self.volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.channel_select = QtWidgets.QComboBox(self)
        self.channel_select.addItem("Channel 0")
        self.ui.toolBar.addWidget(self.volume_slider)
        self.ui.toolBar.addWidget(self.channel_select)

        self.undo_group = QtGui.QUndoGroup(self)
        self.file_undo_stack = QtGui.QUndoStack(self)
        self.undo_act = self.undo_group.createUndoAction(self, "Undo")
        self.redo_act = self.undo_group.createRedoAction(self, "Redo")
        self.create_actions()
        self.refresh_settings()

    def play_audio(self):
        if self.media_player.playbackState() in [
            QtMultimedia.QMediaPlayer.PlaybackState.StoppedState,
            QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
        ]:
            self.media_player.play()
        elif (
            self.media_player.playbackState()
            == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState
        ):
            self.media_player.pause()

    def handleAudioState(self, state):
        if state == QtMultimedia.QMediaPlayer.PlaybackState.StoppedState:
            self.ui.playAct.setChecked(False)

    def file_loaded(self, ready):
        self.ui.playAct.setEnabled(ready)

    def update_play_button(self, playing):
        self.ui.playAct.setChecked(playing)

    def open_options(self):
        dialog = OptionsDialog(self)
        if dialog.exec_():
            self.settings.sync()
            self.refresh_settings()

    def refresh_settings(self):
        self.refresh_style_sheets()
        self.refresh_shortcuts()
        self.media_player.refresh_settings()
        self.ui.centralwidget.refresh_settings()

    def refresh_shortcuts(self):
        self.ui.playAct.setShortcut(self.settings.value(SplaatSettings.PLAY_KEYBIND))
        self.ui.zoomInAct.setShortcut(self.settings.value(SplaatSettings.ZOOM_IN_KEYBIND))
        self.ui.zoomOutAct.setShortcut(self.settings.value(SplaatSettings.ZOOM_OUT_KEYBIND))
        self.ui.zoomAllAct.setShortcut(self.settings.value(SplaatSettings.ZOOM_ALL_KEYBIND))
        self.ui.zoomToSelectionAct.setShortcut(
            self.settings.value(SplaatSettings.ZOOM_TO_SELECTION_KEYBIND)
        )
        self.ui.panLeftAct.setShortcut(self.settings.value(SplaatSettings.PAN_LEFT_KEYBIND))
        self.ui.panRightAct.setShortcut(self.settings.value(SplaatSettings.PAN_RIGHT_KEYBIND))
        self.ui.searchAct.setShortcut(self.settings.value(SplaatSettings.SEARCH_KEYBIND))
        self.undo_act.setShortcut(self.settings.value(SplaatSettings.UNDO_KEYBIND))
        self.redo_act.setShortcut(self.settings.value(SplaatSettings.REDO_KEYBIND))

    def refresh_style_sheets(self):
        dark_mode = QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        if dark_mode:
            QtGui.QIcon.setThemeName("dark")
        else:
            QtGui.QIcon.setThemeName("light")

    def create_actions(self):
        self.ui.actionPreferences.triggered.connect(self.open_options)
        self.ui.playAct.triggered.connect(self.play_audio)
        self.media_player.playbackStateChanged.connect(self.update_play_act)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.setValue(self.settings.value(self.settings.VOLUME))
        self.volume_slider.valueChanged.connect(self.media_player.set_volume)

        self.undo_act.setIcon(QtGui.QIcon.fromTheme("edit-undo"))
        self.redo_act.setIcon(QtGui.QIcon.fromTheme("edit-redo"))
        self.ui.menuEdit.addAction(self.undo_act)
        self.ui.menuEdit.addAction(self.redo_act)
        self.undo_group.setActiveStack(self.file_undo_stack)

    def update_play_act(self, state):
        if state == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self.ui.playAct.setChecked(True)
        else:
            self.ui.playAct.setChecked(False)

    def set_models(self, corpus_model, file_id):
        self.corpus_model = corpus_model
        self.file_utterances_model = FileModel(self)
        self.file_selection_model = FileSelectionModel(self.file_utterances_model)
        self.file_utterances_model.set_corpus_model(self.corpus_model)
        self.file_utterances_model.addCommand.connect(self.update_undo_stack)
        self.ui.panLeftAct.triggered.connect(self.file_selection_model.pan_left)
        self.ui.panRightAct.triggered.connect(self.file_selection_model.pan_right)
        self.ui.zoomInAct.triggered.connect(self.file_selection_model.zoom_in)
        self.ui.zoomOutAct.triggered.connect(self.file_selection_model.zoom_out)
        self.ui.zoomToSelectionAct.triggered.connect(self.file_selection_model.zoom_to_selection)
        self.ui.zoomAllAct.triggered.connect(self.file_selection_model.zoom_all)
        self.ui.centralwidget.set_models(
            self.corpus_model, self.file_utterances_model, self.file_selection_model
        )
        self.media_player.set_models(self.file_selection_model)
        self.file_selection_model.set_current_file(file_id, 0, 10000)

    def update_undo_stack(self, command):
        self.undo_group.setActiveStack(self.file_undo_stack)
        self.file_undo_stack.push(command)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.file_utterances_model.clean_up_for_close()
        self.settings.setValue(SplaatSettings.DETAIL_GEOMETRY, self.saveGeometry())
        self.settings.setValue(SplaatSettings.DETAIL_WINDOW_STATE, self.saveState())
        self.settings.sync()
        while self.file_selection_model.thread_pool.activeThreadCount() > 0:
            pass
        a0.accept()


class MainWindow(QtWidgets.QMainWindow):
    configUpdated = QtCore.Signal(object)
    g2pLoaded = QtCore.Signal(object)
    ivectorExtractorLoaded = QtCore.Signal(object)
    acousticModelLoaded = QtCore.Signal(object)
    languageModelLoaded = QtCore.Signal(object)
    newSpeaker = QtCore.Signal(object)
    styleSheetChanged = QtCore.Signal(object)

    def __init__(self, debug=False):
        super().__init__()
        self.workers = []
        self.settings = SplaatSettings()
        fonts = [
            "GentiumPlus",
            "CharisSIL",
            "NotoSans-Black",
            "NotoSans-Bold",
            "NotoSans-BoldItalic",
            "NotoSans-Italic",
            "NotoSans-Light",
            "NotoSans-Medium",
            "NotoSans-MediumItalic",
            "NotoSans-Regular",
            "NotoSans-Thin",
            "NotoSerif-Black",
            "NotoSerif-Bold",
            "NotoSerif-BoldItalic",
            "NotoSerif-Italic",
            "NotoSerif-Light",
            "NotoSerif-Medium",
            "NotoSerif-MediumItalic",
            "NotoSerif-Regular",
            "NotoSerif-Thin",
        ]
        for font in fonts:
            QtGui.QFontDatabase.addApplicationFont(f":fonts/{font}.ttf")
        os.makedirs(get_temporary_directory(), exist_ok=True)
        self._db_engine = None

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.status_indicator = ProgressWidget()
        self.status_indicator.setFixedWidth(self.ui.statusBar.height())
        self.ui.statusBar.addPermanentWidget(self.status_indicator, 0)
        self.debug = debug
        self.corpus_model: CorpusModel = None
        self.initialize_database()
        self.set_up_models()

        self.corpus_worker = workers.ImportCorpusWorker(self)
        self.corpus_worker.signals.result.connect(self.finalize_load_corpus)
        self.corpus_worker.signals.error.connect(self.handle_error)
        self.workers.append(self.corpus_worker)
        if self.settings.contains(SplaatSettings.GEOMETRY):
            self.restoreGeometry(self.settings.value(SplaatSettings.GEOMETRY))

        self.thread_pool = QtCore.QThreadPool()

        self.single_runners = {
            "Counting utterance results": None,
            "Querying utterances": None,
        }
        self.sequential_runners = {}

        if self.settings.value(SplaatSettings.AUTOLOAD):
            self.load_corpus()
        else:
            self.set_application_state("unloaded")

        self.create_actions()
        self.refresh_settings()

    def set_up_models(self):
        self.corpus_model = CorpusModel(self)

        self.ui.centralwidget.set_models(self.corpus_model)
        self.corpus_model.runFunction.connect(self.execute_runnable)

    def execute_runnable(self, function, finished_function, extra_args=None):
        if self.corpus_model.corpus_name is None:
            return
        delayed_start = False
        if function == "Counting utterance results":
            worker = workers.QueryUtterancesWorker(self.corpus_model.session, **extra_args[0])
            worker.signals.result.connect(finished_function)
        elif function in {"Analyzing alignments", "Counting alignment analysis results"}:
            worker = workers.AlignmentAnalysisWorker(self.corpus_model.session, **extra_args[0])
            worker.signals.result.connect(finished_function)
        elif function == "Querying utterances":
            worker = workers.QueryUtterancesWorker(self.corpus_model.session, **extra_args[0])
            worker.signals.result.connect(finished_function)
        elif function == "Loading phones":
            worker = workers.LoadPhonesWorker(self.corpus_model.session, *extra_args)
            worker.signals.result.connect(finished_function)
        elif function == "Loading words":
            worker = workers.LoadWordsWorker(self.corpus_model.session, *extra_args)
            worker.signals.result.connect(finished_function)
        elif function == "Creating speaker tiers":
            worker = workers.FileUtterancesWorker(self.corpus_model.session, *extra_args)
            worker.signals.result.connect(finished_function)
        else:
            if extra_args is None:
                extra_args = []
            worker = workers.Worker(function, *extra_args)
            worker.signals.result.connect(finished_function)
        if function in self.single_runners:
            if self.single_runners[function] is not None:
                self.single_runners[function].cancel()
            self.single_runners[function] = worker
        if function in self.sequential_runners:
            delayed_start = len(self.sequential_runners[function]) > 0
            if delayed_start:
                self.sequential_runners[function][-1].signals.finished.connect(
                    lambda: self.thread_pool.start(worker)
                )
            self.sequential_runners[function].append(worker)
            worker.signals.finished.connect(self.update_sequential_runners)

        worker.signals.error.connect(self.handle_error)
        # Execute
        if not delayed_start:
            self.thread_pool.start(worker)
        if isinstance(function, str):
            worker.name = function
        self.status_indicator.add_worker(worker)

    def update_sequential_runners(self):
        sender = self.sender()
        for k, v in self.sequential_runners.items():
            self.sequential_runners[k] = [x for x in v if x.signals != sender]

    def create_actions(self):
        self.ui.actionOpenFolder.triggered.connect(self.change_corpus)
        self.ui.actionView.triggered.connect(self.show_detail)
        self.ui.centralwidget.viewEditRequested.connect(self.show_detail)
        self.ui.actionExport_modified_files.triggered.connect(self.export_files)

    def export_files(self):
        export_directory = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self,
            caption="Select an export directory",
            dir=self.settings.value(SplaatSettings.DEFAULT_DIRECTORY),
        )
        self.settings.setValue(SplaatSettings.DEFAULT_DIRECTORY, os.path.dirname(export_directory))
        self.corpus_model.export_files(export_directory)

    def show_detail(self):
        rows = self.ui.centralwidget.table_widget.selectionModel().selectedRows()
        for r in sorted(rows):
            file_id = self.corpus_model.file_id_at(r)
            file_name = self.corpus_model.file_name_at(r)
            window = FileDetailWindow(self)
            window.setWindowTitle(f"Splaat - {file_name}")
            window.set_models(self.corpus_model, file_id)
            window.ui.centralwidget.set_search_term(
                self.ui.centralwidget.search_text_box.query(),
                self.ui.centralwidget.search_phones_box.query(),
            )
            window.show()

    @property
    def db_string(self):
        return f"sqlite:///{self.settings.temp_directory.joinpath('splaat_gui.db')}"

    @property
    def db_engine(self) -> sqlalchemy.engine.Engine:
        """Database engine"""
        if self._db_engine is None:
            self._db_engine = sqlalchemy.create_engine(self.db_string)
        return self._db_engine

    def initialize_database(self):
        from splaat.desktop.gui_db import SplaatSqlBase

        if not self.settings.temp_directory.joinpath("splaat_gui.db").exists():
            SplaatSqlBase.metadata.create_all(self.db_engine)

    def change_corpus(self):
        corpus_name = self.sender().text()
        with sqlalchemy.orm.Session(self.db_engine) as session:
            session.query(gui_db.Corpus).update({gui_db.Corpus.current: False})
            session.flush()
            m = session.query(gui_db.Corpus).filter(gui_db.Corpus.name == corpus_name).first()
            if m is None:
                corpus_directory = QtWidgets.QFileDialog.getExistingDirectory(
                    parent=self,
                    caption="Select a corpus directory",
                    dir=self.settings.value(SplaatSettings.DEFAULT_DIRECTORY),
                )
                if not corpus_directory or not os.path.exists(corpus_directory):
                    return
                corpus_name = os.path.basename(corpus_directory)
                self.settings.temp_directory.joinpath(corpus_name).mkdir(
                    exist_ok=True, parents=True
                )
                self.settings.setValue(
                    SplaatSettings.DEFAULT_DIRECTORY, os.path.dirname(corpus_directory)
                )
                m = session.query(gui_db.Corpus).filter(gui_db.Corpus.name == corpus_name).first()
                if m is None:
                    m = gui_db.Corpus(name=corpus_name, path=corpus_directory, current=True)
                    session.add(m)
            m.current = True
            m.last_used = datetime.datetime.now()
            session.commit()
        self.refresh_corpus_history()
        self.close_corpus()
        self.load_corpus()

    def close_corpus(self):
        self.set_application_state("unloaded")
        if self.corpus_model.corpus_name is not None:
            self.corpus_model.clear_data()
            self.corpus_model.session.close()
            del self.corpus_model.session
        self.corpus_model.setCorpus(None)

    def set_application_state(self, state, worker=None):
        if state == "loading":
            self.ui.actionOpenFolder.setEnabled(False)
        elif state == "loaded":
            self.ui.actionOpenFolder.setEnabled(True)
        elif state == "unloaded":
            self.ui.actionOpenFolder.setEnabled(True)

    def refresh_corpus_history(self):
        self.ui.loadRecentFoldersMenu.clear()
        with sqlalchemy.orm.Session(self.db_engine) as session:
            corpora = (
                session.query(gui_db.Corpus)
                # .filter_by(current=False)
                .order_by(gui_db.Corpus.last_used.desc())
                .order_by(gui_db.Corpus.id.desc())
                .limit(10)
            )
            for c in corpora:
                a = QtGui.QAction(c.name, parent=self)
                if c.current:
                    a.setChecked(True)
                a.triggered.connect(self.change_corpus)
                self.ui.loadRecentFoldersMenu.addAction(a)

    def load_corpus(self):
        with sqlalchemy.orm.Session(self.db_engine) as session:
            c = session.query(gui_db.Corpus).filter_by(current=True).first()
            if c is None:
                self.set_application_state("unloaded")
                return
            self.set_application_state("loading")
        self.corpus_worker.set_params(c.path)
        self.corpus_worker.start()

    def handle_error(self, trace_args):
        exctype, value, trace = trace_args
        reply = DetailedMessageBox(detailed_message=trace)
        reply.reportBug.connect(self.ui.actionReport_bug.trigger)
        _ = reply.exec_()
        self.check_actions()
        if self.corpus_model is not None:
            self.set_application_state("loaded")

    def check_actions(self):
        self.ui.actionReport_bug.setEnabled(True)
        self.ui.actionOpenFolder.setEnabled(True)
        self.ui.actionPreferences.setEnabled(True)
        self.ui.actionOpenGuidelines.setEnabled(True)

        if self.corpus_model.corpus_name is None:
            self.ui.actionInfo.setEnabled(False)
            self.ui.actionExport_modified_files.setEnabled(False)
            self.ui.actionView.setEnabled(False)
            self.ui.actionRemove.setEnabled(False)
        else:
            self.ui.actionInfo.setEnabled(True)
            self.ui.actionExport_modified_files.setEnabled(True)
            self.ui.actionView.setEnabled(True)
            self.ui.actionRemove.setEnabled(True)

    def finalize_load_corpus(self, corpus_name: str):
        if corpus_name is None:
            self.set_application_state("unloaded")
        self.corpus_model.setCorpus(corpus_name)
        self.check_actions()

    def refresh_settings(self):
        self.refresh_style_sheets()
        self.corpus_model.set_limit(self.settings.value(self.settings.RESULTS_PER_PAGE))
        self.ui.centralwidget.refresh_settings()

    def refresh_style_sheets(self):
        dark_mode = QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        if dark_mode:
            QtGui.QIcon.setThemeName("dark")
        else:
            QtGui.QIcon.setThemeName("light")

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        for worker in self.workers:
            worker.stopped.set()
        self.settings.setValue(SplaatSettings.GEOMETRY, self.saveGeometry())
        self.settings.setValue(SplaatSettings.WINDOW_STATE, self.saveState())

        self.settings.sync()
        self.set_application_state("loading")
        self.close_timer = QtCore.QTimer()
        self.close_timer.timeout.connect(lambda: self._actual_close(a0))
        self.close_timer.start(1000)

    def _actual_close(self, a0):
        for worker in self.workers:
            if not worker.finished():
                return
        if self.thread_pool.activeThreadCount() > 0:
            return
        if self.corpus_model.session is not None:
            self.corpus_model.session = None
            sqlalchemy.orm.close_all_sessions()
        a0.accept()


class OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_PreferencesDialog()
        self.ui.setupUi(self)
        self.settings = SplaatSettings()

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.ui.fontEdit.set_font(self.settings.font)

        self.ui.playAudioShortcutEdit.setKeySequence(
            self.settings.value(self.settings.PLAY_KEYBIND)
        )
        self.ui.zoomInShortcutEdit.setKeySequence(
            self.settings.value(self.settings.ZOOM_IN_KEYBIND)
        )
        self.ui.zoomToSelectionShortcutEdit.setKeySequence(
            self.settings.value(self.settings.ZOOM_TO_SELECTION_KEYBIND)
        )
        self.ui.zoomOutShortcutEdit.setKeySequence(
            self.settings.value(self.settings.ZOOM_OUT_KEYBIND)
        )
        self.ui.panLeftShortcutEdit.setKeySequence(
            self.settings.value(self.settings.PAN_LEFT_KEYBIND)
        )
        self.ui.panRightShortcutEdit.setKeySequence(
            self.settings.value(self.settings.PAN_RIGHT_KEYBIND)
        )
        self.ui.saveShortcutEdit.setKeySequence(self.settings.value(self.settings.SAVE_KEYBIND))
        self.ui.searchShortcutEdit.setKeySequence(
            self.settings.value(self.settings.SEARCH_KEYBIND)
        )
        self.ui.undoShortcutEdit.setKeySequence(self.settings.value(self.settings.UNDO_KEYBIND))
        self.ui.redoShortcutEdit.setKeySequence(self.settings.value(self.settings.REDO_KEYBIND))

        self.ui.autosaveOnExitCheckBox.setChecked(self.settings.value(self.settings.AUTOSAVE))

        self.ui.autoloadLastUsedCorpusCheckBox.setChecked(
            self.settings.value(self.settings.AUTOLOAD)
        )
        self.ui.enableFadeCheckBox.setChecked(self.settings.value(self.settings.ENABLE_FADE))
        self.ui.resultsPerPageEdit.setValue(self.settings.value(self.settings.RESULTS_PER_PAGE))
        self.ui.timeDirectionComboBox.setCurrentIndex(
            self.ui.timeDirectionComboBox.findText(
                self.settings.value(self.settings.TIME_DIRECTION)
            )
        )

        self.ui.dynamicRangeEdit.setValue(self.settings.value(self.settings.SPEC_DYNAMIC_RANGE))
        self.ui.specMaxTimeEdit.setText(str(self.settings.value(self.settings.SPEC_MAX_TIME)))
        self.ui.fftSizeEdit.setValue(self.settings.value(self.settings.SPEC_N_FFT))
        self.ui.numTimeStepsEdit.setValue(self.settings.value(self.settings.SPEC_N_TIME_STEPS))
        self.ui.windowSizeEdit.setText(str(self.settings.value(self.settings.SPEC_WINDOW_SIZE)))
        self.ui.preemphasisEdit.setText(str(self.settings.value(self.settings.SPEC_PREEMPH)))
        self.ui.maxFrequencyEdit.setValue(self.settings.value(self.settings.SPEC_MAX_FREQ))

        self.setWindowTitle("Preferences")
        self.ui.tabWidget.setCurrentIndex(0)

    def accept(self) -> None:
        self.settings.setValue(
            self.settings.SPEC_DYNAMIC_RANGE, int(self.ui.dynamicRangeEdit.value())
        )
        self.settings.setValue(self.settings.SPEC_N_FFT, int(self.ui.fftSizeEdit.value()))
        self.settings.setValue(
            self.settings.SPEC_N_TIME_STEPS, int(self.ui.numTimeStepsEdit.value())
        )
        self.settings.setValue(
            self.settings.SPEC_WINDOW_SIZE, float(self.ui.windowSizeEdit.text())
        )
        self.settings.setValue(self.settings.SPEC_PREEMPH, float(self.ui.preemphasisEdit.text()))
        self.settings.setValue(self.settings.SPEC_MAX_TIME, float(self.ui.specMaxTimeEdit.text()))
        self.settings.setValue(self.settings.SPEC_MAX_FREQ, int(self.ui.maxFrequencyEdit.value()))

        self.settings.setValue(self.settings.FONT, self.ui.fontEdit.font.toString())

        self.settings.setValue(
            self.settings.PLAY_KEYBIND, self.ui.playAudioShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.ZOOM_IN_KEYBIND, self.ui.zoomInShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.ZOOM_OUT_KEYBIND, self.ui.zoomOutShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.ZOOM_TO_SELECTION_KEYBIND,
            self.ui.zoomToSelectionShortcutEdit.keySequence().toString(),
        )
        self.settings.setValue(
            self.settings.PAN_LEFT_KEYBIND, self.ui.panLeftShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.PAN_RIGHT_KEYBIND, self.ui.panRightShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.SAVE_KEYBIND, self.ui.saveShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.SEARCH_KEYBIND, self.ui.searchShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.UNDO_KEYBIND, self.ui.undoShortcutEdit.keySequence().toString()
        )
        self.settings.setValue(
            self.settings.REDO_KEYBIND, self.ui.redoShortcutEdit.keySequence().toString()
        )

        self.settings.setValue(
            self.settings.AUTOLOAD, self.ui.autoloadLastUsedCorpusCheckBox.isChecked()
        )
        self.settings.setValue(self.settings.ENABLE_FADE, self.ui.enableFadeCheckBox.isChecked())
        self.settings.setValue(self.settings.AUTOSAVE, self.ui.autosaveOnExitCheckBox.isChecked())
        self.settings.setValue(self.settings.RESULTS_PER_PAGE, self.ui.resultsPerPageEdit.value())
        self.settings.setValue(
            self.settings.TIME_DIRECTION, self.ui.timeDirectionComboBox.currentText()
        )
        self.settings.sync()
        super().accept()

    def reject(self):
        self.settings.sync()
        super().reject()
