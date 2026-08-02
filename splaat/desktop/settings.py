import dataclasses
import os.path
import pathlib
from typing import Any, Optional

from PySide6 import QtCore, QtGui, QtMultimedia

from splaat.utils import get_temporary_directory


@dataclasses.dataclass
class PlotTheme:
    background_color: QtGui.QColor
    error_color: QtGui.QColor
    error_text_color: QtGui.QColor
    play_line_color: QtGui.QColor
    selected_range_color: QtGui.QColor
    selected_interval_color: QtGui.QColor
    hover_line_color: QtGui.QColor
    moving_line_color: QtGui.QColor
    break_line_color: QtGui.QColor
    wave_line_color: QtGui.QColor
    text_color: QtGui.QColor
    selected_text_color: QtGui.QColor
    axis_color: QtGui.QColor
    interval_background_color: QtGui.QColor
    pitch_color: QtGui.QColor
    spectrogram_color: QtGui.QColor


class SplaatSettings(QtCore.QSettings):
    themeUpdated = QtCore.Signal()

    TEMPORARY_DIRECTORY = "splaat/temporary_directory"
    DEFAULT_DIRECTORY = "splaat/default_directory"

    CORPUS_PATH = "splaat/path"

    AUTOSAVE = "splaat/autosave"
    AUTOLOAD = "splaat/autoload"

    VOLUME = "splaat/audio/volume"
    ENABLE_FADE = "splaat/enable_fade"

    GEOMETRY = "splaat/MainWindow/geometry"
    DETAIL_GEOMETRY = "splaat/DetailView/geometry"
    WINDOW_STATE = "splaat/MainWindow/windowState"
    DETAIL_WINDOW_STATE = "splaat/DetailView/windowState"

    FONT = "splaat/theme/font"
    MAIN_TEXT_COLOR = "splaat/theme/text_color"
    THEME_PRESET = "splaat/theme/theme_preset"
    SELECTED_TEXT_COLOR = "splaat/theme/selected_text_color"
    ERROR_COLOR = "splaat/theme/error_color"
    PRIMARY_BASE_COLOR = "splaat/theme/primary_color/base"
    PRIMARY_LIGHT_COLOR = "splaat/theme/primary_color/light"
    PRIMARY_DARK_COLOR = "splaat/theme/primary_color/dark"
    PRIMARY_VERY_LIGHT_COLOR = "splaat/theme/primary_color/very_light"
    PRIMARY_VERY_DARK_COLOR = "splaat/theme/primary_color/very_dark"
    ACCENT_BASE_COLOR = "splaat/theme/accent_color/base"
    ACCENT_LIGHT_COLOR = "splaat/theme/accent_color/light"
    ACCENT_DARK_COLOR = "splaat/theme/accent_color/dark"
    ACCENT_VERY_LIGHT_COLOR = "splaat/theme/accent_color/very_light"
    ACCENT_VERY_DARK_COLOR = "splaat/theme/accent_color/very_dark"

    PLAY_KEYBIND = "splaat/keybinds/play"
    DELETE_KEYBIND = "splaat/keybinds/delete"
    SAVE_KEYBIND = "splaat/keybinds/save"
    SEARCH_KEYBIND = "splaat/keybinds/search"
    SPLIT_KEYBIND = "splaat/keybinds/split"
    MERGE_KEYBIND = "splaat/keybinds/merge"
    ZOOM_IN_KEYBIND = "splaat/keybinds/zoom_in"
    ZOOM_OUT_KEYBIND = "splaat/keybinds/zoom_out"
    ZOOM_ALL_KEYBIND = "splaat/keybinds/zoom_all"
    ZOOM_TO_SELECTION_KEYBIND = "splaat/keybinds/zoom_to_selection"
    PAN_LEFT_KEYBIND = "splaat/keybinds/pan_left"
    PAN_RIGHT_KEYBIND = "splaat/keybinds/pan_right"
    UNDO_KEYBIND = "splaat/keybinds/undo"
    REDO_KEYBIND = "splaat/keybinds/redo"
    TIME_DIRECTION = "splaat/time_direction"
    RTL = "Right-to-left"
    LTR = "Left-to-right"

    RESULTS_PER_PAGE = "splaat/results_per_page"
    SPEC_MAX_TIME = "splaat/spectrogram/max_time"
    SPEC_DYNAMIC_RANGE = "splaat/spectrogram/dynamic_range"
    SPEC_N_FFT = "splaat/spectrogram/n_fft"
    SPEC_N_TIME_STEPS = "splaat/spectrogram/time_steps"
    SPEC_WINDOW_SIZE = "splaat/spectrogram/window_size"
    SPEC_PREEMPH = "splaat/spectrogram/preemphasis"
    SPEC_MAX_FREQ = "splaat/spectrogram/max_frequency"

    PLOT_THREAD_COUNT = "splaat/plot/max_thread_count"

    def __init__(self, *args):
        super(SplaatSettings, self).__init__(
            QtCore.QSettings.Format.NativeFormat,
            QtCore.QSettings.Scope.UserScope,
            "Splaat Lab",
            "Splaat",
        )

        self.default_values = {
            SplaatSettings.DEFAULT_DIRECTORY: os.path.expanduser("~"),
            SplaatSettings.TEMPORARY_DIRECTORY: str(get_temporary_directory()),
            SplaatSettings.AUTOSAVE: False,
            SplaatSettings.AUTOLOAD: False,
            SplaatSettings.VOLUME: 100,
            SplaatSettings.ENABLE_FADE: True,
            SplaatSettings.GEOMETRY: None,
            SplaatSettings.DETAIL_GEOMETRY: None,
            SplaatSettings.WINDOW_STATE: None,
            SplaatSettings.DETAIL_WINDOW_STATE: None,
            SplaatSettings.FONT: QtGui.QFont("Noto Sans", 12).toString(),
            SplaatSettings.PLAY_KEYBIND: "Tab",
            SplaatSettings.DELETE_KEYBIND: "Delete",
            SplaatSettings.SAVE_KEYBIND: "Ctrl+S",
            SplaatSettings.SEARCH_KEYBIND: "Ctrl+F",
            SplaatSettings.SPLIT_KEYBIND: "Ctrl+D",
            SplaatSettings.MERGE_KEYBIND: "Ctrl+M",
            SplaatSettings.ZOOM_IN_KEYBIND: "Ctrl+I",
            SplaatSettings.ZOOM_OUT_KEYBIND: "Ctrl+O",
            SplaatSettings.ZOOM_TO_SELECTION_KEYBIND: "Ctrl+N",
            SplaatSettings.ZOOM_ALL_KEYBIND: "Ctrl+A",
            SplaatSettings.PAN_LEFT_KEYBIND: "Left",
            SplaatSettings.PAN_RIGHT_KEYBIND: "Right",
            SplaatSettings.UNDO_KEYBIND: "Ctrl+Z",
            SplaatSettings.REDO_KEYBIND: "Ctrl+Shift+Z",
            SplaatSettings.RESULTS_PER_PAGE: 100,
            SplaatSettings.SPEC_MAX_TIME: 30,
            SplaatSettings.SPEC_DYNAMIC_RANGE: 50,
            SplaatSettings.SPEC_N_FFT: 256,
            SplaatSettings.SPEC_N_TIME_STEPS: 1000,
            SplaatSettings.SPEC_MAX_FREQ: 5000,
            SplaatSettings.SPEC_WINDOW_SIZE: 0.005,
            SplaatSettings.SPEC_PREEMPH: 0.97,
            SplaatSettings.TIME_DIRECTION: SplaatSettings.LTR,
            SplaatSettings.PLOT_THREAD_COUNT: 10,
        }
        self.border_radius = 5
        self.text_padding = 2
        self.border_width = 2
        self.base_menu_button_width = 16
        self.menu_button_width = self.base_menu_button_width + self.border_width * 2

        self.sort_indicator_size = 20
        self.sort_indicator_padding = 15
        self.scroll_bar_height = 25
        self.icon_size = 25
        self.scroll_bar_border_radius = int(self.scroll_bar_height / 2) - 2

    @property
    def all_keybinds(self):
        return {
            QtGui.QKeySequence(self.value(x))
            for x in [
                self.DELETE_KEYBIND,
                self.MERGE_KEYBIND,
                self.PAN_LEFT_KEYBIND,
                self.PAN_RIGHT_KEYBIND,
                self.PLAY_KEYBIND,
                self.REDO_KEYBIND,
                self.UNDO_KEYBIND,
                self.SAVE_KEYBIND,
                self.SEARCH_KEYBIND,
                self.SPLIT_KEYBIND,
                self.ZOOM_IN_KEYBIND,
                self.ZOOM_OUT_KEYBIND,
                self.ZOOM_TO_SELECTION_KEYBIND,
            ]
        }

    @property
    def right_to_left(self) -> bool:
        return self.value(SplaatSettings.TIME_DIRECTION) == SplaatSettings.RTL

    def value(self, arg__1: str, defaultValue: Optional[Any] = ..., t: object = ...) -> Any:
        if arg__1 == SplaatSettings.FONT:
            value = QtGui.QFont()
            value.fromString(
                super(SplaatSettings, self).value(arg__1, self.default_values[arg__1])
            )
        elif "color" in arg__1:
            value = super(SplaatSettings, self).value(arg__1, self.default_values[arg__1])
            if value is None:
                value = self.default_values[arg__1]
            value = QtGui.QColor(value)
        elif "keybind" in arg__1:
            value = QtGui.QKeySequence(
                super(SplaatSettings, self).value(arg__1, self.default_values[arg__1])
            )
        elif "auto" in arg__1:
            value = super(SplaatSettings, self).value(arg__1, self.default_values[arg__1], bool)
        elif arg__1 in {
            SplaatSettings.GEOMETRY,
            SplaatSettings.DETAIL_GEOMETRY,
            SplaatSettings.WINDOW_STATE,
            SplaatSettings.DETAIL_WINDOW_STATE,
        }:
            value = super(SplaatSettings, self).value(arg__1, self.default_values[arg__1])
        else:
            value = super(SplaatSettings, self).value(
                arg__1,
                self.default_values.get(arg__1, ""),
                type=type(self.default_values.get(arg__1, "")),
            )
            if isinstance(value, float):
                value = round(value, 6)

        return value

    @property
    def temp_directory(self) -> pathlib.Path:
        return pathlib.Path(self.value(SplaatSettings.TEMPORARY_DIRECTORY))

    @property
    def font(self) -> QtGui.QFont:
        font = self.value(SplaatSettings.FONT)
        return font

    @property
    def big_font(self) -> QtGui.QFont:
        font = self.value(SplaatSettings.FONT)
        font.setPointSize(int(1.25 * font.pointSize()))
        return font

    @property
    def small_font(self) -> QtGui.QFont:
        font = self.value(SplaatSettings.FONT)
        font.setPointSize(int(0.75 * font.pointSize()))
        return font

    @property
    def title_font(self) -> QtGui.QFont:
        font = self.value(SplaatSettings.FONT)
        font.setPointSize(int(3 * font.pointSize()))
        return font

    @property
    def plot_theme(self) -> PlotTheme:
        dark_mode = QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        palette = QtGui.QGuiApplication.palette()
        interval_background_color = palette.color(palette.ColorGroup.Active, palette.ColorRole.Mid)
        if dark_mode:
            background_color = palette.color(palette.ColorGroup.Active, palette.ColorRole.Shadow)
            spectrogram_color = palette.color(
                palette.ColorGroup.Active, palette.ColorRole.BrightText
            )
        else:
            background_color = palette.color(palette.ColorGroup.Active, palette.ColorRole.Base)
            spectrogram_color = palette.color(palette.ColorGroup.Active, palette.ColorRole.Shadow)
        return PlotTheme(
            **{
                "background_color": background_color,
                "error_color": palette.color(palette.ColorGroup.Active, palette.ColorRole.Accent),
                "error_text_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.Dark
                ),
                "play_line_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.Accent
                ),
                "selected_range_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.Highlight
                ),
                "selected_interval_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.Highlight
                ),
                "hover_line_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.Accent
                ),
                "moving_line_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.Accent
                ),
                "break_line_color": palette.color(
                    palette.ColorGroup.Inactive, palette.ColorRole.ButtonText
                ),
                "wave_line_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.WindowText
                ),
                "text_color": palette.color(palette.ColorGroup.Active, palette.ColorRole.Text),
                "selected_text_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.HighlightedText
                ),
                "axis_color": palette.color(palette.ColorGroup.Active, palette.ColorRole.Accent),
                "interval_background_color": interval_background_color,
                "pitch_color": palette.color(
                    palette.ColorGroup.Active, palette.ColorRole.Highlight
                ),
                "spectrogram_color": spectrogram_color,
            }
        )

    @property
    def error_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.ERROR_COLOR)

    @property
    def selected_text_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.SELECTED_TEXT_COLOR)

    @property
    def text_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.MAIN_TEXT_COLOR)

    @property
    def primary_base_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.PRIMARY_BASE_COLOR)

    @property
    def primary_light_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.PRIMARY_LIGHT_COLOR)

    @property
    def primary_dark_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.PRIMARY_DARK_COLOR)

    @property
    def primary_very_light_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.PRIMARY_VERY_LIGHT_COLOR)

    @property
    def primary_very_dark_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.PRIMARY_VERY_DARK_COLOR)

    @property
    def accent_base_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.ACCENT_BASE_COLOR)

    @property
    def accent_light_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.ACCENT_LIGHT_COLOR)

    @property
    def accent_dark_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.ACCENT_DARK_COLOR)

    @property
    def accent_very_light_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.ACCENT_VERY_LIGHT_COLOR)

    @property
    def accent_very_dark_color(self) -> QtGui.QColor:
        return self.value(SplaatSettings.ACCENT_VERY_DARK_COLOR)

    @property
    def current_theme(self):
        return {
            SplaatSettings.MAIN_TEXT_COLOR: self.text_color,
            SplaatSettings.SELECTED_TEXT_COLOR: self.selected_text_color,
            SplaatSettings.ERROR_COLOR: self.error_color,
            SplaatSettings.PRIMARY_BASE_COLOR: self.primary_base_color,
            SplaatSettings.PRIMARY_LIGHT_COLOR: self.primary_light_color,
            SplaatSettings.PRIMARY_DARK_COLOR: self.primary_dark_color,
            SplaatSettings.PRIMARY_VERY_LIGHT_COLOR: self.primary_very_light_color,
            SplaatSettings.PRIMARY_VERY_DARK_COLOR: self.primary_very_dark_color,
            SplaatSettings.ACCENT_BASE_COLOR: self.accent_base_color,
            SplaatSettings.ACCENT_LIGHT_COLOR: self.accent_light_color,
            SplaatSettings.ACCENT_DARK_COLOR: self.accent_dark_color,
            SplaatSettings.ACCENT_VERY_LIGHT_COLOR: self.accent_very_light_color,
            SplaatSettings.ACCENT_VERY_DARK_COLOR: self.accent_very_dark_color,
        }
