from __future__ import annotations

import logging
import os.path
import re
import typing

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from splaat.db import PhoneInterval, Utterance, WordInterval
from splaat.desktop.models import TextFilterQuery
from splaat.desktop.settings import SplaatSettings
from splaat.utils import get_next_primary_key

pg.setConfigOption("imageAxisOrder", "row-major")  # best performance
pg.setConfigOptions(antialias=True)

logger = logging.getLogger("anchor")

if typing.TYPE_CHECKING:
    from splaat.desktop.models import CorpusModel, FileModel, FileSelectionModel


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strings = super().tickStrings(values, scale, spacing)
        strings = [x.replace("-", "") for x in strings]
        return strings


class AudioPlotItem(pg.PlotItem):
    def __init__(self, top_point, bottom_point):
        super().__init__(axisItems={"bottom": TimeAxis("bottom")})
        self.settings = SplaatSettings()
        self.plot_theme = self.settings.plot_theme
        self.setDefaultPadding(0)
        self.setClipToView(True)

        self.getAxis("bottom").setPen(self.plot_theme.break_line_color)
        self.getAxis("bottom").setTextPen(self.plot_theme.break_line_color)
        self.getAxis("bottom").setTickFont(self.settings.small_font)
        rect = QtCore.QRectF()
        rect.setTop(top_point)
        rect.setBottom(bottom_point)
        rect.setLeft(0)
        rect.setRight(10)
        rect = rect.normalized()
        self.setRange(rect=rect)
        self.hideAxis("left")
        self.setMouseEnabled(False, False)

        self.setMenuEnabled(False)
        self.hideButtons()


class SpeakerTierItem(pg.PlotItem):
    def __init__(self, top_point, bottom_point):
        super().__init__()
        self.settings = SplaatSettings()
        self.setDefaultPadding(0)
        self.setClipToView(True)
        self.hideAxis("left")
        self.hideAxis("bottom")
        rect = QtCore.QRectF()
        rect.setTop(top_point)
        rect.setBottom(bottom_point)
        rect.setLeft(0)
        rect.setRight(10)
        rect = rect.normalized()
        self.setRange(rect=rect)
        self.setMouseEnabled(False, False)

        self.setMenuEnabled(False)
        self.hideButtons()


class SplaatPlot(QtWidgets.QWidget):
    undoRequested = QtCore.Signal()
    redoRequested = QtCore.Signal()
    playRequested = QtCore.Signal()

    def __init__(self, *args):
        super().__init__(*args)
        self.settings = SplaatSettings()
        self.corpus_model: typing.Optional[CorpusModel] = None
        self.file_model: typing.Optional[FileModel] = None
        self.selection_model: typing.Optional[FileSelectionModel] = None
        layout = QtWidgets.QVBoxLayout()
        self.bottom_point = 0
        self.top_point = 8
        self.height = self.top_point - self.bottom_point
        self.separator_point = (self.height / 2) + self.bottom_point
        self.text_search_term = None
        self.phones_search_term = None

        # self.break_line.setZValue(30)
        self.audio_layout = pg.GraphicsLayoutWidget()
        self.audio_layout.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False
        )
        self.audio_layout.centralWidget.layout.setContentsMargins(0, 0, 0, 0)
        self.audio_layout.centralWidget.layout.setSpacing(0)
        self.plot_theme = self.settings.plot_theme
        self.audio_layout.setBackground(self.plot_theme.background_color)
        self.audio_plot = AudioPlots(2, 1, 0)
        self.audio_plot_item = AudioPlotItem(2, 0)
        self.audio_plot_item.addItem(self.audio_plot)
        # self.audio_plot.setZValue(0)
        self.audio_layout.addItem(self.audio_plot_item)

        self.speaker_tier_layout = pg.GraphicsLayoutWidget()
        self.speaker_tier_layout.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False
        )
        self.speaker_tier_layout.setAspectLocked(False)
        self.speaker_tier_layout.centralWidget.layout.setContentsMargins(0, 0, 0, 0)
        self.speaker_tier_layout.centralWidget.layout.setSpacing(0)
        self.speaker_tier_layout.setBackground(self.plot_theme.background_color)
        self.speaker_tiers: dict[SpeakerTier] = {}
        self.speaker_tier_items = {}
        self.tier_scroll_area = QtWidgets.QScrollArea()
        self.audio_scroll_area = QtWidgets.QScrollArea()
        self.audio_scroll_area.setContentsMargins(0, 0, 0, 0)
        self.tier_scroll_area.setWidget(self.speaker_tier_layout)
        self.tier_scroll_area.setWidgetResizable(True)
        self.tier_scroll_area.setContentsMargins(0, 0, 0, 0)
        self.tier_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.audio_scroll_area)
        scroll_layout.addWidget(self.audio_layout)
        self.audio_scroll_area.setLayout(scroll_layout)
        layout.addWidget(self.tier_scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll_layout.setSpacing(0)
        self.setLayout(layout)

    def set_search_term(self, text_search_term, phones_search_term):
        self.text_search_term = text_search_term
        self.phones_search_term = phones_search_term
        for v in self.speaker_tiers.values():
            v.set_search_term(text_search_term, phones_search_term)

    def set_models(
        self,
        corpus_model: CorpusModel,
        file_model: FileModel,
        selection_model: FileSelectionModel,
    ):
        self.corpus_model = corpus_model
        self.file_model = file_model
        self.selection_model = selection_model
        for t in self.speaker_tiers.values():
            t.set_models(corpus_model, selection_model)
        self.audio_plot.set_models(selection_model)
        self.selection_model.viewChanged.connect(self.update_plot)
        self.selection_model.resetView.connect(self.reset_plot)
        self.file_model.utterancesReady.connect(self.finalize_loading_utterances)
        self.selection_model.spectrogramReady.connect(self.finalize_loading_spectrogram)
        self.selection_model.waveformReady.connect(self.finalize_loading_auto_wave_form)
        self.file_model.selectionRequested.connect(self.finalize_loading_utterances)

    def refresh_theme(self):
        self.audio_layout.setBackground(self.plot_theme.background_color)
        self.speaker_tier_layout.setBackground(self.plot_theme.background_color)
        self.audio_plot.wave_form.update_theme()
        self.audio_plot.spectrogram.update_theme()
        # self.audio_plot.pitch_track.update_theme()
        # self.audio_plot_item.getAxis("bottom").setPen(self.plot_theme.break_line_color)
        # self.audio_plot_item.getAxis("bottom").setTextPen(self.plot_theme.break_line_color)

    def refresh(self):
        self.finalize_loading_utterances()
        self.finalize_loading_auto_wave_form()
        # self.finalize_loading_pitch_track()
        self.finalize_loading_spectrogram()

    def get_minimum_per_tier_height(self):
        fm = QtGui.QFontMetrics(self.settings.font)
        minimum_height = 25 + 10  # y margins for base box
        minimum_height += fm.height() * 3
        minimum_height += fm.height() * 2
        return minimum_height

    def finalize_loading_utterances(self):
        if self.file_model.file is None:
            return
        scroll_to = None
        self.speaker_tiers = {}
        self.speaker_tier_items = {}
        self.speaker_tier_layout.clear()
        height = self.speaker_tier_layout.height()
        pixel_size = (self.separator_point - self.bottom_point) / height
        speaker_tier_height = self.get_minimum_per_tier_height() * pixel_size
        speaker_name = "words"
        top_point = 0 * speaker_tier_height
        bottom_point = top_point - speaker_tier_height
        tier = SpeakerTier(
            top_point,
            bottom_point,
            speaker_name,
            self.corpus_model,
            self.file_model,
            self.selection_model,
        )
        tier.set_search_term(self.text_search_term, self.phones_search_term)
        tier.draggingLine.connect(self.audio_plot.update_drag_line)
        tier.lineDragFinished.connect(self.audio_plot.hide_drag_line)
        tier.receivedWheelEvent.connect(self.audio_plot.wheelEvent)
        tier.receivedGestureEvent.connect(self.audio_plot.gestureEvent)
        tier.setZValue(30)
        self.speaker_tiers[speaker_name] = tier
        for i, (key, tier) in enumerate(self.speaker_tiers.items()):
            tier.refresh()
            top_point = i * speaker_tier_height
            bottom_point = top_point - speaker_tier_height
            tier_item = SpeakerTierItem(top_point, bottom_point)
            tier_item.setRange(
                xRange=[self.selection_model.plot_min, self.selection_model.plot_max]
            )
            tier_item.addItem(tier)
            self.speaker_tier_items[key] = tier_item
            self.speaker_tier_layout.addItem(tier_item, i, 0)
        row_height = self.tier_scroll_area.height()
        if len(self.speaker_tiers) * self.get_minimum_per_tier_height() > height:
            self.tier_scroll_area.verticalScrollBar().setSingleStep(row_height)
            self.tier_scroll_area.verticalScrollBar().setPageStep(row_height)
            self.tier_scroll_area.verticalScrollBar().setMinimum(0)
            self.tier_scroll_area.verticalScrollBar().setMaximum(
                len(self.speaker_tiers) * row_height
            )
            self.tier_scroll_area.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
            )
            self.audio_layout.centralWidget.layout.setContentsMargins(
                0, 0, self.settings.scroll_bar_height, 0
            )
            if scroll_to is not None:
                self.tier_scroll_area.verticalScrollBar().setValue(
                    scroll_to * self.tier_scroll_area.height()
                )
        else:
            self.audio_layout.centralWidget.layout.setContentsMargins(0, 0, 0, 0)
            self.tier_scroll_area.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )

    def finalize_loading_spectrogram(self):
        self.audio_plot.spectrogram.hide()
        if self.selection_model.spectrogram is None:
            self.audio_plot.spectrogram.clear()
            return
        self.audio_plot.spectrogram.setData(
            self.selection_model.spectrogram,
            self.selection_model.selected_channel,
            self.selection_model.plot_min,
            self.selection_model.plot_max,
            self.selection_model.min_db,
            self.selection_model.max_db,
        )

    def finalize_loading_auto_wave_form(self):
        self.audio_plot.wave_form.hide()
        if self.selection_model.waveform_y is None:
            return
        self.audio_plot_item.setRange(
            xRange=[self.selection_model.plot_min, self.selection_model.plot_max]
        )
        self.audio_plot.update_plot()
        self.audio_plot.wave_form.setData(
            x=self.selection_model.waveform_x, y=self.selection_model.waveform_y
        )
        self.audio_plot.wave_form.show()

    def reset_text_grid(self):
        for tier in self.speaker_tiers.values():
            tier.reset_tier()

    def refresh_text_grid(self):
        for tier in self.speaker_tiers.values():
            tier.refresh(reset_bounds=True)

    def draw_text_grid(self):
        for i, (key, tier) in enumerate(self.speaker_tiers.items()):
            self.speaker_tier_items[key].hide()
            tier.refresh()
            self.speaker_tier_items[key].setRange(
                xRange=[self.selection_model.plot_min, self.selection_model.plot_max]
            )
            self.speaker_tier_items[key].show()

    def update_show_speakers(self, state):
        self.show_all_speakers = state > 0
        self.update_plot()

    def reset_plot(self, *args):
        self.reset_text_grid()
        self.audio_plot.wave_form.clear()
        # self.audio_plot.pitch_track.clear()
        self.audio_plot.spectrogram.clear()

    def update_plot(self, *args):
        if self.corpus_model.rowCount() == 0:
            return
        if self.file_model.file is None or self.selection_model.min_time is None:
            return
        self.audio_plot.update_plot()
        self.draw_text_grid()


class WordLine(pg.InfiniteLine):
    hoverChanged = QtCore.Signal(object)
    snapModeChanged = QtCore.Signal(object)

    def __init__(
        self, *args, movingPen=None, view_min=None, view_max=None, initial=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.movingPen = movingPen
        self.initial = initial
        self.view_min = view_min
        self.view_max = view_max
        self.bounding_width = 0.1
        self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)

    def hoverEvent(self, ev):
        if (
            (not ev.isExit())
            and self.movable
            and (
                (self.initial and self.pos().x() - self.mapToParent(ev.pos()).x() < 0)
                or (not self.initial and self.pos().x() - self.mapToParent(ev.pos()).x() > 0)
            )
            and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        ):
            self.setMouseHover(True)
            self._boundingRect = None
            self.hoverChanged.emit(True)
        else:
            self.setMouseHover(False)
            self.hoverChanged.emit(False)
            self._boundingRect = None

    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.snapModeChanged.emit(ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
            if ev.isStart() and (
                (self.initial and self.pos().x() - self.mapToParent(ev.buttonDownPos()).x() < 0)
                or (
                    not self.initial
                    and self.pos().x() - self.mapToParent(ev.buttonDownPos()).x() > 0
                )
            ):
                self.moving = True
                self._boundingRect = None
                self.currentPen = self.movingPen
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()

            if not self.moving:
                return
            p = self.cursorOffset + self.mapToParent(ev.pos())
            p.setY(self.startPosition.y())
            if p.x() > self.view_max:
                p.setX(self.view_max)
            if p.x() < self.view_min:
                p.setX(self.view_min)
            self.setPos(p)
            self.sigDragged.emit(self)
            if ev.isFinish():
                self.currentPen = self.pen
                self._boundingRect = None
                self._bounds = None
                self._lastViewSize = None
                self.moving = False
                self.sigPositionChangeFinished.emit(self)
                self.update()

    def _computeBoundingRect(self):
        # br = UIGraphicsItem.boundingRect(self)
        vr = self.viewRect()  # bounds of containing ViewBox mapped to local coords.
        if vr is None:
            return QtCore.QRectF()

        # add a 4-pixel radius around the line for mouse interaction.
        px = self.pixelLength(
            direction=pg.Point(1, 0), ortho=True
        )  # get pixel length orthogonal to the line
        if px is None:
            px = 0
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = max(self.bounding_width, self._maxMarkerSize + pw) + 1
        w = w * px
        br = QtCore.QRectF(vr)
        if self.initial:
            br.setBottom(-w)
            br.setTop(0)
        else:
            br.setTop(w)
            br.setBottom(0)

        if not self.moving:
            left = self.span[0]
            right = self.span[1]
        else:
            length = br.width()
            left = br.left()
            right = br.left() + length

        br.setLeft(left)
        br.setRight(right)
        br = br.normalized()

        vs = self.getViewBox().size()

        if self._bounds != br or self._lastViewSize != vs:
            self._bounds = br
            self._lastViewSize = vs
            self.prepareGeometryChange()

        self._endPoints = (left, right)
        self._lastViewRect = vr

        return self._bounds


class SplaatRegion(pg.LinearRegionItem):
    selectRequested = QtCore.Signal(object, object, object)
    audioSelected = QtCore.Signal(object, object)
    viewRequested = QtCore.Signal(object, object)

    def __init__(
        self,
        item: Utterance,
        corpus_model: CorpusModel,
        file_model: FileModel,
        selection_model: FileSelectionModel,
        bottom_point: float = 0,
        top_point: float = 1,
    ):
        pg.GraphicsObject.__init__(self)
        self.item = item

        self.settings = SplaatSettings()
        self.plot_theme = self.settings.plot_theme

        self.item_min = self.item.start
        self.item_max = self.item.end
        if selection_model.settings.right_to_left:
            self.item_min, self.item_max = -self.item_max, -self.item_min
        self.corpus_model = corpus_model
        self.file_model = file_model
        self.selection_model = selection_model
        self.bottom_point = bottom_point
        self.top_point = top_point
        self.span = (self.bottom_point, self.top_point)
        self.text_margin_pixels = 2
        self.height = abs(self.top_point - self.bottom_point)

        self.interval_background_color = self.plot_theme.interval_background_color
        self.hover_line_color = self.plot_theme.hover_line_color
        self.moving_line_color = self.plot_theme.moving_line_color

        self.break_line_color = self.plot_theme.break_line_color
        self.text_color = self.plot_theme.text_color
        self.selected_interval_color = self.plot_theme.selected_interval_color
        self.plot_text_font = self.settings.big_font
        self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        self.pen = pg.mkPen(self.break_line_color, width=3)
        self.pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        self.border_pen = pg.mkPen(self.break_line_color, width=2)
        self.border_pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)

        if self.selection_model.checkSelected(getattr(self.item, "id", None)):
            self.background_brush = pg.mkBrush(self.selected_interval_color)
        else:
            # self.interval_background_color.setAlpha(0)
            self.background_brush = pg.mkBrush(self.interval_background_color)

        self.hoverPen = pg.mkPen(self.hover_line_color, width=3)
        self.movingPen = pg.mkPen(
            self.moving_line_color, width=3, style=QtCore.Qt.PenStyle.DashLine
        )
        self.orientation = "vertical"
        self.bounds = QtCore.QRectF()
        self.blockLineSignal = False
        self.moving = False
        self.mouseHovering = False
        self.swapMode = "sort"
        self.clipItem = None
        self._boundingRectCache = None
        self.movable = False
        self.cached_visible_duration = None
        self.cached_view = None
        self.currentBrush = self.background_brush
        self.picture = QtGui.QPicture()
        self.rect = QtCore.QRectF(
            left=self.item_min,
            top=self.top_point,
            width=self.item_max - self.item_min,
            height=self.height,
        )
        self.rect.setTop(self.top_point)
        self.rect.setLeft(self.item_min)
        self.rect.setRight(self.item_max)
        self.rect.setBottom(self.bottom_point)
        self._generate_picture()
        self.sigRegionChanged.connect(self.update_bounds)
        self.sigRegionChangeFinished.connect(self.update_bounds)
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def update_bounds(self):
        beg, end = self.getRegion()
        self.rect.setLeft(beg)
        self.rect.setRight(end)
        self._generate_picture()

    def _generate_picture(self):
        if self.selection_model is None:
            return
        painter = QtGui.QPainter(self.picture)
        painter.setPen(self.border_pen)
        painter.setBrush(self.currentBrush)
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def mouseClickEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        self.audioSelected.emit(self.item_min, self.item_max)
        ev.accept()

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        self.audioSelected.emit(self.item_min, self.item_max)
        padding = (self.item_max - self.item_min) / 2
        self.viewRequested.emit(self.item_min - padding, self.item_max + padding)
        ev.accept()

    def setSelected(self, selected: bool):
        if selected:
            new_brush = pg.mkBrush(self.selected_interval_color)
        else:
            new_brush = pg.mkBrush(self.interval_background_color)
        if new_brush != self.currentBrush:
            self.currentBrush = new_brush
            self._generate_picture()
        self.update()

    def setMouseHover(self, hover: bool):
        # Inform the item that the mouse is(not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        self.popup(hover)
        self.update()

    def select_self(self, deselect=False, reset=True):
        self.selected = True
        if self.selected and not deselect and not reset:
            return

    def boundingRect(self):
        try:
            visible_begin = max(self.item_min, self.selection_model.plot_min)
            visible_end = min(self.item_max, self.selection_model.plot_max)
        except TypeError:
            visible_begin = self.item_min
            visible_end = self.item_max
        br = QtCore.QRectF(self.picture.boundingRect())
        br.setLeft(visible_begin)
        br.setRight(visible_end)

        br.setTop(self.top_point)
        br.setBottom(self.bottom_point + 0.01)
        br = br.normalized()

        if self._boundingRectCache != br:
            self._boundingRectCache = br
            self.prepareGeometryChange()
        return br


class IntervalLine(pg.InfiniteLine):
    hoverChanged = QtCore.Signal(object, object)

    def __init__(
        self,
        pos,
        index=None,
        index_count=None,
        pen=None,
        movingPen=None,
        hoverPen=None,
        bottom_point: float = 0,
        top_point: float = 1,
        bound_min=None,
        bound_max=None,
        movable=True,
    ):
        super().__init__(
            pos,
            angle=90,
            span=(bottom_point, top_point),
            pen=pen,
            hoverPen=hoverPen,
            movable=movable,
        )
        self.index = index
        self.index_count = index_count
        self.initial = index <= 0
        self.final = index >= index_count - 1
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.movingPen = movingPen
        self.bounding_width = 0.1
        if self.movable:
            self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)

    def setMouseHover(self, hover):
        if hover and self.movable:
            self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        elif self.movable:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.hoverChanged.emit(hover, self)
        super().setMouseHover(hover)

    def _computeBoundingRect(self):
        # br = UIGraphicsItem.boundingRect(self)
        vr = self.viewRect()  # bounds of containing ViewBox mapped to local coords.
        if vr is None:
            return QtCore.QRectF()

        # add a 4-pixel radius around the line for mouse interaction.
        px = self.pixelLength(
            direction=pg.Point(1, 0), ortho=True
        )  # get pixel length orthogonal to the line
        if px is None:
            px = 0
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = max(self.bounding_width, self._maxMarkerSize + pw) + 5
        w = w * px
        br = QtCore.QRectF(vr)
        br.setBottom(-w)
        br.setTop(w)

        left = self.span[0]
        right = self.span[1]

        br.setLeft(left)
        br.setRight(right)
        br = br.normalized()

        vs = self.getViewBox().size()

        if self._bounds != br or self._lastViewSize != vs:
            self._bounds = br
            self._lastViewSize = vs
            self.prepareGeometryChange()

        self._endPoints = (left, right)
        self._lastViewRect = vr

        return self._bounds

    def hoverEvent(self, ev):
        if (
            (not ev.isExit())
            and self.movable
            # and (
            #    (self.initial and self.pos().x() - self.mapToParent(ev.pos()).x() < 0)
            #    or (not self.initial and self.pos().x() - self.mapToParent(ev.pos()).x() > 0)
            # )
            and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        ):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self._boundingRect = None
                self.currentPen = self.movingPen
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()

            if not self.moving:
                return
            p = self.cursorOffset + self.mapToParent(ev.pos())
            p.setY(self.startPosition.y())
            if p.x() >= self.bound_max - 0.01:
                p.setX(self.bound_max - 0.01)
            if p.x() <= self.bound_min + 0.01:
                p.setX(self.bound_min + 0.01)
            self.setPos(p)
            self.sigDragged.emit(self)
            if ev.isFinish():
                self.currentPen = self.pen
                self.moving = False
                self.sigPositionChangeFinished.emit(self)


class IntervalTier(pg.GraphicsObject):
    highlightRequested = QtCore.Signal(object)

    def __init__(
        self,
        parent: UtteranceRegion,
        utterance: Utterance,
        intervals: typing.List[PhoneInterval],
        selection_model: FileSelectionModel,
        top_point: float,
        bottom_point: float,
        movable: bool = False,
        lookup="phone_intervals",
    ):
        super().__init__()
        self.setParentItem(parent)
        self.intervals = intervals
        self.settings = SplaatSettings()
        self.plot_theme = self.settings.plot_theme
        self.anchor = pg.Point((0.5, 0.5))
        self.plot_text_font = self.settings.font
        self.movable = movable

        self.background_color = self.plot_theme.background_color
        self.hover_line_color = self.plot_theme.hover_line_color
        self.moving_line_color = self.plot_theme.moving_line_color
        self.break_line_color = self.plot_theme.break_line_color
        self.text_color = self.plot_theme.text_color
        self.selected_interval_color = self.plot_theme.selected_interval_color
        self.highlight_interval_color = self.plot_theme.selected_interval_color
        self.highlight_text_color = self.plot_theme.background_color

        self.top_point = top_point
        self.bottom_point = bottom_point
        self.selection_model = selection_model
        self.utterance = utterance
        self.lookup = lookup
        self.lines = []
        self.selected = None

        self._boundingRectCache = None
        self._cached_pixel_size = None

        self.hoverPen = pg.mkPen(self.hover_line_color, width=3)
        self.movingPen = pg.mkPen(
            self.moving_line_color, width=3, style=QtCore.Qt.PenStyle.DashLine
        )
        self.border_pen = pg.mkPen(self.break_line_color, width=3)
        self.border_pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        self.text_pen = pg.mkPen(self.text_color)
        self.text_brush = pg.mkBrush(self.text_color)
        self.highlight_text_pen = pg.mkPen(self.plot_theme.error_color)
        self.highlight_text_brush = pg.mkBrush(self.plot_theme.error_color)
        self.search_term = None
        self.search_regex = None
        self.update_intervals(self.utterance)

    def refresh_boundaries(self, interval_id, new_time):
        for index, interval in enumerate(self.intervals):
            if interval.id == interval_id:
                try:
                    self.lines[index].setPos(new_time)
                except IndexError:
                    pass
                break
        self.refresh_tier()

    def update_intervals(self, utterance):
        self.intervals = sorted(
            getattr(utterance, self.lookup, self.intervals), key=lambda x: x.start
        )
        for line in self.lines:
            if line.scene() is not None:
                line.scene().removeItem(line)
        self.lines = []
        bound_min = self.utterance.start
        for i, interval in enumerate(self.intervals):
            if i == 0:
                continue

            line = IntervalLine(
                interval.start,
                index=i - 1,
                index_count=len(self.intervals) - 1,
                bound_min=bound_min,
                bound_max=interval.end,
                bottom_point=self.bottom_point,
                top_point=self.top_point,
                pen=self.border_pen,
                movingPen=self.movingPen,
                hoverPen=self.hoverPen,
                movable=self.movable,
            )
            line.setZValue(30)
            line.setParentItem(self)
            # line.sigPositionChanged.connect(self._lineMoved)
            self.lines.append(line)
            bound_min = interval.start
        self.refresh_tier()

    def refresh_tier(self):
        self.regenerate_text_boxes()
        self.update()

    def regenerate_text_boxes(self):
        self.array = pg.Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        self.selected_array = pg.Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        self.array.resize(len(self.intervals))
        memory = self.array.ndarray()

        fm = QtGui.QFontMetrics(self.plot_text_font)
        for i, interval in enumerate(self.intervals):
            memory[i, 0] = interval.start
            memory[i, 2] = interval.end - interval.start
            if interval.label not in self.parentItem().painter_path_cache:
                symbol = QtGui.QPainterPath()

                symbol.addText(0, 0, self.plot_text_font, interval.label)
                br = symbol.boundingRect()

                # getting transform object
                tr = QtGui.QTransform()

                # translating
                tr.translate(-br.x() - br.width() / 2.0, fm.height() / 2.0)
                self.parentItem().painter_path_cache[interval.label] = tr.map(symbol)

        memory[:, 1] = self.bottom_point
        memory[:, 3] = self.top_point - self.bottom_point

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            if any(line.mouseHovering for line in self.lines):
                e.ignore()
                return
            time = e.pos().x()

            margin = 21 * self._cached_pixel_size[0]
            if time <= self.utterance.start + margin or time >= self.utterance.end - margin:
                e.ignore()
                return
            memory = self.array.ndarray()
            if memory.shape[0] > 0:
                index = np.searchsorted(memory[:, 0], time) - 1
                interval = self.intervals[index]
                self.selection_model.select_audio(interval.start, interval.end)
                if self.selected == interval:
                    self.selected = None
                self.selected = interval
                self.highlightRequested.emit(self.selected)
                self.update()
                e.accept()
                return

        return super().mousePressEvent(e)

    def reset_selection(self, obj):
        self.selected = None
        if hasattr(obj, "start"):
            for interval in self.intervals:
                if interval.start > obj.start:
                    break
                if interval.start == obj.start and interval.end == obj.end:
                    self.selected = interval
        self.update()

    def set_search_term(self, search_term: typing.Optional[TextFilterQuery]):
        self.search_term = search_term
        self.search_regex = None
        if self.search_term is not None and self.search_term.text:
            self.search_regex = re.compile(self.search_term.generate_expression())

    def paint(self, painter, *args):
        vb = self.getViewBox()
        px = vb.viewPixelSize()
        inst = self.array.instances()
        br = self.boundingRect()
        painter.save()
        painter.setPen(self.border_pen)
        painter.drawRect(br)
        painter.restore()
        total_time = self.selection_model.max_time - self.selection_model.min_time
        if self.selected:
            painter.save()
            painter.setPen(self.border_pen)
            painter.setBrush(pg.mkBrush(self.selected_interval_color))
            selected_rect = QtCore.QRectF(
                self.selected.start,
                self.bottom_point,
                self.selected.end - self.selected.start,
                abs(self.top_point - self.bottom_point),
            )
            painter.drawRect(selected_rect)
            painter.restore()
        for i, interval in enumerate(self.intervals):
            r = inst[i]
            visible_begin = max(r.left(), self.selection_model.plot_min)
            visible_end = min(r.right(), self.selection_model.plot_max)
            visible_duration = visible_end - visible_begin
            if visible_duration / total_time <= 0.0075:
                continue
            x = (r.left() + r.right()) / 2
            painter.save()
            options = QtGui.QTextOption()
            options.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            painter.setRenderHint(painter.RenderHint.Antialiasing, True)
            text_pen = self.text_pen
            text_brush = self.text_brush
            if self.search_regex is not None and self.search_regex.search(interval.label):
                text_pen = self.highlight_text_pen
                text_brush = self.highlight_text_brush
            painter.setPen(text_pen)
            painter.setBrush(text_brush)
            painter.translate(x, (self.top_point + self.bottom_point) / 2)
            path = self.parentItem().painter_path_cache[interval.label]
            painter.scale(px[0], -px[1])
            painter.drawPath(path)
            painter.restore()

    def boundingRect(self):
        br = QtCore.QRectF(
            self.utterance.start,
            self.bottom_point,
            self.utterance.end - self.utterance.start,
            abs(self.top_point - self.bottom_point),
        )
        vb = self.getViewBox()
        self._cached_pixel_size = vb.viewPixelSize()
        if self._boundingRectCache != br:
            self._boundingRectCache = br
            self.prepareGeometryChange()
        return br


class PhoneIntervalTier(IntervalTier):
    draggingLine = QtCore.Signal(object)
    lineDragFinished = QtCore.Signal(object)
    phoneBoundaryChanged = QtCore.Signal(object, object, object)
    phoneIntervalChanged = QtCore.Signal(object, object)
    phoneIntervalInserted = QtCore.Signal(object, object, object, object, object)
    phoneIntervalDeleted = QtCore.Signal(object, object, object, object)
    deleteReferenceAlignments = QtCore.Signal()

    def __init__(
        self,
        parent,
        utterance: Utterance,
        intervals: PhoneInterval,
        selection_model: FileSelectionModel,
        top_point,
        bottom_point,
    ):
        super().__init__(
            parent,
            utterance,
            intervals,
            selection_model,
            top_point,
            bottom_point,
            movable=True,
        )

    def update_intervals(self, utterance):
        self.intervals = sorted(
            getattr(utterance, "phone_intervals", self.intervals), key=lambda x: x.start
        )
        for line in self.lines:
            if line.scene() is not None:
                line.scene().removeItem(line)
        self.lines = []
        bound_min = self.utterance.start
        for i, interval in enumerate(self.intervals):
            if i == 0:
                continue

            line = IntervalLine(
                interval.start,
                index=i - 1,
                index_count=len(self.intervals) - 1,
                bound_min=bound_min,
                bound_max=interval.end,
                bottom_point=self.bottom_point,
                top_point=self.top_point,
                pen=self.border_pen,
                movingPen=self.movingPen,
                hoverPen=self.hoverPen,
                movable=self.movable,
            )
            line.setZValue(30)
            line.setParentItem(self)
            line.sigPositionChangeFinished.connect(self.lineMoveFinished)
            # line.sigPositionChanged.connect(self._lineMoved)
            self.lines.append(line)
            bound_min = interval.start
            line.sigPositionChanged.connect(self.draggingLine.emit)
            line.sigPositionChangeFinished.connect(self.lineDragFinished.emit)
            line.hoverChanged.connect(self.update_hover)
        self.refresh_tier()

    def update_hover(self, hovered, time):
        if hovered:
            self.draggingLine.emit(time)
        else:
            self.lineDragFinished.emit(time)

    def lineMoveFinished(self):
        sender: IntervalLine = self.sender()
        self.phoneBoundaryChanged.emit(
            self.intervals[sender.index], self.intervals[sender.index + 1], sender.pos().x()
        )
        if sender.index != 0:
            self.lines[sender.index - 1].bound_max = sender.pos().x()
        if sender.index != len(self.lines) - 1:
            self.lines[sender.index + 1].bound_min = sender.pos().x()
        self.regenerate_text_boxes()
        self.update()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.RightButton:
            time = e.pos().x()
            memory = self.array.ndarray()
            if memory.shape[0] > 0:
                index = np.searchsorted(memory[:, 0], time) - 1
                interval = self.intervals[index]
                self.selection_model.select_audio(interval.start, interval.end)
                initial = (time - interval.start) < (interval.end - time)
                menu = self.construct_context_menu(index, interval, initial)
                menu.exec_(e.screenPos())
                e.accept()
                return

        return super().mousePressEvent(e)

    def update_phone(self, phone_interval, phone):
        self.phoneIntervalChanged.emit(phone_interval, phone)

    def insert_phone_interval(self, index: int, initial: bool):
        previous_interval = None
        following_interval = None
        if initial:
            following_interval = self.intervals[index]
            word_interval_id = following_interval.word_interval_id
            if index > 0:
                previous_interval = self.intervals[index - 1]
            begin = following_interval.start
            end = (following_interval.start + following_interval.end) / 2
        else:
            previous_interval = self.intervals[index]
            word_interval_id = previous_interval.word_interval_id
            if index < len(self.intervals) - 1:
                following_interval = self.intervals[index + 1]
            begin = (previous_interval.start + previous_interval.end) / 2
            end = previous_interval.end
        self.phoneIntervalInserted.emit(
            previous_interval, following_interval, word_interval_id, begin, end
        )

    def insert_silence_interval(self, index: int, initial: bool):
        previous_interval = None
        following_interval = None
        if initial:
            following_interval = self.intervals[index]
            if index > 0:
                previous_interval = self.intervals[index - 1]
            begin = following_interval.start
            end = (following_interval.start + following_interval.end) / 2
        else:
            previous_interval = self.intervals[index]
            if index < len(self.intervals) - 1:
                following_interval = self.intervals[index + 1]
            begin = (previous_interval.start + previous_interval.end) / 2
            end = previous_interval.end
        self.phoneIntervalInserted.emit(previous_interval, following_interval, None, begin, end)

    def delete_phone_interval(self, index: int, initial: bool):
        previous_interval = None
        interval = self.intervals[index]
        following_interval = None
        if index > 0:
            previous_interval = self.intervals[index - 1]
        if index < len(self.intervals) - 1:
            following_interval = self.intervals[index + 1]
        if initial:
            time_point = interval.end
        else:
            time_point = interval.start
        self.phoneIntervalDeleted.emit(interval, previous_interval, following_interval, time_point)

    def delete_reference(self):
        self.deleteReferenceAlignments.emit()

    def construct_context_menu(
        self,
        index,
        phone_interval: PhoneInterval,
        initial=True,
    ):
        menu = QtWidgets.QMenu()
        change_phone_menu = QtWidgets.QMenu("Change phone")
        for phone_label in sorted(self.parentItem().corpus_model.phones):
            if phone_label == phone_interval.label:
                continue
            a = QtGui.QAction(menu)
            a.setText(phone_label)
            a.triggered.connect(
                lambda triggered, x=phone_interval, y=phone_label: self.update_phone(x, y)
            )
            change_phone_menu.addAction(a)
        menu.addMenu(change_phone_menu)
        a = QtGui.QAction(menu)

        a = QtGui.QAction(menu)
        a.setText("Insert silence/interval")
        a.triggered.connect(lambda triggered, x=index, y=initial: self.insert_phone_interval(x, y))
        menu.addAction(a)

        a = QtGui.QAction(menu)
        a.setText("Delete interval")
        a.triggered.connect(lambda triggered, x=index, y=initial: self.delete_phone_interval(x, y))
        menu.addAction(a)
        return menu


class WordIntervalTier(IntervalTier):
    def __init__(
        self,
        parent,
        utterance: Utterance,
        intervals: typing.List[WordInterval],
        selection_model: FileSelectionModel,
        top_point,
        bottom_point,
    ):
        super().__init__(
            parent,
            utterance,
            intervals,
            selection_model,
            top_point,
            bottom_point,
            lookup="word_intervals",
        )


class UtteranceRegion(SplaatRegion):
    phoneBoundaryChanged = QtCore.Signal(object, object, object, object)
    phoneIntervalChanged = QtCore.Signal(object, object, object)
    wordChanged = QtCore.Signal(object, object, object)
    phoneIntervalInserted = QtCore.Signal(object, object, object, object, object)
    phoneIntervalDeleted = QtCore.Signal(object, object, object, object, object)
    createWord = QtCore.Signal(object)
    draggingLine = QtCore.Signal(object)
    lineDragFinished = QtCore.Signal(object)
    wordBoundariesChanged = QtCore.Signal(object, object)
    phoneTiersChanged = QtCore.Signal(object)

    def __init__(
        self,
        parent,
        utterance: Utterance,
        corpus_model: CorpusModel,
        file_model: FileModel,
        selection_model: FileSelectionModel,
        bottom_point: float = 0,
        top_point: float = 1,
    ):
        super().__init__(
            utterance,
            corpus_model,
            file_model,
            selection_model,
            bottom_point,
            top_point,
        )
        self.setParentItem(parent)
        self.hide()
        self.item = utterance
        self.selection_model = selection_model
        self.num_tiers = 2
        self.per_tier_range = (top_point - bottom_point) / self.num_tiers
        self.selected = self.selection_model.checkSelected(self.item.id)

        self.border_pen = pg.mkPen(self.break_line_color, width=3)
        self.border_pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        self.text_pen = pg.mkPen(self.text_color)
        self.text_brush = pg.mkBrush(self.text_color)

        # note LinearRegionItem.Horizontal and LinearRegionItem.Vertical
        # are kept for backward compatibility.
        lineKwds = dict(
            movable=True,
            bounds=None,
            span=self.span,
            pen=self.pen,
            hoverPen=self.hoverPen,
            movingPen=self.movingPen,
        )
        self.lines = [
            WordLine(
                QtCore.QPointF(self.item_min, 0),
                angle=90,
                initial=True,
                view_min=self.selection_model.plot_min,
                view_max=self.selection_model.plot_max,
                **lineKwds,
            ),
            WordLine(
                QtCore.QPointF(self.item_max, 0),
                angle=90,
                initial=False,
                view_min=self.selection_model.plot_min,
                view_max=self.selection_model.plot_max,
                **lineKwds,
            ),
        ]
        self.snap_mode = False
        self.initial_line_moving = False
        for line in self.lines:
            line.setZValue(30)
            line.setParentItem(self)
            line.sigPositionChangeFinished.connect(self.lineMoveFinished)
            line.hoverChanged.connect(self.popup)
            line.sigPositionChanged.connect(self.draggingLine.emit)
            line.sigPositionChangeFinished.connect(self.lineDragFinished.emit)
            line.snapModeChanged.connect(self.update_snap_mode)
        self.lines[0].sigPositionChanged.connect(self._line0Moved)
        self.lines[1].sigPositionChanged.connect(self._line1Moved)

        self._painter_path_cache = {}
        self._cached_pixel_size = None
        self.normalized_text = None
        self.transcription_text = None
        self.file_model.phoneTierChanged.connect(self.update_phone_tiers)
        i = -1
        tier_top_point = self.top_point - ((i + 1) * self.per_tier_range)
        tier_bottom_point = tier_top_point - self.per_tier_range
        self.word_interval_tier = WordIntervalTier(
            self,
            self.item,
            utterance.word_intervals,
            self.selection_model,
            top_point=tier_top_point,
            bottom_point=tier_bottom_point,
        )
        self.wordBoundariesChanged.connect(self.word_interval_tier.refresh_boundaries)
        self.phoneTiersChanged.connect(self.word_interval_tier.update_intervals)
        i = 0
        tier_top_point = self.top_point - ((i + 1) * self.per_tier_range)
        tier_bottom_point = tier_top_point - self.per_tier_range
        self.phone_interval_tier = PhoneIntervalTier(
            self,
            self.item,
            utterance.phone_intervals,
            self.selection_model,
            top_point=tier_top_point,
            bottom_point=tier_bottom_point,
        )
        self.phone_interval_tier.highlightRequested.connect(
            self.word_interval_tier.reset_selection
        )
        self.word_interval_tier.highlightRequested.connect(
            self.phone_interval_tier.reset_selection
        )
        self.phone_interval_tier.draggingLine.connect(self.draggingLine.emit)
        self.phone_interval_tier.lineDragFinished.connect(self.lineDragFinished.emit)
        self.phone_interval_tier.phoneBoundaryChanged.connect(self.change_phone_boundaries)
        self.phone_interval_tier.phoneIntervalChanged.connect(self.change_phone_interval)
        self.phone_interval_tier.phoneIntervalDeleted.connect(self.delete_phone_interval)
        self.phone_interval_tier.phoneIntervalInserted.connect(self.insert_phone_interval)
        self.phoneTiersChanged.connect(self.phone_interval_tier.update_intervals)

        self.selection_model.viewChanged.connect(self.update_view_times)
        self.selection_model.selectionAudioChanged.connect(self.word_interval_tier.reset_selection)
        self.selection_model.selectionAudioChanged.connect(
            self.phone_interval_tier.reset_selection
        )
        self.show()

    def set_search_term(self, text_search_term, phones_search_term):
        self.word_interval_tier.set_search_term(text_search_term)
        self.phone_interval_tier.set_search_term(phones_search_term)

    def update_snap_mode(self, snap_mode):
        self.snap_mode = snap_mode

    def _line0Moved(self):
        self.lineMoved(0)
        self.initial_line_moving = True

    def _line1Moved(self):
        self.lineMoved(1)
        self.initial_line_moving = False

    def update_phone_tiers(self, utterance):
        if utterance.id != self.item.id:
            return
        self.phoneTiersChanged.emit(utterance)

    @property
    def painter_path_cache(self):
        if self.parentItem() is None:
            return self._painter_path_cache
        return self.parentItem().painter_path_cache

    def change_editing(self, editable: bool):
        self.lines[0].movable = editable
        self.lines[1].movable = editable

    def popup(self, hover: bool):
        if hover or self.moving or self.lines[0].moving or self.lines[1].moving:
            self.setZValue(30)
        else:
            self.setZValue(0)

    def setMovable(self, m=True):
        """Set lines to be movable by the user, or not. If lines are movable, they will
        also accept HoverEvents."""
        for line in self.lines:
            line.setMovable(m)
        self.movable = False
        self.setAcceptHoverEvents(False)

    def update_view_times(self):
        self.lines[0].view_min = self.selection_model.plot_min
        self.lines[0].view_max = self.selection_model.plot_max
        self.lines[1].view_min = self.selection_model.plot_min
        self.lines[1].view_max = self.selection_model.plot_max
        self.update()

    def boundingRect(self):
        br = QtCore.QRectF(self.viewRect())  # bounds of containing ViewBox mapped to local coords.
        vb = self.getViewBox()
        self._cached_pixel_size = vb.viewPixelSize()
        rng = self.getRegion()

        br.setLeft(rng[0])
        br.setRight(rng[1])

        br.setTop(self.top_point)
        br.setBottom(self.bottom_point)

        x_margin_px = 40
        self.size_calculated = True
        for line in self.lines:
            line.bounding_width = int(x_margin_px / 2)
        br = br.normalized()

        if self._boundingRectCache != br:
            self._boundingRectCache = br
            self.prepareGeometryChange()

        return br

    def change_phone_boundaries(
        self, first_phone_interval, second_phone_interval, new_time: float
    ):
        self.phoneBoundaryChanged.emit(
            self.item, first_phone_interval, second_phone_interval, new_time
        )
        if first_phone_interval.word_interval_id != second_phone_interval.word_interval_id:
            self.wordBoundariesChanged.emit(first_phone_interval.word_interval_id, new_time)

    def change_phone_interval(self, phone_interval, new_phone_id):
        self.phoneIntervalChanged.emit(self.item, phone_interval, new_phone_id)

    def delete_phone_interval(self, interval, previous_interval, following_interval, time_point):
        self.phoneIntervalDeleted.emit(
            self.item, interval, previous_interval, following_interval, time_point
        )
        if (
            previous_interval is None
            or following_interval is None
            or previous_interval.word_interval_id != following_interval.word_interval_id
        ):
            self.wordBoundariesChanged.emit(previous_interval.word_interval_id, time_point)

    def insert_phone_interval(
        self,
        previous_interval,
        following_interval,
        word_interval_id,
        start=None,
        end=None,
    ):
        if start is None:
            start = (
                (previous_interval.start + previous_interval.end) / 2
                if previous_interval is not None
                else self.item.start
            )
        if end is None:
            end = (
                (following_interval.start + following_interval.end) / 2
                if following_interval is not None
                else self.item.end
            )
        inserting_word_interval = word_interval_id is None
        word_interval = None
        if inserting_word_interval:
            previous_word_interval_id = (
                previous_interval.word_interval_id if previous_interval is not None else None
            )
            following_word_interval_id = (
                following_interval.word_interval_id if following_interval is not None else None
            )
            at_word_boundary = previous_word_interval_id != following_word_interval_id
            if not at_word_boundary:
                return
            word_interval_id = get_next_primary_key(self.corpus_model.session, WordInterval)
            word = "<eps>"
            word_interval = WordInterval(
                id=word_interval_id,
                word=word,
                start=start,
                end=end,
            )
        else:
            for x in self.item.word_intervals:
                if x.id == word_interval_id:
                    word_interval = x
                    break
        next_pk = get_next_primary_key(self.corpus_model.session, PhoneInterval)
        phone_interval = PhoneInterval(
            id=next_pk,
            phone="sil",
            start=start,
            end=end,
            word_interval=word_interval,
            word_interval_id=word_interval_id,
        )
        self.phoneIntervalInserted.emit(
            self.item,
            phone_interval,
            previous_interval,
            following_interval,
            word_interval if inserting_word_interval else None,
        )


class WaveForm(pg.PlotCurveItem):
    def __init__(self, bottom_point, top_point):
        self.settings = SplaatSettings()
        self.top_point = top_point
        self.bottom_point = bottom_point
        self.mid_point = (self.top_point + self.bottom_point) / 2
        pen = pg.mkPen(self.settings.plot_theme.wave_line_color, width=1)
        super(WaveForm, self).__init__()
        self.setPen(pen)
        self.channel = 0
        self.y = None
        self.selection_model = None
        self.setAcceptHoverEvents(False)

    def update_theme(self):
        pen = pg.mkPen(self.settings.plot_theme.wave_line_color, width=1)
        self.setPen(pen)

    def hoverEvent(self, ev):
        return

    def set_models(self, selection_model: FileSelectionModel):
        self.selection_model = selection_model


class PitchTrack(pg.PlotCurveItem):
    def __init__(self, bottom_point, top_point):
        self.settings = SplaatSettings()
        self.plot_theme = self.settings.plot_theme
        self.top_point = top_point
        self.bottom_point = bottom_point
        self.mid_point = (self.top_point + self.bottom_point) / 2
        pen = pg.mkPen(self.plot_theme.pitch_color, width=3)
        super().__init__()
        self.setPen(pen)
        self.channel = 0
        self.y = None
        self.selection_model = None
        self.setAcceptHoverEvents(False)
        self.min_label = pg.TextItem(
            str(self.settings.PITCH_MIN_F0),
            self.plot_theme.pitch_color,
            anchor=(1, 1),
        )
        self.min_label.setFont(self.settings.font)
        self.min_label.setParentItem(self)
        self.max_label = pg.TextItem(
            str(self.settings.PITCH_MAX_F0),
            self.plot_theme.pitch_color,
            anchor=(1, 0),
        )
        self.max_label.setFont(self.settings.font)
        self.max_label.setParentItem(self)

    def update_theme(self):
        pen = pg.mkPen(self.plot_theme.pitch_color, width=3)
        self.setPen(pen)
        self.min_label.setColor(self.plot_theme.pitch_color)
        self.max_label.setColor(self.plot_theme.pitch_color)

    def hoverEvent(self, ev):
        return

    def set_range(self, min_f0, max_f0, end):
        self.min_label.setText(f"{min_f0} Hz")
        self.max_label.setText(f"{max_f0} Hz")
        self.min_label.setPos(end, self.bottom_point)
        self.max_label.setPos(end, self.top_point)

    def set_models(self, selection_model: FileSelectionModel):
        self.selection_model = selection_model


class Spectrogram(pg.ImageItem):
    def __init__(self, bottom_point, top_point):
        self.settings = SplaatSettings()
        self.plot_theme = self.settings.plot_theme
        self.top_point = top_point
        self.bottom_point = bottom_point
        self.selection_model = None
        self.channel = 0
        super(Spectrogram, self).__init__()
        self.cmap = pg.ColorMap(
            None,
            [
                self.plot_theme.background_color,
                self.plot_theme.spectrogram_color,
            ],
        )
        self.cmap.linearize()
        self.color_bar = pg.ColorBarItem(colorMap=self.cmap)
        self.color_bar.setImageItem(self)
        self.setAcceptHoverEvents(False)
        self.cached_begin = None
        self.cached_end = None
        self.cached_channel = None
        self.stft = None

    def update_theme(self):
        self.cmap = pg.ColorMap(
            None,
            [
                self.plot_theme.background_color,
                self.plot_theme.spectrogram_color,
            ],
        )
        self.cmap.linearize()
        self.color_bar.setColorMap(self.cmap)
        self.color_bar.setImageItem(self)
        self.update()

    def set_models(self, selection_model: FileSelectionModel):
        self.selection_model = selection_model

    def boundingRect(self):
        br = super(Spectrogram, self).boundingRect()
        return br

    def setData(self, stft, channel, begin, end, min_db, max_db):
        self.stft = stft
        self.min_db = min_db
        self.max_db = max_db
        self.cached_end = end
        self.cached_begin = begin
        self.cached_channel = channel
        duration = self.cached_end - self.cached_begin
        rect = [self.cached_begin, self.bottom_point, duration, self.top_point - self.bottom_point]
        self.setLevels([self.min_db, self.max_db], update=False)
        self.setImage(self.stft, colorMap=self.cmap, rect=rect)
        self.show()


class SelectionArea(pg.LinearRegionItem):
    def __init__(self, top_point, bottom_point, brush, clipItem, pen):
        self.settings = SplaatSettings()
        self.selection_model: typing.Optional[FileSelectionModel] = None
        super(SelectionArea, self).__init__(
            values=(-10, -5),
            span=(bottom_point / top_point, 1),
            brush=brush,
            movable=False,
            # clipItem=clipItem,
            pen=pen,
            orientation="vertical",
        )
        self.setZValue(30)
        self.lines[0].label = pg.InfLineLabel(
            self.lines[0], text="", position=1, anchors=[(1, 0), (1, 0)]
        )
        self.lines[1].label = pg.InfLineLabel(
            self.lines[1], text="", position=1, anchors=[(0, 0), (0, 0)]
        )
        font = self.settings.font
        font.setBold(True)
        self.lines[0].label.setFont(font)
        self.lines[1].label.setFont(font)

    def set_model(self, selection_model: FileSelectionModel):
        self.selection_model = selection_model
        self.selection_model.selectionAudioChanged.connect(self.update_region)

    def update_region(self):
        begin = self.selection_model.selected_min_time
        end = self.selection_model.selected_max_time
        if (
            begin is None
            or end is None
            or (begin == self.selection_model.plot_min and end == self.selection_model.plot_max)
        ):
            self.setVisible(False)
        else:
            self.setRegion([begin, end])
            self.lines[0].label.setText(
                f"{begin:.3f}", self.settings.plot_theme.selected_range_color
            )
            self.lines[1].label.setText(
                f"{end:.3f}", self.settings.plot_theme.selected_range_color
            )
            self.setVisible(True)


class AudioPlots(pg.GraphicsObject):
    def __init__(self, top_point, separator_point, bottom_point):
        super().__init__()
        self.settings = SplaatSettings()
        self.plot_theme = self.settings.plot_theme
        self.selection_model: typing.Optional[FileSelectionModel] = None
        self.top_point = top_point
        self.separator_point = separator_point
        self.bottom_point = bottom_point
        self.wave_form = WaveForm(separator_point, self.top_point)
        self.spectrogram = Spectrogram(self.bottom_point, separator_point)
        # self.pitch_track = PitchTrack(self.bottom_point, separator_point)
        self.wave_form.setParentItem(self)
        self.spectrogram.setParentItem(self)
        # self.pitch_track.setParentItem(self)
        self.grabGesture(QtCore.Qt.PinchGesture)
        color = self.plot_theme.selected_range_color
        color.setAlphaF(0.25)
        self.selection_brush = pg.mkBrush(color)
        self.background_pen = pg.mkPen(self.plot_theme.background_color)
        self.background_brush = pg.mkBrush(self.plot_theme.background_color)
        self.selection_area = SelectionArea(
            top_point=self.top_point,
            bottom_point=self.bottom_point,
            brush=self.selection_brush,
            clipItem=self,
            pen=pg.mkPen(self.plot_theme.selected_interval_color),
        )
        self.selection_area.setParentItem(self)

        self.play_timer = QtCore.QTimer()
        self.play_timer.setInterval(1)
        self.play_timer.timeout.connect(self.update_play_line)

        self.play_line = pg.InfiniteLine(
            pos=-20,
            span=(0, 1),
            pen=pg.mkPen("r", width=1),
            movable=False,  # We have our own code to handle dragless moving.
        )
        self.play_line.setParentItem(self)

        self.update_line = pg.InfiniteLine(
            pos=-20,
            span=(0, 1),
            pen=pg.mkPen(
                self.plot_theme.selected_interval_color,
                width=3,
                style=QtCore.Qt.PenStyle.DashLine,
            ),
            movable=False,  # We have our own code to handle dragless moving.
        )
        self.update_line.setParentItem(self)
        self.update_line.hide()
        self.setAcceptHoverEvents(True)
        self.picture = QtGui.QPicture()
        self.rect = QtCore.QRectF(
            left=0, top=self.top_point, width=10, height=self.top_point - self.bottom_point
        )
        self.rect.setTop(self.top_point)
        self.rect.setBottom(self.bottom_point)
        self._generate_picture()

    def update_drag_line(self, line: WordLine):
        self.update_line.setPos(line.pos())
        self.update_line.show()

    def hide_drag_line(self):
        self.update_line.hide()

    def sceneEvent(self, ev):
        if ev.type() == QtCore.QEvent.Gesture:
            return self.gestureEvent(ev)
        return super().sceneEvent(ev)

    def gestureEvent(self, ev):
        ev.accept()
        pinch = ev.gesture(QtCore.Qt.PinchGesture)
        if pinch is not None:
            delta = pinch.scaleFactor()
            sc = delta
            center = self.getViewBox().mapToView(pinch.centerPoint())
            self.selection_model.zoom(sc, center.x())

    def wheelEvent(self, ev: QtWidgets.QGraphicsSceneWheelEvent):
        ev.accept()
        delta = ev.delta()
        sc = 1.001**delta
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            center = self.getViewBox().mapSceneToView(ev.scenePos())
            self.selection_model.zoom(sc, center.x())
        else:
            self.selection_model.pan(sc)

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if self.selection_model.plot_min is None:
            ev.ignore()
            return
        min_time = max(min(ev.buttonDownPos().x(), ev.pos().x()), self.selection_model.plot_min)
        max_time = min(max(ev.buttonDownPos().x(), ev.pos().x()), self.selection_model.plot_max)
        if ev.isStart():
            self.selection_area.setVisible(True)
        if ev.isFinish():
            self.selection_model.select_audio(min_time, max_time)
        ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if ev.modifiers() in [
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        ]:
            time = ev.pos().x()
            if self.selection_model.selected_max_time is not None:
                if (
                    self.selection_model.selected_min_time
                    < time
                    < self.selection_model.selected_max_time
                ):
                    if (
                        time - self.selection_model.selected_min_time
                        < self.selection_model.selected_max_time - time
                    ):
                        min_time = time
                        max_time = self.selection_model.selected_max_time
                    else:
                        min_time = self.selection_model.selected_min_time
                        max_time = time
                else:
                    min_time = min(
                        time,
                        self.selection_model.selected_min_time,
                        self.selection_model.selected_max_time,
                    )
                    max_time = max(
                        time,
                        self.selection_model.selected_min_time,
                        self.selection_model.selected_max_time,
                    )
            else:
                min_time = min(time, self.selection_model.selected_min_time)
                max_time = max(time, self.selection_model.selected_min_time)
            self.selection_area.setRegion((min_time, max_time))
            self.selection_area.setVisible(True)
            self.selection_model.select_audio(min_time, max_time)
        else:
            self.selection_model.request_start_time(ev.pos().x(), update=True)
        ev.accept()

    def hoverEvent(self, ev):
        if not ev.isExit():
            # the mouse is hovering over the image; make sure no other items
            # will receive left click/drag events from here.

            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)

    def set_models(self, selection_model: FileSelectionModel):
        self.selection_model = selection_model
        self.wave_form.set_models(selection_model)
        self.spectrogram.set_models(selection_model)
        self.selection_area.set_model(selection_model)

    def _generate_picture(self):
        if self.selection_model is None:
            return
        painter = QtGui.QPainter(self.picture)
        painter.setPen(self.background_pen)
        painter.setBrush(self.background_brush)
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, *args):
        painter.save()
        painter.drawPicture(0, 0, self.picture)
        painter.restore()

    def boundingRect(self):
        br = QtCore.QRectF(self.picture.boundingRect())
        return br

    def update_play_line(self, time=None):
        if time is None:
            return
        self.play_line.setVisible(
            self.selection_model.min_time <= time <= self.selection_model.max_time
        )
        self.play_line.setPos(time)

    def update_plot(self):
        self.setVisible(False)
        self.play_line.setVisible(False)
        if (
            self.selection_model.model().file is None
            or self.selection_model.model().file.sound_file is None
            or not os.path.exists(self.selection_model.model().file.sound_file.sound_file_path)
        ):
            return
        self.rect.setLeft(self.selection_model.plot_min)
        self.rect.setRight(self.selection_model.plot_max)
        self._generate_picture()
        # self.selection_area.update_region()
        self.setVisible(True)


class SpeakerTier(pg.GraphicsObject):
    receivedWheelEvent = QtCore.Signal(object)
    receivedGestureEvent = QtCore.Signal(object)
    draggingLine = QtCore.Signal(object)
    lineDragFinished = QtCore.Signal(object)

    def __init__(
        self,
        top_point,
        bottom_point,
        speaker_name: str,
        corpus_model: CorpusModel,
        file_model: FileModel,
        selection_model: FileSelectionModel,
    ):
        super().__init__()
        self.file_model = file_model
        self.corpus_model = corpus_model
        self.selection_model = selection_model
        self.settings = SplaatSettings()
        self.plot_theme = self.settings.plot_theme
        self.speaker_name = speaker_name
        self.speaker_index = 0
        self.top_point = top_point
        self.speaker_label = pg.TextItem(self.speaker_name, color=self.plot_theme.break_line_color)
        self.speaker_label.setFont(self.settings.font)
        self.speaker_label.setParentItem(self)
        self.speaker_label.setZValue(40)
        self.speaker_label.setText("Words")
        self.bottom_point = bottom_point
        self.annotation_range = self.top_point - self.bottom_point
        self.visible_utterances: dict[int, UtteranceRegion] = {}
        self.background_brush = pg.mkBrush(self.plot_theme.background_color)
        self.border = pg.mkPen(self.plot_theme.break_line_color)
        self.picture = QtGui.QPicture()
        self.has_visible_words = False
        self.has_selected_words = False
        self.text_search_term = None
        self.phones_search_term = None
        self.rect = QtCore.QRectF(
            left=self.selection_model.plot_min,
            riht=self.selection_model.plot_max,
            top=self.top_point,
            bottom=self.bottom_point,
        )
        self._generate_picture()
        self.selection_model.selectionChanged.connect(self.update_select)
        self.selection_model.model().utterancesReady.connect(self.refresh)
        self.painter_path_cache = {}
        self.grabGesture(QtCore.Qt.PinchGesture)

    def wheelEvent(self, ev):
        self.receivedWheelEvent.emit(ev)

    def sceneEvent(self, ev):
        if ev.type() == QtCore.QEvent.Gesture:
            return self.gestureEvent(ev)
        return super().sceneEvent(ev)

    def gestureEvent(self, ev):
        ev.accept()
        pinch = ev.gesture(QtCore.Qt.PinchGesture)
        if pinch is not None:
            self.receivedGestureEvent.emit(ev)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def _generate_picture(self):
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setPen(self.border)

        painter.setBrush(self.background_brush)
        painter.drawRect(self.rect)
        painter.end()

    def reset_tier(self):
        for reg in self.visible_utterances.values():
            if reg.scene() is not None:
                reg.scene().removeItem(reg)
        self.visible_utterances = {}

    def refresh(self, *args, reset_bounds=False):
        self.hide()
        if self.selection_model.plot_min is None:
            return
        self.has_visible_utterances = False
        self.has_selected_utterances = False
        self.speaker_label.setPos(self.selection_model.plot_min, self.top_point)
        cleanup_ids = []
        model_visible_utterances = self.selection_model.visible_utterances()
        visible_ids = {x.id: x for x in model_visible_utterances}
        for reg in self.visible_utterances.values():
            reg.hide()
            if reset_bounds and reg.item.id in visible_ids:
                with QtCore.QSignalBlocker(reg):
                    reg.item.start, reg.item.end = (
                        visible_ids[reg.item.id].start,
                        visible_ids[reg.item.id].end,
                    )
                    reg.setRegion((reg.item.start, reg.item.end))

            item_min, item_max = reg.getRegion()
            if (
                self.selection_model.min_time - item_max > 15
                or item_min - self.selection_model.max_time > 15
                or (
                    reg.item.id not in visible_ids
                    and (
                        item_min < self.selection_model.max_time
                        or item_max > self.selection_model.min_time
                    )
                )
            ):
                if reg.scene() is not None:
                    reg.scene().removeItem(reg)
                cleanup_ids.append(reg.item.id)
        self.visible_utterances = {
            k: v for k, v in self.visible_utterances.items() if k not in cleanup_ids
        }
        for u in model_visible_utterances:
            if u.id in self.visible_utterances:
                self.visible_utterances[u.id].setSelected(self.selection_model.checkSelected(u.id))
                self.visible_utterances[u.id].show()
                continue
            self.has_visible_utterances = True
            # Utterance region always at the top
            reg = UtteranceRegion(
                self,
                u,
                self.corpus_model,
                self.file_model,
                selection_model=self.selection_model,
                bottom_point=self.bottom_point,
                top_point=self.top_point,
            )
            reg.set_search_term(self.text_search_term, self.phones_search_term)
            reg.sigRegionChanged.connect(self.check_utterance_bounds)
            reg.sigRegionChangeFinished.connect(self.update_utterance)
            reg.draggingLine.connect(self.draggingLine.emit)
            reg.sigRegionChangeFinished.connect(self.lineDragFinished.emit)
            reg.audioSelected.connect(self.selection_model.select_audio)
            reg.viewRequested.connect(self.selection_model.set_view_times)
            reg.phoneBoundaryChanged.connect(self.update_phone_boundaries)
            reg.phoneIntervalChanged.connect(self.update_phone_interval)
            reg.phoneIntervalInserted.connect(self.insert_phone_interval)
            reg.phoneIntervalDeleted.connect(self.delete_phone_interval)
            self.visible_utterances[u.id] = reg

        self.show()

    def update_phone_boundaries(
        self, utterance: Utterance, first_phone_interval, second_phone_interval, new_time: float
    ):
        self.selection_model.model().update_phone_boundaries(
            utterance, first_phone_interval, second_phone_interval, new_time
        )

    def update_phone_interval(self, utterance: Utterance, phone_interval, phone_id):
        self.selection_model.model().update_phone_interval(utterance, phone_interval, phone_id)

    def insert_phone_interval(
        self, utterance: Utterance, interval, previous_interval, following_interval, word_interval
    ):
        self.selection_model.model().insert_phone_interval(
            utterance, interval, previous_interval, following_interval, word_interval
        )

    def delete_phone_interval(
        self, utterance: Utterance, interval, previous_interval, following_interval, time_point
    ):
        self.selection_model.model().delete_phone_interval(
            utterance, interval, previous_interval, following_interval, time_point
        )

    def update_select(self):
        selected_rows = {x.id for x in self.selection_model.selected_utterances()}
        for r in self.visible_utterances.values():
            if r.item.id in selected_rows:
                r.setSelected(True)
            else:
                r.setSelected(False)

    def check_utterance_bounds(self):
        reg: UtteranceRegion = self.sender()
        with QtCore.QSignalBlocker(reg):
            beg, end = reg.getRegion()
            if self.settings.right_to_left:
                if end > 0:
                    reg.setRegion([beg, 0])
                    return
                if (
                    self.selection_model.model().file is not None
                    and -end > self.selection_model.model().file.duration
                ):
                    reg.setRegion([beg, self.selection_model.model().file.duration])
                    return
            else:
                if beg < 0:
                    reg.setRegion([0, end])
                    return
                if (
                    self.selection_model.model().file is not None
                    and end > self.selection_model.model().file.duration
                ):
                    reg.setRegion([beg, self.selection_model.model().file.duration])
                    return
            prev_r = None
            for r in sorted(self.visible_utterances.values(), key=lambda x: x.item_min):
                if r.item.id == reg.item.id:
                    if reg.initial_line_moving and reg.snap_mode and prev_r is not None:
                        other_begin, other_end = prev_r.getRegion()
                        prev_r.setRegion([other_begin, beg])
                        break
                    continue
                other_begin, other_end = r.getRegion()
                if other_begin <= beg < other_end or beg <= other_begin < other_end < end:
                    if reg.initial_line_moving and reg.snap_mode:
                        r.setRegion([other_begin, beg])
                    else:
                        reg.setRegion([other_end, end])
                    break
                if other_begin < end <= other_end or end > other_begin > other_end > beg:
                    if (
                        False
                        and not reg.initial_line_moving
                        and reg.snap_mode
                        and prev_r is not None
                        and prev_r.item.id == reg.item.id
                    ):
                        r.setRegion([end, other_end])
                    else:
                        reg.setRegion([beg, other_begin])
                    break
                prev_r = r

        reg.update()

    def set_search_term(self, text_search_term, phones_search_term):
        self.text_search_term = text_search_term
        self.phones_search_term = phones_search_term
        for utt in self.visible_utterances.values():
            utt.set_search_term(text_search_term, phones_search_term)

    def update_utterance(self):
        reg = self.sender()
        utt = reg.item

        beg, end = reg.getRegion()
        new_begin = round(beg, 4)
        new_end = round(end, 4)
        if new_begin == utt.start and new_end == utt.end:
            return
        self.selection_model.request_start_time(new_begin)
        self.lineDragFinished.emit(True)
