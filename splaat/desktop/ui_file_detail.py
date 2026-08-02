# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'file_detail.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTime,
    QUrl,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMenuBar,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QWidget,
)

from splaat.desktop.widgets import DetailView


class Ui_FileDetailWindow(object):
    def setupUi(self, FileDetailWindow):
        if not FileDetailWindow.objectName():
            FileDetailWindow.setObjectName("FileDetailWindow")
        FileDetailWindow.resize(800, 600)
        self.playAct = QAction(FileDetailWindow)
        self.playAct.setObjectName("playAct")
        self.playAct.setCheckable(True)
        icon = QIcon(QIcon.fromTheme("media-playback-start"))
        self.playAct.setIcon(icon)
        self.playAct.setMenuRole(QAction.TextHeuristicRole)
        self.zoomAllAct = QAction(FileDetailWindow)
        self.zoomAllAct.setObjectName("zoomAllAct")
        self.zoomAllAct.setMenuRole(QAction.NoRole)
        self.zoomInAct = QAction(FileDetailWindow)
        self.zoomInAct.setObjectName("zoomInAct")
        icon1 = QIcon(QIcon.fromTheme("zoom-in"))
        self.zoomInAct.setIcon(icon1)
        self.zoomInAct.setMenuRole(QAction.TextHeuristicRole)
        self.zoomOutAct = QAction(FileDetailWindow)
        self.zoomOutAct.setObjectName("zoomOutAct")
        icon2 = QIcon(QIcon.fromTheme("zoom-out"))
        self.zoomOutAct.setIcon(icon2)
        self.zoomOutAct.setMenuRole(QAction.TextHeuristicRole)
        self.zoomToSelectionAct = QAction(FileDetailWindow)
        self.zoomToSelectionAct.setObjectName("zoomToSelectionAct")
        icon3 = QIcon(QIcon.fromTheme("zoom-fit-best"))
        self.zoomToSelectionAct.setIcon(icon3)
        self.zoomToSelectionAct.setMenuRole(QAction.TextHeuristicRole)
        self.panLeftAct = QAction(FileDetailWindow)
        self.panLeftAct.setObjectName("panLeftAct")
        self.panLeftAct.setMenuRole(QAction.TextHeuristicRole)
        self.panRightAct = QAction(FileDetailWindow)
        self.panRightAct.setObjectName("panRightAct")
        self.panRightAct.setMenuRole(QAction.TextHeuristicRole)
        self.searchAct = QAction(FileDetailWindow)
        self.searchAct.setObjectName("searchAct")
        self.searchAct.setMenuRole(QAction.TextHeuristicRole)
        self.actionPreferences = QAction(FileDetailWindow)
        self.actionPreferences.setObjectName("actionPreferences")
        self.centralwidget = DetailView(FileDetailWindow)
        self.centralwidget.setObjectName("centralwidget")
        FileDetailWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(FileDetailWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        FileDetailWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(FileDetailWindow)
        self.statusbar.setObjectName("statusbar")
        FileDetailWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(FileDetailWindow)
        self.toolBar.setObjectName("toolBar")
        self.toolBar.setMovable(False)
        self.toolBar.setFloatable(False)
        FileDetailWindow.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolBar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menuEdit.addAction(self.actionPreferences)
        self.toolBar.addAction(self.playAct)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.zoomAllAct)
        self.toolBar.addAction(self.zoomInAct)
        self.toolBar.addAction(self.zoomOutAct)
        self.toolBar.addAction(self.zoomToSelectionAct)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.searchAct)

        self.retranslateUi(FileDetailWindow)

        QMetaObject.connectSlotsByName(FileDetailWindow)

    # setupUi

    def retranslateUi(self, FileDetailWindow):
        FileDetailWindow.setWindowTitle(
            QCoreApplication.translate("FileDetailWindow", "MainWindow", None)
        )
        self.playAct.setText(QCoreApplication.translate("FileDetailWindow", "Play", None))
        self.zoomAllAct.setText(QCoreApplication.translate("FileDetailWindow", "All", None))
        # if QT_CONFIG(tooltip)
        self.zoomAllAct.setToolTip(
            QCoreApplication.translate("FileDetailWindow", "Show full file", None)
        )
        # endif // QT_CONFIG(tooltip)
        self.zoomInAct.setText(QCoreApplication.translate("FileDetailWindow", "In", None))
        # if QT_CONFIG(tooltip)
        self.zoomInAct.setToolTip(QCoreApplication.translate("FileDetailWindow", "Zoom in", None))
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(shortcut)
        self.zoomInAct.setShortcut(QCoreApplication.translate("FileDetailWindow", "Ctrl+I", None))
        # endif // QT_CONFIG(shortcut)
        self.zoomOutAct.setText(QCoreApplication.translate("FileDetailWindow", "Out", None))
        # if QT_CONFIG(tooltip)
        self.zoomOutAct.setToolTip(
            QCoreApplication.translate("FileDetailWindow", "Zoom out", None)
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(shortcut)
        self.zoomOutAct.setShortcut(QCoreApplication.translate("FileDetailWindow", "Ctrl+O", None))
        # endif // QT_CONFIG(shortcut)
        self.zoomToSelectionAct.setText(
            QCoreApplication.translate("FileDetailWindow", "Sel", None)
        )
        # if QT_CONFIG(tooltip)
        self.zoomToSelectionAct.setToolTip(
            QCoreApplication.translate("FileDetailWindow", "Zoom to selection", None)
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(shortcut)
        self.zoomToSelectionAct.setShortcut(
            QCoreApplication.translate("FileDetailWindow", "Ctrl+N", None)
        )
        # endif // QT_CONFIG(shortcut)
        self.panLeftAct.setText(QCoreApplication.translate("FileDetailWindow", "Pan left", None))
        # if QT_CONFIG(tooltip)
        self.panLeftAct.setToolTip(
            QCoreApplication.translate("FileDetailWindow", "Pan left", None)
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(shortcut)
        self.panLeftAct.setShortcut(QCoreApplication.translate("FileDetailWindow", "Left", None))
        # endif // QT_CONFIG(shortcut)
        self.panRightAct.setText(QCoreApplication.translate("FileDetailWindow", "Pan right", None))
        # if QT_CONFIG(tooltip)
        self.panRightAct.setToolTip(
            QCoreApplication.translate("FileDetailWindow", "Pan right", None)
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(shortcut)
        self.panRightAct.setShortcut(QCoreApplication.translate("FileDetailWindow", "Right", None))
        # endif // QT_CONFIG(shortcut)
        self.searchAct.setText(QCoreApplication.translate("FileDetailWindow", "Search file", None))
        # if QT_CONFIG(tooltip)
        self.searchAct.setToolTip(
            QCoreApplication.translate("FileDetailWindow", "Search file", None)
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(shortcut)
        self.searchAct.setShortcut(QCoreApplication.translate("FileDetailWindow", "Ctrl+F", None))
        # endif // QT_CONFIG(shortcut)
        self.actionPreferences.setText(
            QCoreApplication.translate("FileDetailWindow", "Preferences...", None)
        )
        self.menuFile.setTitle(QCoreApplication.translate("FileDetailWindow", "File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("FileDetailWindow", "Edit", None))
        self.toolBar.setWindowTitle(
            QCoreApplication.translate("FileDetailWindow", "toolBar", None)
        )

    # retranslateUi
