# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
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
    QVBoxLayout,
    QWidget,
)

from splaat.desktop.widgets import FileListWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(522, 732)
        MainWindow.setMinimumSize(QSize(522, 732))
        MainWindow.setStyleSheet("")
        MainWindow.setAnimated(True)
        MainWindow.setDocumentMode(False)
        MainWindow.setDockOptions(
            QMainWindow.AllowTabbedDocks
            | QMainWindow.AnimatedDocks
            | QMainWindow.ForceTabbedDocks
            | QMainWindow.VerticalTabs
        )
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionOpenGuidelines = QAction(MainWindow)
        self.actionOpenGuidelines.setObjectName("actionOpenGuidelines")
        self.actionSplaat_Intro = QAction(MainWindow)
        self.actionSplaat_Intro.setObjectName("actionSplaat_Intro")
        self.actionOpenFolder = QAction(MainWindow)
        self.actionOpenFolder.setObjectName("actionOpenFolder")
        self.actionView = QAction(MainWindow)
        self.actionView.setObjectName("actionView")
        self.actionView.setMenuRole(QAction.NoRole)
        self.actionPreferences = QAction(MainWindow)
        self.actionPreferences.setObjectName("actionPreferences")
        self.actionReport_bug = QAction(MainWindow)
        self.actionReport_bug.setObjectName("actionReport_bug")
        self.actionAnalyze_alignments = QAction(MainWindow)
        self.actionAnalyze_alignments.setObjectName("actionAnalyze_alignments")
        self.actionSearch = QAction(MainWindow)
        self.actionSearch.setObjectName("actionSearch")
        self.actionSearch.setMenuRole(QAction.NoRole)
        self.actionInfo = QAction(MainWindow)
        self.actionInfo.setObjectName("actionInfo")
        self.actionInfo.setMenuRole(QAction.NoRole)
        self.actionRemove = QAction(MainWindow)
        self.actionRemove.setObjectName("actionRemove")
        self.actionRemove.setMenuRole(QAction.NoRole)
        self.centralwidget = FileListWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 522, 21))
        self.menuSplaat = QMenu(self.menubar)
        self.menuSplaat.setObjectName("menuSplaat")
        self.menuExit = QMenu(self.menuSplaat)
        self.menuExit.setObjectName("menuExit")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuOpen = QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.loadRecentFoldersMenu = QMenu(self.menuOpen)
        self.loadRecentFoldersMenu.setObjectName("loadRecentFoldersMenu")
        self.menuAnalyze = QMenu(self.menubar)
        self.menuAnalyze.setObjectName("menuAnalyze")
        MainWindow.setMenuBar(self.menubar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        self.toolBar.setMovable(False)
        self.toolBar.setAllowedAreas(Qt.BottomToolBarArea)
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        MainWindow.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolBar)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.menubar.addAction(self.menuSplaat.menuAction())
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuAnalyze.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuSplaat.addAction(self.actionAbout)
        self.menuSplaat.addSeparator()
        self.menuSplaat.addAction(self.actionPreferences)
        self.menuSplaat.addSeparator()
        self.menuSplaat.addAction(self.menuExit.menuAction())
        self.menuHelp.addAction(self.actionSplaat_Intro)
        self.menuHelp.addAction(self.actionOpenGuidelines)
        self.menuHelp.addAction(self.actionReport_bug)
        self.menuOpen.addAction(self.actionOpenFolder)
        self.menuOpen.addAction(self.loadRecentFoldersMenu.menuAction())
        self.loadRecentFoldersMenu.addSeparator()
        self.menuAnalyze.addAction(self.actionAnalyze_alignments)
        self.toolBar.addAction(self.actionSearch)
        self.toolBar.addAction(self.actionView)
        self.toolBar.addAction(self.actionInfo)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRemove)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", "Splaat", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", "About Splaat", None))
        self.actionOpenGuidelines.setText(
            QCoreApplication.translate("MainWindow", "Open annotation guidelines", None)
        )
        self.actionSplaat_Intro.setText(
            QCoreApplication.translate("MainWindow", "Splaat Intro", None)
        )
        self.actionOpenFolder.setText(
            QCoreApplication.translate("MainWindow", "Open folder...", None)
        )
        self.actionView.setText(QCoreApplication.translate("MainWindow", "View and Edit", None))
        # if QT_CONFIG(tooltip)
        self.actionView.setToolTip(
            QCoreApplication.translate("MainWindow", "View and edit selected file", None)
        )
        # endif // QT_CONFIG(tooltip)
        self.actionPreferences.setText(
            QCoreApplication.translate("MainWindow", "Preferences...", None)
        )
        self.actionReport_bug.setText(QCoreApplication.translate("MainWindow", "Report bug", None))
        self.actionAnalyze_alignments.setText(
            QCoreApplication.translate("MainWindow", "Analyze alignments", None)
        )
        self.actionSearch.setText(QCoreApplication.translate("MainWindow", "Search", None))
        self.actionInfo.setText(QCoreApplication.translate("MainWindow", "Info", None))
        self.actionRemove.setText(QCoreApplication.translate("MainWindow", "Remove", None))
        self.menuSplaat.setTitle(QCoreApplication.translate("MainWindow", "Splaat", None))
        self.menuExit.setTitle(QCoreApplication.translate("MainWindow", "Exit", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", "Help", None))
        self.menuOpen.setTitle(QCoreApplication.translate("MainWindow", "Open", None))
        self.loadRecentFoldersMenu.setTitle(
            QCoreApplication.translate("MainWindow", "Recent folders", None)
        )
        self.menuAnalyze.setTitle(QCoreApplication.translate("MainWindow", "Analyze", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", "toolBar", None))

    # retranslateUi
