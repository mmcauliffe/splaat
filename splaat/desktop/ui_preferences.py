# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'preferences.ui'
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
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from splaat.desktop.widgets import FontEdit


class Ui_PreferencesDialog(object):
    def setupUi(self, PreferencesDialog):
        if not PreferencesDialog.objectName():
            PreferencesDialog.setObjectName("PreferencesDialog")
        PreferencesDialog.resize(724, 451)
        self.verticalLayout = QVBoxLayout(PreferencesDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QTabWidget(PreferencesDialog)
        self.tabWidget.setObjectName("tabWidget")
        self.generalTab = QWidget()
        self.generalTab.setObjectName("generalTab")
        self.verticalLayout_8 = QVBoxLayout(self.generalTab)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.scrollArea_3 = QScrollArea(self.generalTab)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 682, 199))
        self.formLayout = QFormLayout(self.scrollAreaWidgetContents_3)
        self.formLayout.setObjectName("formLayout")
        self.label_12 = QLabel(self.scrollAreaWidgetContents_3)
        self.label_12.setObjectName("label_12")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_12)

        self.fontEdit = FontEdit(self.scrollAreaWidgetContents_3)
        self.fontEdit.setObjectName("fontEdit")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.fontEdit)

        self.autosaveLabel = QLabel(self.scrollAreaWidgetContents_3)
        self.autosaveLabel.setObjectName("autosaveLabel")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.autosaveLabel)

        self.autosaveOnExitCheckBox = QCheckBox(self.scrollAreaWidgetContents_3)
        self.autosaveOnExitCheckBox.setObjectName("autosaveOnExitCheckBox")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.autosaveOnExitCheckBox)

        self.autoloadLastUsedCorpusLabel = QLabel(self.scrollAreaWidgetContents_3)
        self.autoloadLastUsedCorpusLabel.setObjectName("autoloadLastUsedCorpusLabel")

        self.formLayout.setWidget(
            2, QFormLayout.ItemRole.LabelRole, self.autoloadLastUsedCorpusLabel
        )

        self.autoloadLastUsedCorpusCheckBox = QCheckBox(self.scrollAreaWidgetContents_3)
        self.autoloadLastUsedCorpusCheckBox.setObjectName("autoloadLastUsedCorpusCheckBox")

        self.formLayout.setWidget(
            2, QFormLayout.ItemRole.FieldRole, self.autoloadLastUsedCorpusCheckBox
        )

        self.fadeInLabel = QLabel(self.scrollAreaWidgetContents_3)
        self.fadeInLabel.setObjectName("fadeInLabel")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.fadeInLabel)

        self.enableFadeCheckBox = QCheckBox(self.scrollAreaWidgetContents_3)
        self.enableFadeCheckBox.setObjectName("enableFadeCheckBox")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.enableFadeCheckBox)

        self.resultsPerPageLabel = QLabel(self.scrollAreaWidgetContents_3)
        self.resultsPerPageLabel.setObjectName("resultsPerPageLabel")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.LabelRole, self.resultsPerPageLabel)

        self.resultsPerPageEdit = QSpinBox(self.scrollAreaWidgetContents_3)
        self.resultsPerPageEdit.setObjectName("resultsPerPageEdit")
        self.resultsPerPageEdit.setMaximum(1000)

        self.formLayout.setWidget(4, QFormLayout.ItemRole.FieldRole, self.resultsPerPageEdit)

        self.timeDirectionLabel = QLabel(self.scrollAreaWidgetContents_3)
        self.timeDirectionLabel.setObjectName("timeDirectionLabel")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.LabelRole, self.timeDirectionLabel)

        self.timeDirectionComboBox = QComboBox(self.scrollAreaWidgetContents_3)
        self.timeDirectionComboBox.addItem("")
        self.timeDirectionComboBox.addItem("")
        self.timeDirectionComboBox.setObjectName("timeDirectionComboBox")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.FieldRole, self.timeDirectionComboBox)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.verticalLayout_8.addWidget(self.scrollArea_3, 0, Qt.AlignTop)

        self.tabWidget.addTab(self.generalTab, "")
        self.keybindTab = QWidget()
        self.keybindTab.setObjectName("keybindTab")
        self.verticalLayout_2 = QVBoxLayout(self.keybindTab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.scrollArea_2 = QScrollArea(self.keybindTab)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 682, 353))
        self.verticalLayout_7 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_2.setObjectName("label_2")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.zoomInShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.zoomInShortcutEdit.setObjectName("zoomInShortcutEdit")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.FieldRole, self.zoomInShortcutEdit)

        self.label_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_3.setObjectName("label_3")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_3)

        self.zoomOutShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.zoomOutShortcutEdit.setObjectName("zoomOutShortcutEdit")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.FieldRole, self.zoomOutShortcutEdit)

        self.label_31 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_31.setObjectName("label_31")

        self.formLayout_2.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_31)

        self.zoomToSelectionShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.zoomToSelectionShortcutEdit.setObjectName("zoomToSelectionShortcutEdit")

        self.formLayout_2.setWidget(
            4, QFormLayout.ItemRole.FieldRole, self.zoomToSelectionShortcutEdit
        )

        self.label_4 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_4.setObjectName("label_4")

        self.formLayout_2.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_4)

        self.panLeftShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.panLeftShortcutEdit.setObjectName("panLeftShortcutEdit")

        self.formLayout_2.setWidget(5, QFormLayout.ItemRole.FieldRole, self.panLeftShortcutEdit)

        self.label_5 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_5.setObjectName("label_5")

        self.formLayout_2.setWidget(6, QFormLayout.ItemRole.LabelRole, self.label_5)

        self.panRightShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.panRightShortcutEdit.setObjectName("panRightShortcutEdit")

        self.formLayout_2.setWidget(6, QFormLayout.ItemRole.FieldRole, self.panRightShortcutEdit)

        self.label = QLabel(self.scrollAreaWidgetContents_2)
        self.label.setObjectName("label")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label)

        self.playAudioShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.playAudioShortcutEdit.setObjectName("playAudioShortcutEdit")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.FieldRole, self.playAudioShortcutEdit)

        self.label_9 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_9.setObjectName("label_9")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_9)

        self.saveShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.saveShortcutEdit.setObjectName("saveShortcutEdit")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.FieldRole, self.saveShortcutEdit)

        self.label_10 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_10.setObjectName("label_10")

        self.formLayout_2.setWidget(7, QFormLayout.ItemRole.LabelRole, self.label_10)

        self.searchShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.searchShortcutEdit.setObjectName("searchShortcutEdit")

        self.formLayout_2.setWidget(7, QFormLayout.ItemRole.FieldRole, self.searchShortcutEdit)

        self.undoShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.undoShortcutEdit.setObjectName("undoShortcutEdit")

        self.formLayout_2.setWidget(8, QFormLayout.ItemRole.FieldRole, self.undoShortcutEdit)

        self.label_23 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_23.setObjectName("label_23")

        self.formLayout_2.setWidget(8, QFormLayout.ItemRole.LabelRole, self.label_23)

        self.label_22 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_22.setObjectName("label_22")

        self.formLayout_2.setWidget(9, QFormLayout.ItemRole.LabelRole, self.label_22)

        self.redoShortcutEdit = QKeySequenceEdit(self.scrollAreaWidgetContents_2)
        self.redoShortcutEdit.setObjectName("redoShortcutEdit")

        self.formLayout_2.setWidget(9, QFormLayout.ItemRole.FieldRole, self.redoShortcutEdit)

        self.verticalLayout_7.addLayout(self.formLayout_2)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_2.addWidget(self.scrollArea_2)

        self.tabWidget.addTab(self.keybindTab, "")
        self.spectrogramTab = QWidget()
        self.spectrogramTab.setObjectName("spectrogramTab")
        self.verticalLayout_10 = QVBoxLayout(self.spectrogramTab)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.scrollArea_4 = QScrollArea(self.spectrogramTab)
        self.scrollArea_4.setObjectName("scrollArea_4")
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 682, 353))
        self.verticalLayout_11 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.formLayout_6 = QFormLayout()
        self.formLayout_6.setObjectName("formLayout_6")
        self.label_42 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_42.setObjectName("label_42")

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_42)

        self.dynamicRangeEdit = QSpinBox(self.scrollAreaWidgetContents_4)
        self.dynamicRangeEdit.setObjectName("dynamicRangeEdit")
        self.dynamicRangeEdit.setMaximum(10000)

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.FieldRole, self.dynamicRangeEdit)

        self.fftSizeEdit = QSpinBox(self.scrollAreaWidgetContents_4)
        self.fftSizeEdit.setObjectName("fftSizeEdit")
        self.fftSizeEdit.setMaximum(1000000)

        self.formLayout_6.setWidget(1, QFormLayout.ItemRole.FieldRole, self.fftSizeEdit)

        self.numTimeStepsEdit = QSpinBox(self.scrollAreaWidgetContents_4)
        self.numTimeStepsEdit.setObjectName("numTimeStepsEdit")
        self.numTimeStepsEdit.setMaximum(100000)
        self.numTimeStepsEdit.setSingleStep(10)

        self.formLayout_6.setWidget(2, QFormLayout.ItemRole.FieldRole, self.numTimeStepsEdit)

        self.maxFrequencyEdit = QSpinBox(self.scrollAreaWidgetContents_4)
        self.maxFrequencyEdit.setObjectName("maxFrequencyEdit")
        self.maxFrequencyEdit.setMaximum(1000000)
        self.maxFrequencyEdit.setSingleStep(1000)

        self.formLayout_6.setWidget(3, QFormLayout.ItemRole.FieldRole, self.maxFrequencyEdit)

        self.windowSizeEdit = QLineEdit(self.scrollAreaWidgetContents_4)
        self.windowSizeEdit.setObjectName("windowSizeEdit")

        self.formLayout_6.setWidget(4, QFormLayout.ItemRole.FieldRole, self.windowSizeEdit)

        self.preemphasisEdit = QLineEdit(self.scrollAreaWidgetContents_4)
        self.preemphasisEdit.setObjectName("preemphasisEdit")

        self.formLayout_6.setWidget(5, QFormLayout.ItemRole.FieldRole, self.preemphasisEdit)

        self.label_43 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_43.setObjectName("label_43")

        self.formLayout_6.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_43)

        self.label_44 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_44.setObjectName("label_44")

        self.formLayout_6.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_44)

        self.label_45 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_45.setObjectName("label_45")

        self.formLayout_6.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_45)

        self.label_46 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_46.setObjectName("label_46")

        self.formLayout_6.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_46)

        self.label_47 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_47.setObjectName("label_47")

        self.formLayout_6.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_47)

        self.specMaxTimeEdit = QLineEdit(self.scrollAreaWidgetContents_4)
        self.specMaxTimeEdit.setObjectName("specMaxTimeEdit")

        self.formLayout_6.setWidget(6, QFormLayout.ItemRole.FieldRole, self.specMaxTimeEdit)

        self.label_24 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_24.setObjectName("label_24")

        self.formLayout_6.setWidget(6, QFormLayout.ItemRole.LabelRole, self.label_24)

        self.verticalLayout_11.addLayout(self.formLayout_6)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)

        self.verticalLayout_10.addWidget(self.scrollArea_4)

        self.tabWidget.addTab(self.spectrogramTab, "")

        self.verticalLayout.addWidget(self.tabWidget)

        self.buttonBox = QDialogButtonBox(PreferencesDialog)
        self.buttonBox.setObjectName("buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(PreferencesDialog)
        self.buttonBox.accepted.connect(PreferencesDialog.accept)
        self.buttonBox.rejected.connect(PreferencesDialog.reject)

        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(PreferencesDialog)

    # setupUi

    def retranslateUi(self, PreferencesDialog):
        PreferencesDialog.setWindowTitle(
            QCoreApplication.translate("PreferencesDialog", "Dialog", None)
        )
        self.label_12.setText(QCoreApplication.translate("PreferencesDialog", "Font", None))
        self.fontEdit.setText(QCoreApplication.translate("PreferencesDialog", "PushButton", None))
        self.autosaveLabel.setText(
            QCoreApplication.translate("PreferencesDialog", "Autosave on exit", None)
        )
        self.autoloadLastUsedCorpusLabel.setText(
            QCoreApplication.translate("PreferencesDialog", "Autoload", None)
        )
        self.fadeInLabel.setText(
            QCoreApplication.translate("PreferencesDialog", "Fade in audio on play", None)
        )
        self.enableFadeCheckBox.setText("")
        self.resultsPerPageLabel.setText(
            QCoreApplication.translate("PreferencesDialog", "Results per page", None)
        )
        self.timeDirectionLabel.setText(
            QCoreApplication.translate("PreferencesDialog", "Time direction", None)
        )
        self.timeDirectionComboBox.setItemText(
            0, QCoreApplication.translate("PreferencesDialog", "Left-to-right", None)
        )
        self.timeDirectionComboBox.setItemText(
            1, QCoreApplication.translate("PreferencesDialog", "Right-to-left", None)
        )

        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.generalTab),
            QCoreApplication.translate("PreferencesDialog", "General", None),
        )
        self.label_2.setText(QCoreApplication.translate("PreferencesDialog", "Zoom in", None))
        self.label_3.setText(QCoreApplication.translate("PreferencesDialog", "Zoom out", None))
        self.label_31.setText(
            QCoreApplication.translate("PreferencesDialog", "Zoom to selection", None)
        )
        self.label_4.setText(QCoreApplication.translate("PreferencesDialog", "Pan left", None))
        self.label_5.setText(QCoreApplication.translate("PreferencesDialog", "Pan right", None))
        self.label.setText(QCoreApplication.translate("PreferencesDialog", "Play audio", None))
        self.playAudioShortcutEdit.setKeySequence("")
        self.label_9.setText(
            QCoreApplication.translate("PreferencesDialog", "Save current file", None)
        )
        self.label_10.setText(QCoreApplication.translate("PreferencesDialog", "Search", None))
        self.label_23.setText(QCoreApplication.translate("PreferencesDialog", "Undo", None))
        self.label_22.setText(QCoreApplication.translate("PreferencesDialog", "Redo", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.keybindTab),
            QCoreApplication.translate("PreferencesDialog", "Key shortcuts", None),
        )
        self.label_42.setText(
            QCoreApplication.translate("PreferencesDialog", "Dynamic range (dB)", None)
        )
        self.label_43.setText(QCoreApplication.translate("PreferencesDialog", "FFT size", None))
        self.label_44.setText(
            QCoreApplication.translate("PreferencesDialog", "Number of time steps", None)
        )
        self.label_45.setText(
            QCoreApplication.translate("PreferencesDialog", "Maximum frequency (Hz)", None)
        )
        self.label_46.setText(
            QCoreApplication.translate("PreferencesDialog", "Window size (s)", None)
        )
        self.label_47.setText(
            QCoreApplication.translate("PreferencesDialog", "Pre-emphasis factor", None)
        )
        self.label_24.setText(
            QCoreApplication.translate("PreferencesDialog", "Maximum visible time (s)", None)
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.spectrogramTab),
            QCoreApplication.translate("PreferencesDialog", "Spectrogram", None),
        )

    # retranslateUi
