import os
import sys
import time
import logging

from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QProgressBar, QDialog, QFileDialog, QApplication, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt6.QtCore import Qt, QItemSelection, QThread, QThreadPool, QMutex
from PyQt6.QtGui import QKeySequence, QShortcut, QFont
import pyqtgraph as pg

import numpy as np

from labeler import util
from labeler.project import Project
from labeler.dataloader import RADTrack
from ui.boundingbox import RadarInstance
from ui.loader import DatasetLoader
from ui.plotter import FramePlotManager
from ui.radar import RadarImageView, RadarPlotItem
from ui.models import SequenceModel
from ui.backup import Backup


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(os.path.basename(__file__))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        pg.setConfigOption('imageAxisOrder', 'row-major')
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        uic.loadUi('main_gui.ui', self)
        self.showMaximized()

        self.project: Project = None
        self.dataset: RADTrack = None
        self._activeSequenceRow: int = -1
        self._activeFrame: int = -1
        self._lastActiveFrame: dict[int, int] = {}

        self._selectedInstanceIndex: int = -1
        self._instances: list[RadarInstance] = []

        self._allowEditAllGt = False

        # 20 random colors
        self._cmap = util.random_colors(n=20)

        self._defaultRadarConfig = {
            "designed_frequency": 76.8,
            "config_frequency": 77,
            "maximum_range": 50,
            "range_size": 256,
            "azimuth_size": 256,
            "doppler_size": 64,
            "range_resolution": 0.1953125,
            "angular_resolution": 0.006135923,
            "velocity_resolution": 0.41968030701528203
        }

        self._preloadingMutex = QMutex()
        self._backupMutex = QMutex()

        self.newProjectDialog = NewProjectDialog(self)
        self.newProjectDialog.accepted.connect(self.createNewProject)

        # This will set the left and right dock widgets to stretch over full height
        self.setCorner(Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

        self.loadingProjectBar = self.getLoadingBar()
        self.backupProgressBar = self.getLoadingBar()

        self.frameViews = self.newFrameViews()
        self.setRadarViewAxes(self._defaultRadarConfig, self.frameViews)
        self.addFrameViews(self.frameLayout, self.frameViews)
        self.frameManager = FramePlotManager(self.frameViews, self._cmap)

        QThread.currentThread().setPriority(QThread.Priority.TimeCriticalPriority)

        self.actionNew.triggered.connect(self.openNewProjectDialog)
        self.actionOpen.triggered.connect(self.openProject)
        self.actionOpenPrediction.triggered.connect(self.loadPrediction)
        self.actionBackupGT.triggered.connect(self.backupGroundTruth)
        self.actionReorderIds.triggered.connect(self.reorderIds)

        self.actionOpenPrediction.setEnabled(False)
        self.actionBackupGT.setEnabled(False)
        self.actionReorderIds.setEnabled(False)

        self.nextFrameButton.clicked.connect(self.nextFrame)
        self.prevFrameButton.clicked.connect(self.prevFrame)
        self.nextSequenceButton.clicked.connect(self.nextSequence)
        self.prevSequenceButton.clicked.connect(self.prevSequence)

        self.setupShortcuts()

        self.timelineSlider.valueChanged.connect(self.timelineChanged)
        self.timelineSlider.sliderReleased.connect(self.timelineReleased)

        self.labelSingleButton.clicked.connect(self.labelInstance)
        self.labelAllButton.clicked.connect(lambda: self.labelInstances(None, None))
        self.labelFollowingButton.clicked.connect(self.labelInstancesFollowingFrames)
        self.labelPreviousButton.clicked.connect(self.labelInstancesPreviousFrames)

        self.idSpinBox.valueChanged.connect(self.editingId)
        self.classComboBox.currentTextChanged.connect(self.editingCls)

        # Load default project if it exists
        defaultProjectPath = os.path.join('.', 'projects', 'Default', 'Default.json')
        if os.path.exists(defaultProjectPath):
            logger.info('Loading default project.')
            project = Project.load(defaultProjectPath)
            self.loadProject(project)

    @property
    def activeSequence(self):
        return self.sequenceModel.sequence(self._activeSequenceRow)

    @property
    def activeFrame(self) -> int:
        return self._activeFrame

    @property
    def selectedInstance(self):
        return self.getInstance(self._selectedInstanceIndex)

    def getInstance(self, index: int):
        if index < 0 or index >= len(self._instances):
            return None

        return self._instances[index]

    def setupShortcuts(self):
        # Next Frame
        self.nextFrameShortcut1 = QShortcut(QKeySequence('Right'), self)
        self.nextFrameShortcut1.activated.connect(self.nextFrame)
        self.nextFrameShortcut2 = QShortcut(QKeySequence('D'), self)
        self.nextFrameShortcut2.activated.connect(self.nextFrame)

        # Previous Frame
        self.prevFrameShortcut1 = QShortcut(QKeySequence('Left'), self)
        self.prevFrameShortcut1.activated.connect(self.prevFrame)
        self.prevFrameShortcut2 = QShortcut(QKeySequence('A'), self)
        self.prevFrameShortcut2.activated.connect(self.prevFrame)

        # Start of sequence
        self.startFrameShortcut1 = QShortcut(QKeySequence('Home'), self)
        self.startFrameShortcut1.activated.connect(lambda: self.setFrame(1))
        self.startFrameShortcut2 = QShortcut(QKeySequence('Q'), self)
        self.startFrameShortcut2.activated.connect(lambda: self.setFrame(1))

        # End of sequence
        self.endFrameShortcut1 = QShortcut(QKeySequence('End'), self)
        self.endFrameShortcut1.activated.connect(lambda: self.setFrame(len(self.activeSequence)))
        self.endFrameShortcut2 = QShortcut(QKeySequence('E'), self)
        self.endFrameShortcut2.activated.connect(lambda: self.setFrame(len(self.activeSequence)))

        # Next sequence
        self.nextSequenceShortcut1 = QShortcut(QKeySequence('Down'), self)
        self.nextSequenceShortcut1.activated.connect(self.nextSequence)
        self.nextSequenceShortcut2 = QShortcut(QKeySequence('S'), self)
        self.nextSequenceShortcut2.activated.connect(self.nextSequence)

        # Previous sequence
        self.prevSequenceShortcut1 = QShortcut(QKeySequence('Up'), self)
        self.prevSequenceShortcut1.activated.connect(self.prevSequence)
        self.prevSequenceShortcut2 = QShortcut(QKeySequence('W'), self)
        self.prevSequenceShortcut2.activated.connect(self.prevSequence)

        # Cycle bounding boxes
        self.nextBoundingBoxShortcut = QShortcut(QKeySequence('F'), self)
        self.nextBoundingBoxShortcut.activated.connect(self.nextBoundingBox)

        self.prevBoundingBoxShortcut = QShortcut(QKeySequence('Ctrl+F'), self)
        self.prevBoundingBoxShortcut.activated.connect(self.prevBoundingBox)

        self.noBoundingBoxShortcut = QShortcut(QKeySequence('R'), self)
        self.noBoundingBoxShortcut.activated.connect(lambda: self.selectInstance(-1))

        # Shortcuts acceptable when focus is in ID edit field
        self.idSpinBox.shortcuts = [
            self.nextFrameShortcut2,
            self.prevFrameShortcut2,
            self.nextSequenceShortcut2,
            self.prevSequenceShortcut2,
            self.startFrameShortcut2,
            self.endFrameShortcut2,
            self.nextBoundingBoxShortcut,
            self.prevBoundingBoxShortcut,
            self.noBoundingBoxShortcut,
        ]

        self.idSpinBox.pressedReturn.connect(self.labelSingleButton.click)
        self.idSpinBox.pressedCtrlReturn.connect(self.labelAllButton.click)
        self.idSpinBox.pressedAltReturn.connect(self.labelFollowingButton.click)
        self.idSpinBox.pressedCtrlAltReturn.connect(self.labelPreviousButton.click)

    def getLoadingBar(self):
        loadingBar = QProgressBar()
        loadingBar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loadingBar.setTextVisible(True)
        loadingBar.setMinimum(0)
        loadingBar.setValue(0)
        loadingBar.setMaximumHeight(20)
        loadingBarFont = QFont(QApplication.font())
        loadingBarFont.setPointSize(8)
        loadingBar.setFont(loadingBarFont)

        return loadingBar

    def closeEvent(self, event):
        logger.debug('Closing app...')

        self.datasetLoader.stop()
        event.accept()

    def datasetPreloadingStarted(self, total):
        self._datasetPreloadingStart = time.time()
        self.loadingProjectBar.setMaximum(total)
        self.loadingProjectBar.setFormat(f'Preloading {0:5d}/{total:5d} frames...')

    def datasetPreloadingLoadedGt(self):
        self._allowEditAllGt = True
        self.actionReorderIds.setEnabled(True)
        logger.debug(f'Finished loading GT: {(time.time() - self._datasetPreloadingStart):.3f}s')

    def datasetPreloadingProgress(self, progress):
        self._preloadingMutex.lock()
        self.loadingProjectBar.setValue(progress)
        max = self.loadingProjectBar.maximum()
        self.loadingProjectBar.setFormat(f'Preloading {progress:5d}/{max:5d} frames...')
        self._preloadingMutex.unlock()

    def datasetPreloadingFinished(self):
        self.statusBar().removeWidget(self.loadingProjectBar)
        self.statusBar().showMessage('All frames preloaded and cached.', 10000)
        logger.info(f'Finished preloading dataset: {(time.time() - self._datasetPreloadingStart):.3f}s')

    def backupStarted(self, total):
        self.backupProgressBar.setMaximum(total)
        self.backupProgressBar.setFormat(f'Backing up {0:5d}/{total:5d} GT files...')

    def backupProgress(self, progress):
        self._backupMutex.lock()
        self.backupProgressBar.setValue(progress)
        max = self.backupProgressBar.maximum()
        self.backupProgressBar.setFormat(f'Backing up {progress:5d}/{max:5d} GT files...')
        self._backupMutex.unlock()

    def backupFinished(self):
        self.statusBar().removeWidget(self.backupProgressBar)
        self.statusBar().showMessage('Ground truth files backed up.', 10000)
        self.actionBackupGT.setEnabled(True)
        logger.info(f'Finished backing up ground truth files.')

    def addFrameViews(self, parentLayout, views):
        hLayout = QHBoxLayout()
        hLayout.addWidget(views['stereo'], 2)
        hLayout.addWidget(views['rd'], 1)
        vLayout = QVBoxLayout()
        vLayout.addWidget(views['ra'], 1)
        vLayout.addWidget(views['cart'], 1)
        parentLayout.addLayout(hLayout, 1)
        parentLayout.addLayout(vLayout, 1)

    def newFrameViews(self):
        stereo = pg.ImageView()
        stereo.ui.histogram.hide()
        stereo.ui.roiBtn.hide()
        stereo.ui.menuBtn.hide()

        rd = self.newRadarView()
        ra = self.newRadarView()
        cart = self.newRadarView()

        return {
            'stereo': stereo,
            'rd': rd,
            'ra': ra,
            'cart': cart
        }

    def getViewAxis(self, xlabel, ylabel, xticks, yticks):
        rdAxes = {
            'bottom': pg.AxisItem('bottom'),
            'left': pg.AxisItem('left'),
        }

        rdAxes['bottom'].setTicks([xticks])
        rdAxes['bottom'].setLabel(xlabel)
        rdAxes['left'].setTicks([yticks])
        rdAxes['left'].setLabel(ylabel)

        return rdAxes

    def setRadarViewAxes(self, cfg, views):
        maxV = ((cfg['doppler_size'] * cfg['velocity_resolution']) / 2) * 3.6 # km/h
        maxAngle = np.degrees(np.arcsin((cfg['azimuth_size'] * 2 * np.pi / cfg['azimuth_size'] - np.pi) / (np.pi * cfg['config_frequency'] / cfg['designed_frequency']))) # °

        rangeTicks = self.generateTicks(0, cfg['range_size'], cfg['maximum_range'], 0, 11)
        rdTicks = self.generateTicks(0, cfg['doppler_size'], -maxV, maxV, 5)
        raTicks = self.generateTicks(0, cfg['azimuth_size'], -maxAngle, maxAngle, 9)
        cartTicks = self.generateTicks(0, cfg['range_size'] * 2, -cfg['maximum_range'], cfg['maximum_range'], 11)

        views['rd'].radarPlot.setAxisItems(self.getViewAxis('Doppler-Velocity (km/h)', 'Range (m)', rdTicks, rangeTicks))
        views['ra'].radarPlot.setAxisItems(self.getViewAxis('Angle (°)', 'Range (m)', raTicks, rangeTicks))
        views['cart'].radarPlot.setAxisItems(self.getViewAxis('x (m)', 'z (m)', cartTicks, rangeTicks))

    def newRadarView(self):
        viewBox = pg.ViewBox()
        # TODO: Limit zoom
        # viewBox.setLimits(xMin=xticks[0][0], xMax=xticks[-1][0], yMin=yticks[0][0], yMax=yticks[-1][0])
        # viewBox.setLimits(xMin=xticks[-1][0] * -1, xMax=xticks[-1][0] * 2, yMin=yticks[-1][0] * -1, yMax=yticks[-1][0] * 2)

        plot = RadarPlotItem(viewBox=viewBox)
        view = RadarImageView(plot)

        return view

    def openNewProjectDialog(self):
        self.newProjectDialog.exec()

    def createNewProject(self):
        project_dir = os.path.join(self.newProjectDialog.project_dir, self.newProjectDialog.project_name)
        project = Project(self.newProjectDialog.project_name, project_dir, self.newProjectDialog.dataset_dir)
        project.save()

        logger.info(f'Created new project {self.newProjectDialog.project_name}')
        self.statusBar().showMessage(f'Created new project {self.newProjectDialog.project_name}', 5000)

        self.newProjectDialog.clear()

        self.loadProject(project)
        self.backupGroundTruth()

    def openProject(self):
        (projectFile, _) = QFileDialog.getOpenFileName(self, "Select project", filter='Projects (*.json)')

        if not os.path.exists(projectFile):
            return

        project = Project.load(projectFile)
        self.loadProject(project)

    def loadProject(self, project: Project):
        start_time = time.time()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        self.project = project
        self.setWindowTitle(self.project._name)

        self.dataset = RADTrack(project.dataset_dir, project.cache_dir)

        self._allowEditAllGt = False
        self.actionReorderIds.setEnabled(False)

        self.actionOpenPrediction.setEnabled(True)
        self.actionBackupGT.setEnabled(True)

        self.sequenceModel = SequenceModel(self, self.dataset)
        self.sequenceView.setModel(self.sequenceModel)
        self.setupSequenceView()
        selectionModel = self.sequenceView.selectionModel()
        selectionModel.selectionChanged.connect(self.sequenceChanged)

        self.classComboBox.clear()
        self.classComboBox.addItems(self.dataset.ALL_CLASSES)
        self.classComboBox.setCurrentIndex(-1)
        self.classComboBox.setEnabled(False)

        # Set axis ticks and labels for current dataset
        self.setRadarViewAxes(self.dataset.radar_config, self.frameViews)

        QApplication.restoreOverrideCursor()

        self.preload()

        self.statusBar().showMessage(f'Loaded project {self.project._name}', 5000)
        logger.info(f'Loaded project {self.project._name}: {(time.time() - start_time):.3f}s')

    def preload(self):
        if self.dataset is None:
            return

        self.datasetLoader = DatasetLoader(self.dataset)

        self.statusBar().addWidget(self.loadingProjectBar, 1)
        self.datasetLoader.signals.started.connect(self.datasetPreloadingStarted)
        self.datasetLoader.signals.loadedGt.connect(self.datasetPreloadingLoadedGt)
        self.datasetLoader.signals.progress.connect(self.datasetPreloadingProgress)
        self.datasetLoader.signals.finished.connect(self.datasetPreloadingFinished)
        QThreadPool.globalInstance().start(self.datasetLoader, 1)

    def backupGroundTruth(self):
        if self.dataset is None:
            return

        self.actionBackupGT.setEnabled(False)

        self.backup = Backup(self.dataset, self.project.backup_dir, self.project.max_backups)

        self.statusBar().addWidget(self.backupProgressBar, 1)
        self.backup.signals.startedCleaning.connect(lambda: self.statusBar().showMessage('Cleaning up old backups...', 10000))
        self.backup.signals.started.connect(self.backupStarted)
        self.backup.signals.progress.connect(self.backupProgress)
        self.backup.signals.finished.connect(self.backupFinished)
        QThreadPool.globalInstance().start(self.backup, 10)


    def setupSequenceView(self):
        self.sequenceView.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sequenceView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sequenceView.setShowGrid(False)
        hheader = self.sequenceView.horizontalHeader()
        hheader.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hheader.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        vheader = self.sequenceView.verticalHeader()
        vheader.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        vheader.setDefaultSectionSize(24)

        self.sequenceView.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.setSequence(0)

    def sequenceChanged(self, selected: QItemSelection, deselected: QItemSelection):
        indexes = self.sequenceView.selectionModel().selectedRows()
        if len(indexes) == 0:
            return

        row = indexes[0].row()
        self.setSequence(row)

    def setFrame(self, frame: int, force=False):
        if self.activeSequence is None:
            return

        # Clamp frame
        seqlen = len(self.activeSequence)
        frame = self.clampFrame(frame, seqlen)

        if not force and self.activeFrame == frame:
            return

        start = time.time()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        self.timelineSlider.setMaximum(seqlen)
        self.timelineSlider.setValue(frame)

        self.totalFramesEdit.setText(str(seqlen))

        self.setLabelInstance(None)

        # Remove all clicked event handlers on instances
        for instance in self._instances:
            try:
                instance.clicked.disconnect()
            except Exception:
                pass

        images, instances, predictions = self.frameManager.loadFrame(self.activeSequence, frame)
        self.frameManager.drawFrame(images, instances + predictions)

        # This determines tab order. Currently the tab order is range far -> near (top -> down)
        # The top of the bounding box is considered
        self._instances = sorted(instances, key=lambda instance: instance.bbox3d[0] - instance.bbox3d[3] / 2)
        self._selectedInstanceIndex = -1

        # Add clicked event handlers
        for instance in self._instances:
            instance.clicked.connect(self.radarInstanceClicked)

        self._lastActiveFrame[self._activeSequenceRow] = frame

        self._activeFrame = frame
        self.curFrameEdit.setText(str(frame))

        QApplication.restoreOverrideCursor()
        logger.debug(f'Setting frame {frame}: {(time.time() - start):.3f}s')

    def setSequence(self, row: int):
        row = util.clamp(row, 0, self.sequenceModel.rowCount(None) - 1)

        if self._activeSequenceRow == row:
            return

        self._activeSequenceRow = row

        index = self.sequenceModel.index(row, 0)
        self.sequenceView.setCurrentIndex(index)

        logger.debug(f'Setting sequence {self.activeSequence}')

        # Retrieve frame where user left
        lastActiveFrame = self._lastActiveFrame.get(row, 1)

        self.setFrame(lastActiveFrame, force=True)

    def nextFrame(self):
        self.setFrame(self.activeFrame + 1)

    def nextSequence(self):
        logger.debug('Next Seq')
        self.setSequence(self._activeSequenceRow + 1)

    def prevFrame(self):
        logger.debug('Prev Frame')
        self.setFrame(self.activeFrame - 1)

    def prevSequence(self):
        logger.debug('Prev Seq')
        self.setSequence(self._activeSequenceRow - 1)

    def timelineChanged(self, value):
        logger.debug(f'Timeline changed: {value}')
        if abs(int(self.curFrameEdit.text()) - value) > 1:
            self.setFrame(value)
        else:
            self.curFrameEdit.setText(str(value))

    def timelineReleased(self):
        logger.debug('Timeline released')
        self.setFrame(int(self.curFrameEdit.text()))

    def radarInstanceClicked(self):
        instance = self.sender()
        try:
            self._selectedInstanceIndex = self._instances.index(instance)
            self.selectInstance(self._selectedInstanceIndex)
        except ValueError:
            pass

    def nextBoundingBox(self):
        if len(self._instances) == 0:
            self._selectedInstanceIndex = -1
            return

        self._selectedInstanceIndex = (self._selectedInstanceIndex + 1) % len(self._instances)
        self.selectInstance(self._selectedInstanceIndex)

    def prevBoundingBox(self):
        if len(self._instances) == 0:
            self._selectedInstanceIndex = -1
            return

        self._selectedInstanceIndex = (self._selectedInstanceIndex - 1) % len(self._instances)
        self.selectInstance(self._selectedInstanceIndex)

    def selectInstance(self, index: int):
        """Highlights the instance at `index` in the `_instances` list.

        Args:
            index (int): Index of the instance. Values less than zero results in lifting any selection.
        """
        for i, instance in enumerate(self._instances):
            instance.resetAnnotation()
            if i == index:
                instance.select()
            else:
                instance.deselect()

        selected = self.getInstance(index)
        self.setLabelInstance(selected)

    def setLabelInstance(self, instance: RadarInstance):
        if instance is None:
            self.idSpinBox.setEnabled(False)
            self.idSpinBox.setValue(-1)
            self.classComboBox.setEnabled(False)
            self.classComboBox.setCurrentIndex(-1)
            self.labelSingleButton.setEnabled(False)
            self.labelAllButton.setEnabled(False)
            self.labelFollowingButton.setEnabled(False)
            self.labelPreviousButton.setEnabled(False)
        else:
            self.idSpinBox.setEnabled(True)
            self.idSpinBox.setValue(instance.id)
            self.idSpinBox.setFocus()
            self.idSpinBox.selectAll()
            self.classComboBox.setEnabled(True)
            self.classComboBox.setCurrentText(instance.cls)
            self.labelSingleButton.setEnabled(True)
            self.labelAllButton.setEnabled(self._allowEditAllGt)
            self.labelFollowingButton.setEnabled(self._allowEditAllGt)
            self.labelPreviousButton.setEnabled(self._allowEditAllGt)

    def editingId(self, value: int):
        if self.selectedInstance is None or not value:
            return

        cls = self.classComboBox.currentText()
        if value != self.selectedInstance.id:
            self.selectedInstance.displayAnnotation(value, cls, editing=True)

    def editingCls(self, value: str):
        if self.selectedInstance is None or not value:
            return

        id = self.idSpinBox.value()
        if value != self.selectedInstance.cls:
            self.selectedInstance.displayAnnotation(id, value, editing=True)

    def labelInstance(self):
        if self.selectedInstance is None or self.activeSequence is None:
            return

        start = time.time()

        oldId = self.selectedInstance.id
        newId = self.idSpinBox.value()

        oldCls = self.selectedInstance.cls
        newCls = self.classComboBox.currentText()

        # Nothing to change
        if oldId == newId and oldCls == newCls:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        changes: list[str] = []
        if oldId != newId:
            self.selectedInstance.id = newId
            self.activeSequence.gt[self.activeFrame - 1]['ids'][self.selectedInstance.frameId] = newId
            changes.append(f'{oldId} -> {newId}')

        if oldCls != newCls:
            self.selectedInstance.cls = newCls
            self.activeSequence.gt[self.activeFrame - 1]['classes'][self.selectedInstance.frameId] = newCls
            changes.append(f'{oldCls} -> {newCls}')

        self.activeSequence.save_gt()

        QApplication.restoreOverrideCursor()
        logger.info(f'Saved changed instance {" | ".join(changes)}: {(time.time() - start) * 1000:.0f}ms')

    def labelInstancesFollowingFrames(self):
        self.labelInstances(afterFrame=self.activeFrame)

    def labelInstancesPreviousFrames(self):
        self.labelInstances(beforeFrame=self.activeFrame)

    def labelInstances(self, afterFrame: int | None = None, beforeFrame: int | None = None):
        if not self._allowEditAllGt or self.selectedInstance is None or self.activeSequence is None:
            return

        start1 = time.time()

        oldId = self.selectedInstance.id
        newId = self.idSpinBox.value()

        # Nothing to change
        if oldId == newId:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # Include the active frame
        startFrame = 1
        if afterFrame is not None:
            startFrame = afterFrame
            afterFrame -= 1

        # Find all gt objects where object with old id is located
        affectedFrames, duplicates = self.findAffectedAndDuplicateFrames(self.activeSequence.gt[afterFrame:beforeFrame],
                                                                         oldId, newId, startFrame)

        end1 = time.time()

        if len(duplicates) > 0:
            QApplication.restoreOverrideCursor()

            dupFrames = util.numlist2str(duplicates)
            qm = QMessageBox.warning(self, 'Duplicate ID', (f'Same ID already exists in frames: {dupFrames}.\n'
                                                            'Would you still like to overwrite ID?\n'
                                                            'This would lead to objects with same IDs in a frame.'),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

            if qm != QMessageBox.StandardButton.Yes:
                logger.info('Canceled labeling due to duplicate IDs.')
                return

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        start2 = time.time()

        # Replace id of all affected GT objects/frames by reference
        for gt in affectedFrames:
            gt['ids'][:] = [newId if id == oldId else id for id in gt['ids']]
        self.selectedInstance.id = newId

        self.activeSequence.save_gt()

        end2 = time.time()

        QApplication.restoreOverrideCursor()
        execTimeMs = ((end1 - start1) + (end2 - start2)) * 1000
        if afterFrame is not None and beforeFrame is not None:
            logger.info(f'Saved all changed instances {oldId} -> {newId} between frames {beforeFrame}-{afterFrame}: {execTimeMs:.0f}ms')
        elif afterFrame is not None:
            logger.info(f'Saved all changed instances {oldId} -> {newId} after frame {afterFrame}: {execTimeMs:.0f}ms')
        elif beforeFrame is not None:
            logger.info(f'Saved all changed instances {oldId} -> {newId} before frame {beforeFrame}: {execTimeMs:.0f}ms')
        else:
            logger.info(f'Saved all changed instances {oldId} -> {newId}: {execTimeMs:.0f}ms')

    def findAffectedAndDuplicateFrames(self, gt: list[dict[str]], oldId: int, newId: int, startFrame: int = 1):
        affectedFrames: list[dict[str]] = []
        duplicates: list[int] = []
        for t, frameGt in enumerate(gt, start=startFrame):
            if oldId not in frameGt['ids']:
                continue

            # Check for duplicate id in this frame
            if newId in frameGt['ids']:
                duplicates.append(t)

            affectedFrames.append(frameGt)

        return affectedFrames, duplicates

    def reorderIds(self):
        if not self._allowEditAllGt or self.activeSequence is None:
            return

        nTracks: int = 0
        seenIdMap: dict[int, int] = {}

        start = time.time()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        for gt in self.activeSequence.gt:
            ids = gt['ids']
            newIds: list[int] = []

            for id in ids:
                continuousId = seenIdMap.get(id, -1)
                if continuousId < 0:
                    nTracks += 1
                    continuousId = nTracks
                    seenIdMap[id] = continuousId

                newIds.append(continuousId)

            assert len(gt['ids']) == len(newIds)
            gt['ids'] = newIds

        self.activeSequence.save_gt()

        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage('Reordered IDs to continuous numbers.', 10000)
        logger.info(f'Reordered IDs to continuous numbers: {(time.time() - start):.3f}s')


    def loadPrediction(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select file with predictions", filter="JSON files (*.json);;All files (*)")
        if not file:
            return

        self.activeSequence.load_prediction(file)
        # Reload view
        self.setFrame(self.activeFrame, True)

    @staticmethod
    def clampFrame(frame: int, sequenceLength: int = -1):
        if sequenceLength > 0:
            return util.clamp(frame, 1, sequenceLength)
        else:
            return max(1, frame)

    @staticmethod
    def generateTicks(minValue: float, maxValue: float, minTick: float, maxTick: float, num: int) -> list[tuple[int, str]]:
        valueRange = np.linspace(minValue, maxValue, num)
        if minTick > maxTick:
            tickRange = np.flip(np.linspace(maxTick, minTick, num))
        else:
            tickRange = np.linspace(minTick, maxTick, num)

        tickRange = [f'{t:.2f}'.rstrip('0').rstrip('.') for t in tickRange]

        return list(zip(valueRange, tickRange))


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        uic.loadUi('new_project_gui.ui', self)

        self.browseProjectDirectoryButton.clicked.connect(self.browseProjectDir)
        self.browseDatasetDirectoryButton.clicked.connect(self.browseDatasetDir)

        self.clear()

    def browseProjectDir(self):
        dir = QFileDialog.getExistingDirectory(self, "Select project directory")
        reldir = os.path.relpath(dir)
        self.projectDirectoryEdit.setText(reldir)

    def browseDatasetDir(self):
        dir = QFileDialog.getExistingDirectory(self, "Select dataset")
        reldir = os.path.relpath(dir)
        self.datasetPathEdit.setText(reldir)

    def clear(self):
        self.projectNameEdit.setText('')
        self.projectDirectoryEdit.setText('')
        self.datasetPathEdit.setText('')

    @property
    def project_name(self) -> str:
        return self.projectNameEdit.text()

    @property
    def project_dir(self) -> str:
        return self.projectDirectoryEdit.text()

    @property
    def dataset_dir(self) -> str:
        return self.datasetPathEdit.text()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())
