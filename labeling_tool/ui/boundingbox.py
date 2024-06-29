from typing import NamedTuple

from PyQt6.QtCore import QObject, pyqtSignal, QRectF
from PyQt6.QtGui import QPicture, QPainter, QColor, QPen
import pyqtgraph as pg
import numpy as np

from labeler import util

ICON_MAPPING = {
    'person': '&#128694;',
    'truck': '&#128666;',
    'car': '&#128663;',
    'bus': '&#128652;',
    'motorcycle': '&#127949;',
    'bicycle': '&#128690;',
    '': '?',
}

class BBox(NamedTuple):
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @classmethod
    def from_numpy(cls, ycxchw: np.ndarray):
        yc, xc, h, w = ycxchw
        return cls(int(xc - w / 2), int(yc - h / 2), int(w), int(h))


class RadarInstance(QObject):
    clicked = pyqtSignal()

    def __init__(self, frameId: int, id: int, cls: str, bbox3d: np.ndarray, cartBox: np.ndarray, shapes: dict[str, tuple[int, int]],
                 is_prediction: bool = False, **kwargs):
        super().__init__()

        # This is the internal index in the gt array structures uniquely identifying this instance in the current frame.
        self._frameId = frameId

        # Global id unique across all frames in the sequence.
        self._id = id
        self._cls = cls
        self._bbox3d = bbox3d
        self._cartBox = cartBox
        self._shapes = shapes
        self._is_prediction = is_prediction

        rdBox = BBox.from_numpy(np.array([bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]]))
        raBox = BBox.from_numpy(np.array([bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]]))
        cartBBox = BBox.from_numpy(cartBox)

        self._boundingBoxes: dict[str, BoundingBox] = {}
        if rdBox.w * rdBox.h > 0:
            self._boundingBoxes['rd'] = BoundingBox(self, rdBox, shape=shapes['rd'], symmetrical=True, **kwargs)
        if raBox.w * raBox.h > 0:
            self._boundingBoxes['ra'] = BoundingBox(self, raBox, shape=shapes['ra'], symmetrical=True, **kwargs)
        if cartBBox.w * cartBBox.h > 0:
            self._boundingBoxes['cart'] = BoundingBox(self, cartBBox, shape=shapes['cart'], **kwargs)

        for bb in self.boundingBoxes.values():
            if not self.is_prediction:
                bb._boxItem.clicked.connect(self._bbClicked)
            bb.setAnnotation(id, cls, prediction=is_prediction)

    def _bbClicked(self):
        self.clicked.emit()

    @property
    def frameId(self):
        return self._frameId

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value
        self.displayAnnotation(self._id, self._cls)

    @property
    def cls(self):
        return self._cls

    @cls.setter
    def cls(self, value: str):
        self._cls = value
        self.displayAnnotation(self._id, self._cls)

    @property
    def bbox3d(self):
        return self._bbox3d

    @property
    def cartBox(self):
        return self._cartBox

    @property
    def boundingBoxes(self):
        return self._boundingBoxes

    @property
    def is_prediction(self):
        return self._is_prediction

    def select(self):
        for bb in self.boundingBoxes.values():
            bb._boxItem._select()

    def deselect(self):
        for bb in self.boundingBoxes.values():
            bb._boxItem._deselect()

    def displayAnnotation(self, id: int, cls: str, editing: bool = False):
        """Sets the ID of the instance and updates the annotation in the UI

        Args:
            id (int): New ID to display.
            id (str): New class to display.
            editing (bool, optional): False to indicate that the ID is saved. True to display the ID as currently being edited. Defaults to False.
        """
        for bb in self.boundingBoxes.values():
            bb.setAnnotation(id, cls, editing)

    def resetAnnotation(self):
        for bb in self.boundingBoxes.values():
            bb.setAnnotation(self.id, self.cls)


class BoundingBox():
    def __init__(self, instance: RadarInstance, box: BBox, shape: tuple[int, int], symmetrical: bool = False,
                 color: QColor = 'white', annotation_anchor: tuple [int, int] = (0, 1), **kwargs):
        self.instance = instance
        self.box = box
        self.shape = shape
        self.symmetrical = symmetrical
        self.color = color

        self._boxItem = BoundingBoxItem(self.box, self.shape, self.symmetrical, self.color, **kwargs)

        fill = QColor(color)
        fill.setAlphaF(0.5)
        self._annotationItem = pg.TextItem(html='', anchor=annotation_anchor, fill=fill)
        self.setAnnotation(-1, 'person')

        aa_x, aa_y = annotation_anchor
        if aa_x == 1 and aa_y == 0:
            xtext = shape[1] if (self.box.x + self.box.w) >= shape[1] else (self.box.x + self.box.w)
            ytext = self.box.y
        else:
            xtext = 0 if self.box.x <= 0 else self.box.x
            ytext = self.box.y

        self._annotationItem.setPos(xtext, ytext)

    def getItems(self):
        return [self._boxItem, self._annotationItem]

    def setAnnotation(self, id: int, cls: str, editing: bool = False, prediction: bool = False):
        css = [
            'font-size: 14pt;',
            'color: #ffffff',
        ]
        idCss = []

        if prediction:
            idCss.extend([
                'font-weight: bold;',
            ])
        else:
            if editing:
                idCss.extend([
                    'font-weight: bold;',
                    'font-style: italic;',
                ])

        style = ' '.join(css)
        idStyle = ' '.join(idCss)
        html = f'<span style="{style}"><span style="{idStyle}">{id}</span> {ICON_MAPPING[cls]}</span>'
        self._annotationItem.setHtml(html)


class BoundingBoxItem(pg.GraphicsObject):
    clicked = pyqtSignal()

    def __init__(self, box: BBox, shape: tuple[int, int], symmetrical=False,
                 color='white', linewidth=4, dashed=False, fill=None, fillAlpha: float = 0.5, parent=None):
        super().__init__(parent)

        self.box = box
        self.shape = shape
        self.symmetrical = symmetrical
        self.color = color

        self.highlightColor = 'white'
        self.highlightFillColor = QColor('white')
        self.highlightFillColor.setAlphaF(0.5)

        self.fillColor = None
        if fill:
            self.fillColor = QColor(fill)
            self.fillColor.setAlphaF(fillAlpha)

        self.linewidth = linewidth
        self.dashed = dashed
        self.selected = False

        self.picture = QPicture()
        self._generate_picture()


    def _generate_picture(self, highlight=False):
        painter = QPainter(self.picture)
        if highlight:
            lineColor = self.highlightColor
        else:
            lineColor = self.color
        pen: QPen = pg.mkPen(lineColor, width=self.linewidth)

        if self.dashed:
            pen.setDashPattern([2, 4])

        painter.setPen(pen)

        fill = None
        if highlight:
            fill = self.highlightFillColor
        elif self.fillColor:
            fill = self.fillColor

        if fill:
            painter.setBrush(pg.mkBrush(fill))

        x, y, w, h = self.box
        ymax, xmax = self.shape

        if self.symmetrical:
            if x < 0:
                painter.drawRect(0, y, w + x, h)
                painter.drawRect(xmax + x, y, -x, h)
            elif (x + w) >= xmax:
                painter.drawRect(x, y, xmax - x, h)
                painter.drawRect(0, y, (x + w) - xmax, h)
            else:
                painter.drawRect(x, y, w, h)
        else:
            if xmax > 0:
                xclamp = util.clamp(x, 0, xmax)
            else:
                xclamp = x
            if ymax > 0:
                yclamp = util.clamp(y, 0, ymax)
            else:
                yclamp = y
            wclamp = util.clamp(w, 0, xmax - xclamp)
            hclamp = util.clamp(h, 0, ymax - yclamp)

            painter.drawRect(xclamp, yclamp, wclamp, hclamp)


        painter.end()

    def paint(self, painter: QPainter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())

    def _select(self):
        if self.selected:
            return

        self._generate_picture(highlight=True)
        self.update()
        self.selected = True

    def _deselect(self):
        if not self.selected:
            return

        self._generate_picture()
        self.update()
        self.selected = False

    def mousePressEvent(self, event):
        self.clicked.emit()
        # super().mousePressEvent(event)
