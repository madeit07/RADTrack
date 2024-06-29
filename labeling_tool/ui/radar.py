import pyqtgraph as pg

from ui.boundingbox import BoundingBox


class RadarPlotItem(pg.PlotItem):
    def __init__(self, parent=None, name=None, labels=None, title=None, viewBox=None, axisItems=None, enableMenu=True, **kargs):
        super().__init__(parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs)

        self.boundingBoxes: list[BoundingBox] = []

    def addItem(self, item, *args, **kargs):
        if not isinstance(item, BoundingBox):
            super().addItem(item, *args, **kargs)
            return

        self.boundingBoxes.append(item)

        for i in item.getItems():
            super().addItem(i, *args, **kargs)

    def removeItem(self, item):
        if not isinstance(item, BoundingBox):
            super().removeItem(item)
            return

        for i in item.getItems():
            super().removeItem(i)

        self.boundingBoxes.remove(item)

    def clearBoundingBoxes(self):
        # Iterate in reverse because we remove elements from the list
        for i in range(len(self.boundingBoxes) - 1, -1, -1):
            self.removeItem(self.boundingBoxes[i])


class RadarImageView(pg.ImageView):
    def __init__(self, radarPlot: RadarPlotItem, parent=None, name="RadarImageView", imageItem=None, levelMode='mono', discreteTimeLine=False, roi=None, normRoi=None, *args):
        super().__init__(parent, name, radarPlot, imageItem, levelMode, discreteTimeLine, roi, normRoi, *args)

        self.radarPlot = radarPlot

        cm = pg.colormap.get('viridis')
        self.setColorMap(cm)
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
