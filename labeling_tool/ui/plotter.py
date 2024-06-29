import time
import logging

from PyQt6.QtGui import QColor, QColorConstants
import pyqtgraph as pg
import numpy as np

from labeler.dataloader import RADTrackSequence
from ui.boundingbox import BoundingBox, RadarInstance
from ui.radar import RadarImageView

logger = logging.getLogger(__name__)

def getXMax(image: np.ndarray):
    if image is None:
        return 0

    return image.shape[0]

class FramePlotManager():
    def __init__(self, views: dict[str, pg.ImageView | RadarImageView], cmap: list[tuple]):
        self.views = views
        self.cmap = cmap

    def loadFrame(self, sequence: RADTrackSequence, frame: int):
        start = time.time()

        gt, pred, image, rd, ra, cart = sequence[frame]

        images: dict[str, np.ndarray | None] = {
            'stereo': image,
            'rd': rd,
            'ra': ra,
            'cart': cart
        }

        instances = self.getInstances(gt, rd, ra, cart)
        predictions = self.getInstances(pred, rd, ra, cart, is_prediction=True)

        logger.debug(f'Loading frame: {(time.time() - start) * 1000:.0f}ms ({len(instances)} radar objects)')

        return images, instances, predictions

    def drawFrame(self, images: dict[str, np.ndarray | None], instances: list[RadarInstance]):
        start = time.time()

        for name, view in self.views.items():
            image = images.get(name, None)
            boxes = [instance.boundingBoxes[name] for instance in instances if name in instance.boundingBoxes]
            self.updatePlot(view, image, boxes)

        logger.debug(f'Drawing frame: {(time.time() - start) * 1000:.0f}ms')

    def updatePlot(self, imgView: pg.ImageView | RadarImageView, image: np.ndarray | None, boxes: list[BoundingBox]):
        if image is None:
            imgView.clear()
        else:
            imgView.setImage(image, autoRange=True)

        if isinstance(imgView, RadarImageView):
            imgView.radarPlot.clearBoundingBoxes()

            if image is not None:
                for box in boxes:
                    imgView.radarPlot.addItem(box)

    def getInstances(self, tracks: list[dict[str]], rdImage, raImage, cartImage, is_prediction: bool = False):
        instances: list[RadarInstance] = []

        if not tracks:
            return instances

        shapes = {
            'rd': rdImage.shape if rdImage is not None else (0, 0),
            'ra': raImage.shape if raImage is not None else (0, 0),
            'cart': cartImage.shape if cartImage is not None else (0, 0),
        }

        for i in range(len(tracks['classes'])):
            cls = tracks['classes'][i]
            bbox3d = np.array(tracks['boxes'][i])
            cartBox = np.array(tracks['cart_boxes'][i])
            ids = tracks.get('ids', [])
            id = ids[i] if i < len(ids) else -1

            kwargs: dict[str] = {}
            if is_prediction:
                kwargs['color'] = QColor.fromRgbF(*self.cmap[id % len(self.cmap)])
                kwargs['dashed'] = True
                kwargs['fill'] = QColorConstants.White
                kwargs['annotation_anchor'] = (1, 0)
            else:
                kwargs['color'] = QColor.fromRgbF(*self.cmap[id % len(self.cmap)])


            instances.append(RadarInstance(i, id, cls, bbox3d, cartBox, shapes, is_prediction=is_prediction, **kwargs))

        return instances
