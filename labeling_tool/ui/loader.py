from PyQt6.QtCore import QRunnable, pyqtSignal
import joblib

from labeler.dataloader import RADTrack, RADTrackSequence
from ui.joblib_progress import JoblibSignals, joblib_signal

class DatasetLoaderSignals(JoblibSignals):
    loadedGt = pyqtSignal()

class DatasetLoader(QRunnable):
    def __init__(self, dataset: RADTrack):
        super().__init__()

        self.dataset = dataset
        self.signals = DatasetLoaderSignals()

        self.preloadFrames = 4

        self._preloader = Preloader()

    def run(self):
        self.signals.reset()
        self._preloader.reset()

        sequences = list(self.dataset.sequences.values())
        total = sum(len(sequence) for sequence in sequences)

        self.signals.started.emit(total)

        # First cache GT for labeling
        with joblib.Parallel(n_jobs=-2) as parallel:
            gts = parallel(joblib.delayed(self._preloader.loadGt)(sequence) for sequence in sequences)

        for sequence, gt in zip(sequences, gts):
            sequence.cache_gt(gt)

        self.signals.loadedGt.emit()

        # Next cache all images
        with joblib_signal(self.signals):
            # Preload first N frames
            with joblib.Parallel(n_jobs=-2, prefer='processes') as parallel:
                parallel(joblib.delayed(self._preloader.preload)(sequence, t) for sequence in sequences for t in range(1, self.preloadFrames + 1))

            if self._preloader.isAborting:
                return

            # Load remaining frames
            with joblib.Parallel(n_jobs=-2, prefer='processes') as parallel:
                parallel(joblib.delayed(self._preloader.preload)(sequence, t) for sequence in sequences for t in range(self.preloadFrames + 1, len(sequence) + 1))

        self.signals.finished.emit()

    def stop(self):
        self._preloader.stop()


class Preloader():
    """Wrapper to add aborting mechanism. This cannot be added to DatasetLoader because joblib can't handle working with members when class is from PyQt.
    """
    def __init__(self):
        self._aborting = False

    def preload(self, sequence: RADTrackSequence, frame: int):
        if self._aborting:
            raise RuntimeError('Aborting preloading.')

        sequence.cache(frame)

    def loadGt(self, sequence: RADTrackSequence):
        if self._aborting:
            raise RuntimeError('Aborting preloading.')

        return sequence.load_gt(sequence.gt_file())

    @property
    def isAborting(self):
        return self._aborting

    def stop(self):
        self._aborting = True

    def reset(self):
        self._aborting = False
