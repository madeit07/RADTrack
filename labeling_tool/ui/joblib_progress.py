import contextlib

from PyQt6.QtCore import QObject, pyqtSignal, QMutex
import joblib

class JoblibSignals(QObject):
    started = pyqtSignal(int)
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

        self._mutex = QMutex()

        self.reset()

    def update(self, value):
        self._mutex.lock()
        self._completedBatches += value
        self._mutex.unlock()

    def reset(self):
        self._mutex.lock()
        self._completedBatches = 0
        self._mutex.unlock()

    def report(self, value):
        self.update(value)
        self.progress.emit(self._completedBatches)


# https://stackoverflow.com/a/58936697
@contextlib.contextmanager
def joblib_signal(signaler: JoblibSignals):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            signaler.report(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield signaler
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
