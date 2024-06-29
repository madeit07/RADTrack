import os
import shutil
import datetime

from PyQt6.QtCore import QRunnable, pyqtSignal
import joblib

from labeler.dataloader import RADTrack
from ui.joblib_progress import JoblibSignals, joblib_signal

class BackupSignals(JoblibSignals):
    startedCleaning = pyqtSignal()

class Backup(QRunnable):
    def __init__(self, dataset: RADTrack, backupDir: str, maxBackups: int):
        super().__init__()

        self.dataset = dataset
        self.backupDir = backupDir
        self.maxBackups = maxBackups
        self.datetimeFormat = '%Y-%m-%d_%H-%M-%S'
        self.signals = BackupSignals()

        self._startTime = datetime.datetime.now()

    def run(self):
        self.signals.reset()

        curBackupDir = os.path.join(self.backupDir, self._startTime.strftime(self.datetimeFormat))
        os.makedirs(curBackupDir, exist_ok=True)

        self.signals.startedCleaning.emit()

        # Delete old backups
        all_backups = sorted((d.name for d in os.scandir(self.backupDir) if d.is_dir()),
                             key=lambda d: datetime.datetime.strptime(d, self.datetimeFormat),
                             reverse=True)
        if len(all_backups) > self.maxBackups:
            backups_to_delete = all_backups[self.maxBackups:]
            for backup in backups_to_delete:
                shutil.rmtree(os.path.join(self.backupDir, backup))

        # Create new backup
        sequences = list(self.dataset.sequences.values())
        gt_files = [os.path.join(sequence.full_split, sequence.name, 'gt', 'gt.json') for sequence in sequences]

        backup_subdirs = set(os.path.join(curBackupDir, os.path.dirname(file)) for file in gt_files)
        for dir in backup_subdirs:
            os.makedirs(dir, exist_ok=True)

        self.signals.started.emit(len(gt_files))

        with joblib_signal(self.signals):
            with joblib.Parallel(n_jobs=-2) as parallel:
                parallel(joblib.delayed(shutil.copy2)(os.path.join(self.dataset.data_dir, file), os.path.join(curBackupDir, file)) for file in gt_files)

        self.signals.finished.emit()

