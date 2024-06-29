from PyQt6.QtCore import QModelIndex, Qt, QAbstractTableModel, QVariant

from labeler.dataloader import RADTrack, RADTrackSequence

class SequenceModel(QAbstractTableModel):
    def __init__(self, parent, dataset: RADTrack):
        super().__init__(parent)

        self._dataset = dataset
        self._sequences = list(sorted(self._dataset.sequences.values(), key=lambda s: s.name))
        self._header = ['Sequence', 'Split', 'Length']

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._header[section]

        return QVariant()

    def data(self, index: QModelIndex, role: int):
        if role == Qt.ItemDataRole.DisplayRole:
            seq = self._sequences[index.row()]
            if index.column() == 0:
                return seq.name

            if index.column() == 1:
                return seq.split

            if index.column() == 2:
                return len(seq)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if index.column() != 0:
                return Qt.AlignmentFlag.AlignCenter

        return QVariant()

    def rowCount(self, index):
        return len(self._sequences)

    def columnCount(self, index):
        return 3

    def sequence(self, row) -> RADTrackSequence:
        return self._sequences[row]

    def sequence_index(self, sequence):
        return self._sequences.index(sequence)
