from PyQt6.QtWidgets import QSpinBox
from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut

class IdSpinBox(QSpinBox):
    pressedReturn = pyqtSignal()
    pressedCtrlReturn = pyqtSignal()
    pressedAltReturn = pyqtSignal()
    pressedCtrlAltReturn = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shortcuts: list[QShortcut] = []

        self._returnShortcuts = [
            {
                'signal': self.pressedReturn,
                'keys': [
                    (Qt.KeyboardModifier.NoModifier, Qt.Key.Key_Return),    # Enter
                    (Qt.KeyboardModifier.KeypadModifier, Qt.Key.Key_Enter), # Numpad Enter
                ],
            },
            {
                'signal': self.pressedCtrlReturn,
                'keys': [
                    (Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Return),    # Ctrl+Enter
                    (Qt.KeyboardModifier.KeypadModifier | Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Enter), # Ctrl+Numpad Enter
                ]
            },
            {
                'signal': self.pressedAltReturn,
                'keys': [
                    (Qt.KeyboardModifier.AltModifier, Qt.Key.Key_Return),    # Alt+Enter
                    (Qt.KeyboardModifier.KeypadModifier | Qt.KeyboardModifier.AltModifier, Qt.Key.Key_Enter), # Alt+Numpad Enter
                ]
            },
            {
                'signal': self.pressedCtrlAltReturn,
                'keys': [
                    (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier, Qt.Key.Key_Return),    # Ctrl+Alt+Enter
                    (Qt.KeyboardModifier.KeypadModifier | Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier, Qt.Key.Key_Enter), # Ctrl+Alt+Numpad Enter
                ]
            }
        ]

    def event(self, event):
        if event.type() != QEvent.Type.KeyPress:
            return super().event(event)

        for shortcut in self._returnShortcuts:
            for modifier, key in shortcut['keys']:
                if event.modifiers() == modifier and event.key() == key:
                    shortcut['signal'].emit()
                    return False

        key = QKeySequence(event.keyCombination())
        for shortcut in self.shortcuts:
            if key.matches(shortcut.key()) == QKeySequence.SequenceMatch.ExactMatch:
                shortcut.activated.emit()
                return False

        return super().event(event)
