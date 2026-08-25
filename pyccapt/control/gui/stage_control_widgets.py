"""Shared widgets for specimen-stage and laser-stage jog controls."""

from PyQt6 import QtCore, QtWidgets

from pyccapt.control.smaract_mcs2 import mcs2_stage


class SpeedSelector(QtWidgets.QComboBox):
    """Discrete speed preset selector with slider-compatible accessors."""

    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent, lo, hi, default, max_mm_s, table):
        super().__init__(parent=parent)
        for level in range(lo, hi + 1):
            velocity_mm_s = mcs2_stage.speed_level_to_m_s(
                level, hi, max_mm_s, table=table
            ) * 1000
            self.addItem(f"{velocity_mm_s:.3f} mm/s", level)
        default_index = self.findData(default)
        self.setCurrentIndex(default_index if default_index >= 0 else 0)
        self.setMinimumWidth(132)
        self.setStyleSheet(
            "QComboBox{background:white;color:#17364d;border:1px solid #7895aa;"
            "border-radius:5px;padding:4px 8px;font-weight:600;}"
            "QComboBox:hover{border-color:#347eaa;background:#f3f9fc;}"
            "QComboBox::drop-down{border:0;width:22px;}"
            "QComboBox QAbstractItemView{selection-background-color:#9fd2f0;}"
        )
        self.currentIndexChanged.connect(
            lambda _index: self.valueChanged.emit(self.value())
        )

    def value(self):
        return int(self.currentData())

    def setValue(self, value):
        index = self.findData(int(value))
        if index >= 0:
            self.setCurrentIndex(index)


JOG_BUTTON_STYLE = """
QPushButton {
    background-color: #e8f0f6;
    color: #17364d;
    border: 1px solid #7895aa;
    border-radius: 7px;
    font-size: 10pt;
    font-weight: 600;
    padding: 3px;
}
QPushButton:hover {
    background-color: #d7ebf8;
    border-color: #347eaa;
}
QPushButton:pressed {
    background-color: #9fd2f0;
    border: 2px solid #176b9b;
}
QPushButton:disabled {
    background-color: #eeeeee;
    color: #9a9a9a;
    border-color: #c5c5c5;
}
"""

JOG_GROUP_STYLE = """
QGroupBox {
    border: 1px solid #a7b6c2;
    border-radius: 8px;
    margin-top: 8px;
    font-weight: bold;
    color: #29475d;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
}
"""


def make_jog_button(parent, text, width=58):
    button = QtWidgets.QPushButton(text, parent=parent)
    button.setFixedSize(QtCore.QSize(width, 42))
    button.setStyleSheet(JOG_BUTTON_STYLE)
    return button
