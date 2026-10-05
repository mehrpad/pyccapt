"""Tests for configured camera-ring illumination behaviour."""

from pyccapt.control.gui.gui_cameras import Ui_Cameras_Alignment


class RecordingController:
    def __init__(self):
        self.commands = []

    def set_color(self, red, green, blue):
        self.commands.append(("color", red, green, blue))

    def set_brightness(self, percent):
        self.commands.append(("brightness", percent))


def _ui(color="green"):
    ui = Ui_Cameras_Alignment.__new__(Ui_Cameras_Alignment)
    ui.conf = {"camera_illumination_color": color}
    ui.flag_super_user = True
    ui.illumination_controller = RecordingController()
    ui._report_illumination_error = lambda exc: None
    return ui


def test_dimming_change_reapplies_configured_green_before_brightness():
    ui = _ui()

    ui.update_illumination_percent(40)

    assert ui.illumination_controller.commands == [
        ("color", 0, 255, 0),
        ("brightness", 40),
    ]


def test_unknown_configured_color_safely_falls_back_to_green():
    ui = _ui("chartreuse")

    assert ui._configured_illumination_color() == ("green", (0, 255, 0))
