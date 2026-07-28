"""Serial controller for the camera-alignment NeoPixel illumination.

The matching Arduino sketch lives in ``devices_test/arduino_neopixel_illumination``.
It exposes a deliberately small, line-based protocol so the controller can be
tested from a serial terminal as well as the Cameras & Alignment GUI.
"""

from __future__ import annotations

import time
from typing import Iterable

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - reported clearly at runtime on minimal installs
    serial = None
    list_ports = None


class ArduinoIllumination:
    """Control the Nano running the PyCCAPT NeoPixel illumination sketch.

    The Arduino accepts ``PING``, ``ON <percent>``, ``OFF``,
    ``BRIGHTNESS <percent>``, and ``COLOR <red> <green> <blue>`` commands.
    Each command responds with an ``OK``
    line, allowing auto-detection without claiming an unrelated serial device.
    """

    BAUDRATE = 115_200
    HANDSHAKE = "PYCCAPT_ILLUMINATION"

    def __init__(self, port: str = "auto", timeout: float = 0.5):
        self.port = str(port).strip()
        self.timeout = timeout
        self.serial_port = None

    @property
    def connected(self) -> bool:
        return self.serial_port is not None and self.serial_port.is_open

    def connect(self) -> str:
        """Open and verify the configured Arduino; return its resolved port."""
        if self.connected:
            return self.serial_port.port
        if serial is None or list_ports is None:
            raise RuntimeError("pyserial is required for Arduino illumination control.")

        candidates = self._candidate_ports()
        if not candidates:
            raise RuntimeError("No candidate Arduino serial ports found.")

        errors: list[str] = []
        for candidate in candidates:
            try:
                connection = serial.Serial(candidate, self.BAUDRATE, timeout=self.timeout)
                # Opening an Arduino Nano serial port resets it. Nano clones
                # can need several seconds to leave the bootloader, so retry
                # PING rather than assuming one early attempt is conclusive.
                deadline = time.monotonic() + 5.0
                reply = ""
                while time.monotonic() < deadline:
                    connection.reset_input_buffer()
                    connection.write(b"PING\n")
                    connection.flush()
                    reply = connection.readline().decode("ascii", errors="replace").strip()
                    if reply == self.HANDSHAKE:
                        self.serial_port = connection
                        return candidate
                    time.sleep(0.25)
                connection.close()
                errors.append(f"{candidate}: unexpected reply {reply!r}")
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        details = "; ".join(errors) or "no response"
        raise RuntimeError(f"PyCCAPT illumination Arduino not found ({details}).")

    def set_on(self, percent: int) -> None:
        """Turn all NeoPixels on at ``percent`` brightness (0..100)."""
        self._command(f"ON {self._validate_percent(percent)}")

    def set_off(self) -> None:
        """Turn all NeoPixels off while retaining the configured brightness."""
        self._command("OFF")

    def set_brightness(self, percent: int) -> None:
        """Store brightness and apply it immediately when illumination is on."""
        self._command(f"BRIGHTNESS {self._validate_percent(percent)}")

    def set_color(self, red: int, green: int, blue: int) -> None:
        """Store the NeoPixel RGB colour and apply it immediately when on."""
        self._command(
            "COLOR "
            f"{self._validate_channel(red)} "
            f"{self._validate_channel(green)} "
            f"{self._validate_channel(blue)}"
        )

    def close(self) -> None:
        if self.serial_port is None:
            return
        try:
            self.serial_port.close()
        finally:
            self.serial_port = None

    def _candidate_ports(self) -> Iterable[str]:
        if self.port and self.port.lower() != "auto":
            return (self.port,)

        keywords = ("arduino", "ch340", "ch341", "usb serial", "cp210")
        return tuple(
            entry.device
            for entry in list_ports.comports()
            if any(keyword in f"{entry.description} {entry.hwid}".lower() for keyword in keywords)
        )

    def _command(self, command: str) -> None:
        if not self.connected:
            self.connect()
        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.write(f"{command}\n".encode("ascii"))
            self.serial_port.flush()
            reply = self.serial_port.readline().decode("ascii", errors="replace").strip()
        except Exception:
            self.close()
            raise
        if reply != "OK":
            raise RuntimeError(f"Arduino rejected {command!r}: {reply or 'no response'}")

    @staticmethod
    def _validate_percent(percent: int) -> int:
        return max(0, min(100, int(percent)))

    @staticmethod
    def _validate_channel(value: int) -> int:
        return max(0, min(255, int(value)))
