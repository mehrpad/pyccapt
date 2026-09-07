from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from pyccapt.control.core.safety import (
    NullSafetyInterlock,
    NIDaqSafetyInterlock,
    audit_safety_override,
    build_safety_interlock,
)


def test_null_interlock_is_explicitly_safe():
    interlock = build_safety_interlock({"safety_interlock_backend": "none"})
    assert isinstance(interlock, NullSafetyInterlock)
    assert interlock.is_safe()
    interlock.kick_watchdog()
    interlock.close()


def test_unknown_interlock_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        build_safety_interlock({"safety_interlock_backend": "magic"})


def test_override_is_audited(tmp_path):
    path = audit_safety_override(
        tmp_path,
        operator="operator-1",
        disabled_devices=["tdc", "v_p"],
        reason="maintenance",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["operator"] == "operator-1"
    assert record["disabled_devices"] == ["tdc", "v_p"]


def test_nidaq_estop_watchdog_and_close_leave_output_inactive(monkeypatch):
    tasks = []
    state = SimpleNamespace(safe=True)

    class Channels:
        def add_di_chan(self, channel):
            self.channel = channel

        def add_do_chan(self, channel):
            self.channel = channel

    class Task:
        def __init__(self):
            self.di_channels = Channels()
            self.do_channels = Channels()
            self.writes = []
            self.closed = False
            tasks.append(self)

        def read(self):
            return state.safe

        def write(self, value):
            self.writes.append(bool(value))

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "nidaqmx", SimpleNamespace(Task=Task))
    interlock = NIDaqSafetyInterlock("Dev1/port0/line0", "Dev1/port0/line1")

    assert interlock.is_safe() is True
    interlock.kick_watchdog()
    state.safe = False
    assert interlock.is_safe() is False
    interlock.close()

    assert tasks[1].writes == [False, True, False]
    assert all(task.closed for task in tasks)
