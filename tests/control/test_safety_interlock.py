from __future__ import annotations

import json

import pytest

from pyccapt.control.core.safety import (
    NullSafetyInterlock,
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
