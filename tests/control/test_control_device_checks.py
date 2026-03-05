"""Tests for control startup device validation."""

from __future__ import annotations

from types import SimpleNamespace

from pyccapt.control.core import device_checks


def _variables() -> SimpleNamespace:
    return SimpleNamespace(
        COM_PORT_V_dc="COM6",
        COM_PORT_V_p="COM4",
        COM_PORT_signal_generator="USB0::TEST::INSTR",
    )


def test_collect_startup_device_issues_skips_devices_set_to_off():
    conf = {"v_dc": "off", "v_p": "off", "signal_generator": "off"}

    issues = device_checks.collect_startup_device_issues(
        conf,
        _variables(),
        pulse_mode="Voltage",
        serial_factory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("must not be called")),
        resource_manager_factory=lambda: (_ for _ in ()).throw(RuntimeError("must not be called")),
    )

    assert issues == []


def test_collect_startup_device_issues_reports_serial_error():
    conf = {"v_dc": "on", "v_p": "off", "signal_generator": "off"}

    def failing_serial(**_kwargs):
        raise OSError("access denied")

    issues = device_checks.collect_startup_device_issues(
        conf,
        _variables(),
        pulse_mode="Voltage",
        serial_factory=failing_serial,
        resource_manager_factory=lambda: None,
    )

    assert len(issues) == 1
    assert issues[0].device == "v_dc"
    assert "access denied" in issues[0].reason


def test_collect_startup_device_issues_reports_signal_generator_error():
    conf = {"v_dc": "off", "v_p": "off", "signal_generator": "on"}

    class FailingManager:
        def open_resource(self, _resource_name):
            raise ConnectionError("visa timeout")

        def close(self):
            return None

    issues = device_checks.collect_startup_device_issues(
        conf,
        _variables(),
        pulse_mode="Voltage",
        serial_factory=lambda **_kwargs: None,
        resource_manager_factory=FailingManager,
    )

    assert len(issues) == 1
    assert issues[0].device == "signal_generator"
    assert "visa timeout" in issues[0].reason


def test_collect_startup_device_issues_reports_missing_tdc_model():
    conf = {"tdc": "on", "v_dc": "off", "v_p": "off", "signal_generator": "off"}

    issues = device_checks.collect_startup_device_issues(
        conf,
        _variables(),
        pulse_mode="Voltage",
        serial_factory=lambda **_kwargs: None,
        resource_manager_factory=lambda: None,
    )

    assert len(issues) == 1
    assert issues[0].device == "tdc"
    assert "missing tdc_model" in issues[0].reason


def test_collect_startup_device_issues_skips_voltage_pulse_hardware_in_laser_mode():
    conf = {"v_dc": "off", "v_p": "on", "signal_generator": "on"}

    issues = device_checks.collect_startup_device_issues(
        conf,
        _variables(),
        pulse_mode="Laser",
        serial_factory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("must not be called")),
        resource_manager_factory=lambda: (_ for _ in ()).throw(RuntimeError("must not be called")),
    )

    assert issues == []


def test_format_startup_device_issue_message_contains_guidance():
    message = device_checks.format_startup_device_issue_message(
        [
            device_checks.DeviceIssue("v_dc", "cannot open 'COM6'"),
            device_checks.DeviceIssue("signal_generator", "cannot open 'USB0::TEST::INSTR'"),
        ]
    )

    assert "Experiment start blocked" in message
    assert "v_dc: cannot open 'COM6'" in message
    assert "set them to 'off' in config.toml" in message
