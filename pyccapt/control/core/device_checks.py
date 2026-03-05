"""Startup device validation helpers for experiment launch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class DeviceIssue:
    """Represents a startup issue for a configured hardware device."""

    device: str
    reason: str


def _is_enabled(conf: Mapping[str, Any], key: str) -> bool:
    return str(conf.get(key, "off")).strip().lower() == "on"


def _is_voltage_pulse_mode(pulse_mode: str | None) -> bool:
    return pulse_mode in {"Voltage", "VoltageLaser"}


def _resolve_config_value(conf: Mapping[str, Any], variables: Any, key: str) -> str:
    value = getattr(variables, key, None)
    if value is None:
        value = conf.get(key, "")
    return str(value).strip()


def _check_serial_device(
    *,
    device_label: str,
    port: str,
    baudrate: int,
    serial_factory: Callable[..., Any],
) -> DeviceIssue | None:
    if not port:
        return DeviceIssue(device=device_label, reason="missing COM port in configuration")

    handle = None
    try:
        handle = serial_factory(
            port=port,
            baudrate=baudrate,
            timeout=0.2,
        )
    except Exception as exc:  # pragma: no cover - exact backend errors are environment-specific
        return DeviceIssue(device=device_label, reason=f"cannot open '{port}' ({exc})")
    finally:
        if handle is not None and hasattr(handle, "close"):
            try:
                handle.close()
            except Exception:
                pass

    return None


def _check_signal_generator(
    *,
    resource_name: str,
    resource_manager_factory: Callable[[], Any],
) -> DeviceIssue | None:
    if not resource_name:
        return DeviceIssue(device="signal_generator", reason="missing resource identifier in configuration")

    manager = None
    resource = None
    try:
        manager = resource_manager_factory()
        resource = manager.open_resource(resource_name)
    except Exception as exc:  # pragma: no cover - exact backend errors are environment-specific
        return DeviceIssue(device="signal_generator", reason=f"cannot open '{resource_name}' ({exc})")
    finally:
        if resource is not None and hasattr(resource, "close"):
            try:
                resource.close()
            except Exception:
                pass
        if manager is not None and hasattr(manager, "close"):
            try:
                manager.close()
            except Exception:
                pass

    return None


def collect_startup_device_issues(
    conf: Mapping[str, Any],
    variables: Any,
    *,
    pulse_mode: str | None,
    serial_factory: Callable[..., Any] | None = None,
    resource_manager_factory: Callable[[], Any] | None = None,
) -> list[DeviceIssue]:
    """Collect startup device issues for experiment-critical devices.

    Only devices configured as `"on"` are checked.
    """

    issues: list[DeviceIssue] = []
    needs_voltage_hw = _is_enabled(conf, "v_dc") or (
        _is_enabled(conf, "v_p") and _is_voltage_pulse_mode(pulse_mode)
    )
    needs_signal_generator = _is_enabled(conf, "signal_generator") and _is_voltage_pulse_mode(pulse_mode)

    if serial_factory is None and needs_voltage_hw:
        import serial

        serial_factory = serial.Serial

    if resource_manager_factory is None and needs_signal_generator:
        import pyvisa

        resource_manager_factory = pyvisa.ResourceManager

    if _is_enabled(conf, "v_dc"):
        vdc_port = _resolve_config_value(conf, variables, "COM_PORT_V_dc")
        issue = _check_serial_device(
            device_label="v_dc",
            port=vdc_port,
            baudrate=115200,
            serial_factory=serial_factory,
        )
        if issue is not None:
            issues.append(issue)

    if _is_enabled(conf, "tdc"):
        tdc_model = str(conf.get("tdc_model", "")).strip()
        if not tdc_model:
            issues.append(DeviceIssue(device="tdc", reason="missing tdc_model in configuration"))

    if _is_enabled(conf, "v_p") and _is_voltage_pulse_mode(pulse_mode):
        vp_port = _resolve_config_value(conf, variables, "COM_PORT_V_p")
        issue = _check_serial_device(
            device_label="v_p",
            port=vp_port,
            baudrate=115200,
            serial_factory=serial_factory,
        )
        if issue is not None:
            issues.append(issue)

    if needs_signal_generator:
        signal_resource = _resolve_config_value(conf, variables, "COM_PORT_signal_generator")
        issue = _check_signal_generator(
            resource_name=signal_resource,
            resource_manager_factory=resource_manager_factory,
        )
        if issue is not None:
            issues.append(issue)

    return issues


def format_startup_device_issue_message(issues: Sequence[DeviceIssue]) -> str:
    """Format user-facing startup issue message for terminal and GUI."""

    details = "; ".join(f"{item.device}: {item.reason}" for item in issues)
    return (
        "Experiment start blocked. Device check failed for enabled devices: "
        f"{details}. Turn on/fix these devices, or set them to 'off' in config.toml."
    )
