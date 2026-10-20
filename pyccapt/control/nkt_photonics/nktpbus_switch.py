"""Bidirectional NKTPBus <-> CLI mode switch for the Origami OXPS laser.

Vendor / origin
---------------
This module talks to **NKT Photonics A/S** hardware (Origami XPS laser)
via NKT's proprietary protocols and SDK:

* The ``NKTPDLL.dll`` shared library (loaded at runtime from
  ``%NKTP_SDK_PATH%\\NKTPDLL\\x64\\``) and the bundled Python wrapper
  ``Examples/DLL_Example_Python/NKTP_DLL.py`` are NKT's property and
  are installed by NKT's SDK installer. PyCCAPT does not redistribute
  them.
* The Interbus / NKTPBus binary protocol, the OXPS register map
  (module type ``0x95``, register ``0x39`` = serial-port communication
  protocol), the CLI command set, and the device-address conventions
  are all defined and owned by NKT.

This file is local pyccapt code that *uses* NKT's SDK; it does not
contain NKT source. Use of the OXPS hardware and these interfaces is
subject to NKT Photonics' licence terms.

The Origami OXPS exposes the same RS-232 port as either:

* CLI mode (38 400 baud, ASCII commands like ``ly_oxp2_listen``) — what
  ``origamiClassCLI`` uses everywhere else in the codebase.
* NKTPBus mode (115 200 baud, framed binary "Interbus" telegrams) — what
  the official NKT CONTROL software uses.

The two are mutually exclusive: in NKTPBus mode the laser ignores ASCII,
and in CLI mode it ignores Interbus telegrams. So if the laser was last
left in NKTPBus mode (the factory default for some firmware versions, or
because someone last touched it from CONTROL), the CLI-based GUI cannot
talk to it at all.

This module wraps NKT's official NKTPDLL.dll (already installed on this
PC under ``%NKTP_SDK_PATH%``; the installer ships an official Python
binding in ``Examples/DLL_Example_Python/NKTP_DLL.py``) and exposes two
high-level helpers:

* :func:`switch_to_cli`  — assume NKTPBus mode, write register 0x39 := 1.
* :func:`switch_to_nktpbus` — pure CLI command (no DLL needed for this
  direction; provided here so both directions live in one place).

The OXPS register that controls the active interface is documented in
the SDK register file ``95.txt`` (module type 0x95):

    Register 0x39   "Serial port communication protocol"
                    [0: NKTP Bus / 1: CLI]   U8

After switching to CLI, the laser side immediately starts listening on
38 400 baud; the host must close the 115 200-baud port and re-open at
38 400 to resume normal CLI traffic.

Failure modes are surfaced as :class:`NKTPSwitchError` rather than
generic exceptions so callers can present a clear message to the
operator.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

_log = logging.getLogger("pyccapt.laser")

# ---------------------------------------------------------------- constants

# Module type byte for the Origami OXPS main controller (file ``95.txt``
# in the NKT SDK Register Files directory).
OXPS_MODULE_TYPE = 0x95

# Register that controls which interface the OXPS speaks on its serial
# port. Value 0 = NKTPBus, value 1 = CLI. U8.
OXPS_INTERFACE_MODE_REGISTER = 0x39
INTERFACE_MODE_NKTPBUS = 0
INTERFACE_MODE_CLI = 1
REGISTER_RESULT_TIMEOUT = 6  # NKT SDK RegisterResultTypes: RegResultTimeout

# Preferred addresses when selecting OXPS from the SDK's scan table.
# These are table lookups, not direct reads from possibly absent devices.
_DEFAULT_PROBE_ADDRESSES: tuple[int, ...] = (1, 15, 16, 10, 0x60, 0x67)


class NKTPSwitchError(RuntimeError):
    """Raised when we cannot complete a mode switch."""


def _resolve_sdk_path() -> Path | None:
    """Return the NKT SDK install root, or None if not found."""
    candidate = os.environ.get("NKTP_SDK_PATH")
    if candidate:
        path = Path(candidate)
        if path.is_dir():
            return path
    # Fall back to the well-known install location.
    default = Path(r"C:\Users\Public\Documents\NKT Photonics\SDK")
    return default if default.is_dir() else None


def _load_nktp_dll_module():
    """Import NKTP_DLL.py from the SDK install. Cached after first load.

    The bundled NKTP_DLL.py uses the ``NKTP_SDK_PATH`` env var to locate
    NKTPDLL.dll, which is exactly what we want — no copies, no version
    skew. Importing it eagerly loads the DLL and surfaces install
    problems immediately.
    """
    if "_pyccapt_NKTP_DLL" in sys.modules:
        return sys.modules["_pyccapt_NKTP_DLL"]

    sdk = _resolve_sdk_path()
    if sdk is None:
        raise NKTPSwitchError(
            "NKT SDK not found. Expected NKTP_SDK_PATH env var or "
            r"C:\Users\Public\Documents\NKT Photonics\SDK to exist. "
            "Install the NKT SDK first."
        )
    module_file = sdk / "Examples" / "DLL_Example_Python" / "NKTP_DLL.py"
    if not module_file.is_file():
        raise NKTPSwitchError(
            f"NKTP_DLL.py not found at {module_file}. Re-install the NKT SDK with the Python examples option."
        )
    spec = importlib.util.spec_from_file_location("_pyccapt_NKTP_DLL", module_file)
    if spec is None or spec.loader is None:
        raise NKTPSwitchError(f"Could not load module spec for {module_file}")
    module = importlib.util.module_from_spec(spec)
    # The vendor wrapper loads its DLL using this environment variable, even
    # when we found the SDK through the default install location.
    os.environ['NKTP_SDK_PATH'] = str(sdk)
    sys.modules["_pyccapt_NKTP_DLL"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        # A failed import must not poison subsequent button clicks with a
        # partially initialized module cached for the rest of the GUI session.
        sys.modules.pop("_pyccapt_NKTP_DLL", None)
        raise NKTPSwitchError(
            f"Could not load NKTPDLL: {exc}. Confirm the SDK matches "
            "your Python architecture (64-bit Python needs the x64 DLL)."
        ) from exc
    return module


# ---------------------------------------------------------------- helpers


def _expand_probe_addresses(extra: Iterable[int] | None = None) -> list[int]:
    addresses: list[int] = list(_DEFAULT_PROBE_ADDRESSES)
    if extra:
        for value in extra:
            if value not in addresses:
                addresses.append(int(value))
    # Remaining low addresses in the cached scan, without serial transactions.
    for value in range(1, 32):
        if value not in addresses:
            addresses.append(value)
    return addresses


def _find_oxps_address(nkt, port: str, extra_addresses: Iterable[int] | None) -> int:
    """Use the SDK's completed auto-scan table, indexed by module address.

    Do not issue deviceGetType reads to absent addresses: the installed DLL
    can crash natively on those reads, which Python cannot catch.
    """
    result, types = nkt.deviceGetAllTypes(port)
    if result != 0:
        raise NKTPSwitchError(
            f'Could not read the NKTPBus scan on {port}: {nkt.DeviceResultTypes(result)}. '
            'The SDK may have closed the port after finding no responding devices. '
            'Check controller power, the serial cable and the NKT CONTROL connection.'
        )
    candidates = list(extra_addresses or ()) + _expand_probe_addresses() + list(range(len(types)))
    for dev_id in candidates:
        if 0 < dev_id < len(types) and types[dev_id] == OXPS_MODULE_TYPE:
            _log.info("Found OXPS module at devId=0x%02X on %s", dev_id, port)
            return dev_id
    raise NKTPSwitchError(
        f"No Origami OXPS module found on {port} via NKTPBus. Check the cable, baudrate, and that the laser is powered."
    )


# ---------------------------------------------------------------- public API


def request_cli_protocol(nkt, port: str, device_address: int) -> int:
    """Request the resetting protocol change; its ACK can disappear mid-reset.

    Callers must close the SDK port and verify CLI before claiming success.
    Only the SDK's timeout is ambiguous; NACK/CRC/type errors still fail here.
    """
    result = nkt.registerWriteU8(port, device_address, OXPS_INTERFACE_MODE_REGISTER, INTERFACE_MODE_CLI, -1)
    if result not in (0, REGISTER_RESULT_TIMEOUT):
        raise NKTPSwitchError(f'registerWriteU8 failed: {nkt.RegisterResultTypes(result)}')
    if result == REGISTER_RESULT_TIMEOUT:
        _log.warning(
            'CLI protocol write on %s (devId=0x%02X) returned %s; '
            'the laser may reset before acknowledging. Releasing NKTPBus and verifying CLI.',
            port, device_address, nkt.RegisterResultTypes(result),
        )
    return result


def wait_for_cli(port: str, *, settle_seconds: float = 1.0, attempts: int = 3,
                 wait=None, cancelled=None) -> None:
    """Verify numeric CLI status after SDK closure, never by writing again."""
    wait = time.sleep if wait is None else wait
    wait(max(0.0, float(settle_seconds)))
    last_error = 'no valid numeric CLI status reply'
    for attempt in range(1, attempts + 1):
        if cancelled is not None and cancelled():
            raise NKTPSwitchError('CLI reconnect cancelled during shutdown.')
        _log.info('Verifying CLI on %s: attempt %d of %d', port, attempt, attempts)
        try:
            if is_cli_responding(port, raise_on_port_error=True):
                _log.info('CLI verified on %s at 38400 baud', port)
                return
        except NKTPSwitchError as exc:
            last_error = str(exc)
        if attempt < attempts:
            wait(0.5)
    raise NKTPSwitchError(
        f'CLI was not verified on {port} after {attempts} attempts: {last_error}. '
        'Disconnect NKT CONTROL/serial terminals and click Switch to CLI again.'
    )


def switch_to_cli(
    port: str,
    *,
    device_address: int | None = None,
    extra_addresses: Iterable[int] | None = None,
    settle_seconds: float = 1.0,
    wait=None,
    cancelled=None,
) -> int:
    """Switch the OXPS at ``port`` from NKTPBus mode to CLI mode.

    Args:
            port: Serial port name, e.g. ``"COM9"``.
            device_address: Force a specific Interbus device address. If
                    None (the default), the function probes well-known
                    addresses and sweeps low values.
            extra_addresses: Additional addresses to try before the sweep.
            settle_seconds: Initial reset delay after closing the SDK port.
            wait: Optional GUI-friendly delay function.
            cancelled: Optional shutdown predicate, checked before each probe.

    Returns:
            The device address that was used, after CLI status is verified.

    Raises:
            NKTPSwitchError: if NKTPDLL is not installed, the laser cannot
                    be reached on the port, no OXPS module is found, or the
                    register write fails.
    """
    nkt = _load_nktp_dll_module()

    try:
        # autoMode=1, liveMode=0 — scan once, no continuous monitoring.
        # Even a failed/partial open must go through closePorts below.
        open_result = nkt.openPorts(port, 1, 0)
        if open_result != 0:
            raise NKTPSwitchError(
                f"openPorts({port}) failed: {nkt.PortResultTypes(open_result)}. "
                "Disconnect the laser in NKT CONTROL or any serial terminal "
                "before retrying; PyCCAPT can stay open."
            )
        dev_id = device_address if device_address is not None else _find_oxps_address(nkt, port, extra_addresses)
        write_result = request_cli_protocol(nkt, port, dev_id)
        _log.info(
            "Requested reg 0x%02X = %d (CLI) on %s (devId=0x%02X)",
            OXPS_INTERFACE_MODE_REGISTER,
            INTERFACE_MODE_CLI,
            port,
            dev_id,
        )
    finally:
        had_error = sys.exc_info()[0] is not None
        try:
            close_result = nkt.closePorts(port)
            if close_result != 0:
                raise NKTPSwitchError(
                    f"Could not release NKTPBus port {port}: {nkt.PortResultTypes(close_result)}"
                )
        except Exception as exc:
            if not had_error:
                raise NKTPSwitchError(f"Could not release NKTPBus port {port}: {exc}") from exc
            _log.warning('NKTPBus cleanup on %s failed: %s', port, exc)
    try:
        wait_for_cli(port, settle_seconds=settle_seconds, wait=wait, cancelled=cancelled)
    except NKTPSwitchError as exc:
        raise NKTPSwitchError(
            f'CLI protocol request returned {nkt.RegisterResultTypes(write_result)}; {exc}'
        ) from exc
    return dev_id


def switch_to_nktpbus(port: str) -> None:
    """Switch the OXPS from CLI mode to NKTPBus mode (CLI -> NKTPBus).

    Pure CLI command — no DLL needed. Provided here so both directions
    live in the same place. Equivalent to ``ly_oxp2_nktpbus=1``.
    """
    from pyccapt.control.nkt_photonics.origamiClassCLI import origClass

    device = origClass(port)
    try:
        if device.open_port() != 0:
            raise NKTPSwitchError(
                f"Could not open {port}: {device.last_error}. Disconnect the laser "
                "in NKT CONTROL or any serial terminal before retrying."
            )
        device.InterbusEnable()
    finally:
        device.close_port()
    _log.info("Sent CLI -> NKTPBus switch on %s", port)


def is_cli_responding(port: str, *, timeout_s: float = 2.0, raise_on_port_error: bool = False) -> bool:
    """Quick probe: open the port at 38 400 baud and ask for status.

    Requires a parsed numeric status, rather than just a command echo.
    False means CLI communication was not established; it does not prove
    NKTPBus mode. Uses the same framing as the GUI, including promptless replies.
    With ``raise_on_port_error``, an unavailable/busy port raises instead of
    being mistaken for a different protocol and passed to the SDK.
    """
    try:
        from pyccapt.control.nkt_photonics.origamiClassCLI import origClass
    except ImportError:
        return False
    device = origClass(port)
    try:
        if device.open_port() != 0:
            if raise_on_port_error:
                raise NKTPSwitchError(
                    f"Could not open {port}: {device.last_error}. Disconnect the laser "
                    "in NKT CONTROL or any serial terminal before retrying; PyCCAPT can stay open."
                )
            return False
        return bool(device.StatusRead(timeout=timeout_s))
    except NKTPSwitchError:
        raise
    except Exception:
        return False
    finally:
        try:
            device.close_port()
        except Exception:
            pass
