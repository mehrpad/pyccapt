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

# Default Interbus device addresses to try when scanning fails. The OXPS
# is most commonly at address 1 or 15 (0x0F) in shipped systems; we try
# both and fall back to a sweep across plausible values.
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
			f"NKTP_DLL.py not found at {module_file}. "
			"Re-install the NKT SDK with the Python examples option."
		)
	spec = importlib.util.spec_from_file_location("_pyccapt_NKTP_DLL", module_file)
	if spec is None or spec.loader is None:
		raise NKTPSwitchError(f"Could not load module spec for {module_file}")
	module = importlib.util.module_from_spec(spec)
	sys.modules["_pyccapt_NKTP_DLL"] = module
	try:
		spec.loader.exec_module(module)
	except OSError as exc:  # DLL load failure
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
	# Final fallback: sweep the small-address range. Slow but reliable.
	for value in range(1, 32):
		if value not in addresses:
			addresses.append(value)
	return addresses


def _find_oxps_address(nkt, port: str, extra_addresses: Iterable[int] | None) -> int:
	"""Probe a port for the OXPS module address.

	Tries the well-known addresses first, then sweeps. Uses the
	``deviceGetType`` lightweight call which returns the 16-bit module
	type code — we match against ``OXPS_MODULE_TYPE``.
	"""
	for dev_id in _expand_probe_addresses(extra_addresses):
		try:
			result, dev_type = nkt.deviceGetType(port, dev_id)
		except Exception:
			continue
		# 0 = success in NKT's result enum; dev_type 0 means "no module".
		if result == 0 and dev_type and (dev_type & 0xFF) == OXPS_MODULE_TYPE:
			_log.info("Found OXPS module at devId=0x%02X on %s", dev_id, port)
			return dev_id
	raise NKTPSwitchError(
		f"No Origami OXPS module found on {port} via NKTPBus. "
		"Check the cable, baudrate, and that the laser is powered."
	)


# ---------------------------------------------------------------- public API

def switch_to_cli(
		port: str,
		*,
		device_address: int | None = None,
		extra_addresses: Iterable[int] | None = None,
		settle_seconds: float = 1.0,
) -> int:
	"""Switch the OXPS at ``port`` from NKTPBus mode to CLI mode.

	Args:
		port: Serial port name, e.g. ``"COM9"``.
		device_address: Force a specific Interbus device address. If
			None (the default), the function probes well-known
			addresses and sweeps low values.
		extra_addresses: Additional addresses to try before the sweep.
		settle_seconds: How long to wait after the register write before
			closing the port — gives the laser firmware time to switch
			its UART driver to 38 400 baud.

	Returns:
		The device address that was used.

	Raises:
		NKTPSwitchError: if NKTPDLL is not installed, the laser cannot
			be reached on the port, no OXPS module is found, or the
			register write fails.
	"""
	nkt = _load_nktp_dll_module()

	# autoMode=1, liveMode=0 — scan once, no continuous monitoring.
	open_result = nkt.openPorts(port, 1, 0)
	if open_result != 0:
		raise NKTPSwitchError(
			f"openPorts({port}) failed: {nkt.PortResultTypes(open_result)}. "
			"Either the port name is wrong, another process owns the "
			"port (e.g. NKT CONTROL is open), or the laser is in CLI "
			"mode already."
		)
	try:
		dev_id = (
			device_address
			if device_address is not None
			else _find_oxps_address(nkt, port, extra_addresses)
		)
		write_result = nkt.registerWriteU8(
			port, dev_id, OXPS_INTERFACE_MODE_REGISTER, INTERFACE_MODE_CLI, -1,
		)
		if write_result != 0:
			raise NKTPSwitchError(
				f"registerWriteU8 failed: "
				f"{nkt.RegisterResultTypes(write_result)}"
			)
		# Best-effort: give the laser a moment to flip the UART before
		# we close the host side and let CLI try to talk to it.
		time.sleep(max(0.0, float(settle_seconds)))
		_log.info(
			"Wrote reg 0x%02X = %d (CLI) to OXPS on %s (devId=0x%02X)",
			OXPS_INTERFACE_MODE_REGISTER, INTERFACE_MODE_CLI, port, dev_id,
		)
		return dev_id
	finally:
		try:
			nkt.closePorts(port)
		except Exception:
			pass


def switch_to_nktpbus(port: str) -> None:
	"""Switch the OXPS from CLI mode to NKTPBus mode (CLI -> NKTPBus).

	Pure CLI command — no DLL needed. Provided here so both directions
	live in the same place. Equivalent to ``ly_oxp2_nktpbus=1``.
	"""
	import serial as _serial  # local import keeps the heavy import optional

	with _serial.Serial(
			port=port,
			baudrate=38400,
			stopbits=_serial.STOPBITS_ONE,
			bytesize=_serial.EIGHTBITS,
			rtscts=False,
			timeout=1.0,
	) as ser:
		ser.write(b"ly_oxp2_nktpbus=1\n")
		# Drain reply if any so the next opener does not see stale bytes.
		time.sleep(0.2)
		try:
			ser.read(ser.in_waiting or 0)
		except Exception:
			pass
	_log.info("Sent CLI -> NKTPBus switch on %s", port)


def is_cli_responding(port: str, *, timeout_s: float = 2.0) -> bool:
	"""Quick probe: open the port at 38 400 baud and ask for status.

	Returns True iff the laser replies with the expected ``ly_oxp2_*``
	echo, meaning it is currently in CLI mode. Anything else (no reply,
	garbage at 38 400 baud because the laser is at 115 200, exception)
	returns False without raising.

	This is the simplest, lowest-risk way to detect "laser is in
	NKTPBus mode" before attempting an automatic recovery.
	"""
	try:
		import serial as _serial
	except ImportError:
		return False
	try:
		with _serial.Serial(
				port=port,
				baudrate=38400,
				stopbits=_serial.STOPBITS_ONE,
				bytesize=_serial.EIGHTBITS,
				rtscts=False,
				timeout=timeout_s,
		) as ser:
			ser.reset_input_buffer()
			ser.write(b"ly_oxp2_dev_status\n")
			deadline = time.time() + timeout_s
			buf = b""
			while time.time() < deadline:
				chunk = ser.read(64)
				if chunk:
					buf += chunk
					if b"\n" in buf:
						break
				else:
					break
		return b"ly_oxp2" in buf
	except Exception:
		return False
