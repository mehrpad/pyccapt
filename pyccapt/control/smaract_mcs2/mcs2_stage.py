"""SmarAct MCS2 stage driver used by the stage and laser GUIs.

Lives in a folder named ``smaract_mcs2`` (NOT ``smaract``) because the
SmarAct SDK ships its Python bindings as a top-level package called
``smaract``.  Putting our wrapper inside a folder of that exact name caused
``import smaract.ctl`` to resolve to the local empty package on some
environments and made the SDK appear missing.

This module is a thin object-oriented wrapper around the ``smaract.ctl`` SDK,
modelled after ``pyccapt/control/devices_test/stage_smartact.py`` (the
original procedural prototype).  Each GUI window creates one
``SmarActStage`` instance and keeps the connection open for the lifetime of
the window so subsequent moves do not pay the Open/Close cost on every click.

Conventions
-----------
* Public functions accept and return values in **meters** (``float``).
* The MCS2 SDK uses **picometers** (``int``); conversion happens internally.
* Axis indices: X=0, Y=1, Z=2 - matches the channel mapping on the controller.

Two MCS2 controllers can be present at the same time (one for the sample stage
and one for the laser focusing stage).  They are addressed by their locator
string (e.g. ``"network:sn:MCS2-00017939"``) which is taken from
``config.toml``.
"""

from __future__ import annotations

import time
from typing import Optional

try:
	import smaract.ctl as ctl

	_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import-time only
	ctl = None
	_IMPORT_ERROR = exc

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

M_TO_PM = 1e12
PM_TO_M = 1e-12


def _m_to_pm(value_m: float) -> int:
	return int(round(value_m * M_TO_PM))


def _pm_to_m(value_pm: int) -> float:
	return value_pm * PM_TO_M


def list_devices() -> list:
	"""Return all reachable MCS2 device locator strings."""
	if ctl is None:
		return []
	buffer = ctl.FindDevices()
	if not buffer:
		return []
	return [loc.strip() for loc in buffer.split("\n") if loc.strip()]


class SmarActStageError(RuntimeError):
	"""Raised when a SmarAct call fails or the SDK is unavailable."""


def _hint_for(code: int, locator: str = "") -> str:
	hints = {
		0xF003: "SA_CTL_ERROR_NOT_FOUND - device not found.",
		0xF004: (
			"SA_CTL_ERROR_NO_SENSOR_PRESENT - the channel has no valid "
			"sensor reading.  This usually means referencing was "
			"interrupted or never completed; run Reference again with "
			"the path clear."
		),
		0xF005: f"SA_CTL_ERROR_INVALID_LOCATOR - '{locator}' is not valid.",
		0xF00C: (
			"SA_CTL_ERROR_DEVICE_LIMIT_REACHED - too many open handles. "
			"Close other sessions first."
		),
		0xF010: (
			"SA_CTL_ERROR_NOT_REFERENCED - the channel must be "
			"referenced before this command works."
		),
		0xF01A: (
			"SA_CTL_ERROR_NETWORK_TIMEOUT - controller found but TCP "
			"connection timed out.  Check that the MCS2 is powered on, "
			"that no other process holds an open handle, and that port "
			"55551 is not blocked by a firewall."
		),
	}
	return hints.get(code, "")


def _wrap_ctl(operation: str, exc: "ctl.Error", locator: str = "") -> SmarActStageError:
	"""Convert a raw smaract.ctl Error into our SmarActStageError type.

	Centralised so every public method can wrap consistently.  The result
	carries the original error code and a human-readable hint when one is
	known.
	"""
	code = exc.args[1] if len(exc.args) >= 2 else 0
	hint = _hint_for(code, locator)
	msg = f"{operation} failed (error code {code:#06x})."
	if hint:
		msg += f"\n  Hint: {hint}"
	return SmarActStageError(msg)


class SmarActStage:
	"""Persistent connection to one MCS2 controller.

	Use as a regular object (the connection stays open) or as a context
	manager.  All public ``move_*`` / ``get_*`` methods are safe to call from
	Qt slots; long moves can be made non-blocking via ``wait=False``.
	"""

	def __init__(self, locator: str):
		if ctl is None:
			raise SmarActStageError(
				f"smaract.ctl SDK is not available: {_IMPORT_ERROR}"
			)
		if not locator:
			raise SmarActStageError("Empty locator string.")
		self.locator = locator
		try:
			self._handle = ctl.Open(locator)
		except ctl.Error as exc:
			code = exc.args[1] if len(exc.args) >= 2 else 0
			hint = _hint_for(code, locator)
			raise SmarActStageError(
				f"ctl.Open('{locator}') failed (error code {code:#06x})."
				+ (f"\n  Hint: {hint}" if hint else "")
			) from exc

	# ------------------------------------------------------------------ utils

	def __enter__(self) -> "SmarActStage":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def close(self) -> None:
		if getattr(self, "_handle", None) is None:
			return
		try:
			ctl.Close(self._handle)
		except Exception:
			pass
		self._handle = None

	def _wait(self, channel: int, timeout_s: float = 30.0,
	          cancel_event: "Optional[Any]" = None) -> None:
		"""Block until ``channel`` is no longer ACTIVELY_MOVING.

		``cancel_event`` is an optional ``threading.Event``; when set, the
		wait returns early after issuing ``ctl.Stop`` on the channel.  This
		is what makes Reference cancellable from the GUI.
		"""
		deadline = time.time() + timeout_s
		while time.time() < deadline:
			if cancel_event is not None and cancel_event.is_set():
				try:
					ctl.Stop(self._handle, channel)
				except Exception:
					pass
				return
			try:
				state = ctl.GetProperty_i32(
					self._handle, channel, ctl.Property.CHANNEL_STATE
				)
			except ctl.Error as exc:
				raise _wrap_ctl(f"GetProperty_i32(CHANNEL_STATE, ch={channel})", exc)
			if not (state & ctl.ChannelState.ACTIVELY_MOVING):
				return
			time.sleep(0.02)
		raise SmarActStageError(
			f"Channel {channel} did not stop within {timeout_s} s."
		)

	# ------------------------------------------------------------------ moves

	def move_relative(
			self,
			dx_m: float = 0.0,
			dy_m: float = 0.0,
			dz_m: float = 0.0,
			velocity_m_s: float = 1e-3,
			wait: bool = True,
	) -> None:
		velocity_pm_s = _m_to_pm(velocity_m_s)
		axes = [(AXIS_X, dx_m), (AXIS_Y, dy_m), (AXIS_Z, dz_m)]
		try:
			for channel, delta_m in axes:
				if delta_m == 0.0:
					continue
				ctl.SetProperty_i32(
					self._handle, channel, ctl.Property.MOVE_MODE,
					ctl.MoveMode.CL_RELATIVE,
				)
				ctl.SetProperty_i64(
					self._handle, channel, ctl.Property.MOVE_VELOCITY, velocity_pm_s,
				)
				ctl.Move(self._handle, channel, _m_to_pm(delta_m), 0)
		except ctl.Error as exc:
			raise _wrap_ctl("move_relative", exc)
		if wait:
			for channel, delta_m in axes:
				if delta_m == 0.0:
					continue
				self._wait(channel)

	def move_relative_axis(
			self,
			axis: int,
			delta_m: float,
			velocity_m_s: float,
			wait: bool = False,
	) -> None:
		"""Move a single axis - convenience wrapper used by the per-axis sliders."""
		if delta_m == 0.0:
			return
		try:
			ctl.SetProperty_i32(
				self._handle, axis, ctl.Property.MOVE_MODE, ctl.MoveMode.CL_RELATIVE,
			)
			ctl.SetProperty_i64(
				self._handle, axis, ctl.Property.MOVE_VELOCITY, _m_to_pm(velocity_m_s),
			)
			ctl.Move(self._handle, axis, _m_to_pm(delta_m), 0)
		except ctl.Error as exc:
			raise _wrap_ctl(f"move_relative_axis(axis={axis})", exc)
		if wait:
			self._wait(axis)

	def move_absolute(
			self,
			x_m: Optional[float] = None,
			y_m: Optional[float] = None,
			z_m: Optional[float] = None,
			velocity_m_s: float = 1e-3,
			wait: bool = True,
	) -> None:
		velocity_pm_s = _m_to_pm(velocity_m_s)
		axes = [(AXIS_X, x_m), (AXIS_Y, y_m), (AXIS_Z, z_m)]
		try:
			for channel, pos_m in axes:
				if pos_m is None:
					continue
				ctl.SetProperty_i32(
					self._handle, channel, ctl.Property.MOVE_MODE,
					ctl.MoveMode.CL_ABSOLUTE,
				)
				ctl.SetProperty_i64(
					self._handle, channel, ctl.Property.MOVE_VELOCITY, velocity_pm_s,
				)
				ctl.Move(self._handle, channel, _m_to_pm(pos_m), 0)
		except ctl.Error as exc:
			raise _wrap_ctl("move_absolute", exc)
		if wait:
			for channel, pos_m in axes:
				if pos_m is None:
					continue
				self._wait(channel)

	def stop(self) -> None:
		"""Immediately stop all three axes (best-effort, errors swallowed)."""
		for channel in (AXIS_X, AXIS_Y, AXIS_Z):
			try:
				ctl.Stop(self._handle, channel)
			except Exception:
				pass

	# ------------------------------------------------------------------ state

	def get_position(self) -> dict:
		"""Return current position of all three axes in meters.

		Raises ``SmarActStageError`` if any axis can't report a position
		(e.g. 0xF004 SA_CTL_ERROR_NO_SENSOR_PRESENT after an interrupted
		reference search).  Callers that poll on a timer should treat
		repeat errors as a single condition and back off, not spam.
		"""
		out = {}
		try:
			for name, channel in (("x", AXIS_X), ("y", AXIS_Y), ("z", AXIS_Z)):
				pm = ctl.GetProperty_i64(self._handle, channel, ctl.Property.POSITION)
				out[name] = _pm_to_m(pm)
		except ctl.Error as exc:
			raise _wrap_ctl("get_position", exc)
		return out

	# SmarAct REFERENCING_OPTIONS bits (from SmarActControlConstants.h):
	#   START_DIR     = 0x01  (0 = forward, 1 = backward as starting dir)
	#   REVERSE_DIR   = 0x02  (auto-reverse direction if end stop is hit
	#                          before the reference mark is found)
	#   AUTO_ZERO     = 0x04  (set position to 0 once the mark is found)
	# Default 0x07 = START_DIR | REVERSE_DIR | AUTO_ZERO:
	#   * starts BACKWARD so the search moves away from anything mounted
	#     in front of the stage (sample, load lock, optics) instead of
	#     into it,
	#   * auto-reverses if it hits the back end stop without finding the
	#     mark,
	#   * zeros the absolute position once the mark is found.
	REFERENCING_OPTIONS_DEFAULT = 0x07

	def find_reference(self, timeout_s: float = 120.0,
	                   cancel_event: "Optional[Any]" = None,
	                   referencing_options: int = None,
	                   velocity_m_s: float = None) -> None:
		"""Run the reference search on all three axes.

		This calls ``ctl.Reference`` (NOT ``ctl.Calibrate``):
		  * Reference  -> looks for the encoder's physical reference mark
						  and zeros the absolute position to it.  Stops
						  as soon as the mark is found.
		  * Calibrate  -> mechanical end-stop characterisation, only run
						  after hardware changes.  Not invoked here.

		Args:
			timeout_s: per-axis timeout for the wait loop.
			cancel_event: ``threading.Event``.  When set, ``_wait`` issues
				ctl.Stop on the current axis and returns; subsequent axes
				are skipped.  Lets the GUI's STOP button interrupt the
				search instead of being blocked by ``_wait``.
			referencing_options: SmarAct REFERENCING_OPTIONS bitmask.
				Defaults to ``REFERENCING_OPTIONS_DEFAULT`` (0x06 =
				REVERSE_DIR | AUTO_ZERO).  Pass 0 to reproduce the old
				"find or grind" behaviour.
			velocity_m_s: search velocity (m/s).  Set explicitly on each
				channel before ``ctl.Reference`` because otherwise the
				search uses whatever MOVE_VELOCITY was last set there -
				usually a slow per-axis jog speed, making the search
				feel broken.  ``None`` skips the velocity write (use
				channel's existing value).
		"""
		if referencing_options is None:
			referencing_options = self.REFERENCING_OPTIONS_DEFAULT
		velocity_pm_s = _m_to_pm(velocity_m_s) if velocity_m_s is not None else None
		try:
			for channel in (AXIS_X, AXIS_Y, AXIS_Z):
				ctl.SetProperty_i32(
					self._handle, channel,
					ctl.Property.REFERENCING_OPTIONS,
					int(referencing_options),
				)
				if velocity_pm_s is not None:
					ctl.SetProperty_i64(
						self._handle, channel,
						ctl.Property.MOVE_VELOCITY, velocity_pm_s,
					)
				ctl.Reference(self._handle, channel, 0)
		except ctl.Error as exc:
			raise _wrap_ctl("find_reference (start)", exc)
		for channel in (AXIS_X, AXIS_Y, AXIS_Z):
			self._wait(channel, timeout_s=timeout_s, cancel_event=cancel_event)
			if cancel_event is not None and cancel_event.is_set():
				# Skip remaining axes - the user asked us to abort.
				for ch in (AXIS_X, AXIS_Y, AXIS_Z):
					try:
						ctl.Stop(self._handle, ch)
					except Exception:
						pass
				return


# ---------------------------------------------------------------------------
# Helpers used by the GUIs
# ---------------------------------------------------------------------------

def speed_level_to_m_s(
		level: int,
		max_level: int,
		max_mm_s: float,
		table=None,
) -> float:
	"""Map a Simple-Mode speed level (1..max_level) to a velocity in m/s.

	If ``table`` is given, it is treated as a list of velocities in **mm/s**
	indexed by ``level - 1`` and used as a direct lookup.  This is how you
	pin the GUI to whatever velocity your MCS2 hand-control shows for each
	Simple Mode level.

	Otherwise falls back to a quadratic mapping where ``level == max_level``
	returns exactly ``max_mm_s * 1e-3``.
	"""
	if table:
		idx = max(1, min(int(level), len(table)))
		return float(table[idx - 1]) * 1e-3
	if max_level <= 0:
		return max_mm_s * 1e-3
	level = max(1, min(int(level), max_level))
	fraction = (level / max_level) ** 2
	return fraction * max_mm_s * 1e-3


def click_step_m(velocity_m_s: float, click_duration_s: float) -> float:
	"""Per-click jog distance derived from velocity.

	Higher slider level -> faster velocity -> larger jog per click.  The
	MCS2 hand-control behaves the same way: only speed is set, the actual
	distance scales with how long you deflect the joystick.  We model that
	with a fixed effective deflection time.
	"""
	return float(velocity_m_s) * float(click_duration_s)


def split_meters_mm_um_nm(value_m: float) -> tuple:
	"""Split a position in meters into (mm_int, um_int, nm_int) for the LCDs.

	Mirrors the SmarAct hand-control display, where each column shows the
	digits that belong to that decimal range.  Negative values are reported
	with a negative sign on the *mm* digit only - the other columns hold the
	positive remainder, which is also how the controller renders them.
	"""
	sign = -1 if value_m < 0 else 1
	abs_pm = int(round(abs(value_m) * M_TO_PM))
	nm_total = abs_pm // 1000  # picometers -> nanometers, drop sub-nm noise
	mm_int, rem_nm = divmod(nm_total, 1_000_000)
	um_int, nm_int = divmod(rem_nm, 1_000)
	return sign * mm_int, um_int, nm_int
