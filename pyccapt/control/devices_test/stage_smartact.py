"""
SmarAct MCS2 Stage Controller
------------------------------
Uses the SmarAct `smaract.ctl` Python package (MCS2 SDK).

Units:
    - The MCS2 SDK uses picometers (pm) internally.
    - All public functions in this module accept values in **meters (m)**.
      Conversion: 1 m = 1e12 pm

Axis mapping (channel index on the MCS2 controller):
    AXIS_X = 0
    AXIS_Y = 1
    AXIS_Z = 2

Common errors
-------------
61466 / 0xF01A  SA_CTL_ERROR_NETWORK_TIMEOUT
    The controller was found via FindDevices() but the TCP connection timed out.
    Causes:
      * The MCS2 controller is powered off or not reachable on the network.
      * Another process (e.g. SmarActServiceTool, a previous Python session) still
        holds an open handle.  Restart the controller or kill the other process.
      * Wrong network interface / firewall blocking port 55551.
"""

import time

import smaract.ctl as ctl

# ---------------------------------------------------------------------------
# Axis channel mapping
# ---------------------------------------------------------------------------
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
M_TO_PM = 1e12   # meters  -> picometers
PM_TO_M = 1e-12  # picometers -> meters


def _m_to_pm(value_m: float) -> int:
    """Convert meters to picometers (MCS2 native unit)."""
    return int(round(value_m * M_TO_PM))


def _pm_to_m(value_pm: int) -> float:
    """Convert picometers to meters."""
    return value_pm * PM_TO_M


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def list_devices() -> list:
    """
    Return a list of all reachable MCS2 device locator strings.

    Useful for diagnosing connection problems before calling any other function.
    """
    buffer = ctl.FindDevices()
    if not buffer:
        return []
    return [loc.strip() for loc in buffer.split("\n") if loc.strip()]


def _find_controller():
    """
    Return the locator string for the first MCS2 controller that can be opened.

    Tries every device returned by FindDevices() in order and returns the
    locator of the first one that responds.  Raises RuntimeError if none work.
    """
    devices = list_devices()
    if not devices:
        raise RuntimeError(
            "No SmarAct MCS2 device found via FindDevices(). "
            "Check USB/network connection and that the MCS2 controller is powered on."
        )
    last_exc = None
    for locator in devices:
        try:
            handle = ctl.Open(locator)
            ctl.Close(handle)
            return locator
        except ctl.Error as exc:
            last_exc = (exc, locator)
    # All devices failed
    exc, locator = last_exc
    _raise_friendly(exc, locator)


def _open_controller(locator: str = None):
    """Open a connection to the MCS2 controller and return the handle."""
    if locator is None:
        locator = _find_controller()
    try:
        handle = ctl.Open(locator)
    except ctl.Error as exc:
        _raise_friendly(exc, locator)
    return handle


def _close_controller(handle):
    """Close the connection to the MCS2 controller."""
    try:
        ctl.Close(handle)
    except Exception:
        pass  # best-effort close


def _raise_friendly(exc: "ctl.Error", locator: str):
    """Re-raise a ctl.Error with a human-readable message."""
    code = exc.args[1] if len(exc.args) >= 2 else "?"
    hints = {
        0xF01A: (
            "SA_CTL_ERROR_NETWORK_TIMEOUT – controller found but TCP connection timed out.\n"
            "  • Is the MCS2 powered on and reachable? (try ping / SmarActServiceTool)\n"
            "  • Another process may hold an open handle – restart the controller.\n"
            "  • Firewall may be blocking port 55551."
        ),
        0xF003: "SA_CTL_ERROR_NOT_FOUND – device not found. Check the locator string.",
        0xF005: f"SA_CTL_ERROR_INVALID_LOCATOR – '{locator}' is not a valid locator.",
        0xF00C: (
            "SA_CTL_ERROR_DEVICE_LIMIT_REACHED – too many open handles. "
            "Close other sessions first."
        ),
    }
    hint = hints.get(code, "")
    msg = f"ctl.Open('{locator}') failed with error code {code:#06x}."
    if hint:
        msg += f"\n  Hint: {hint}"
    raise RuntimeError(msg) from exc


def _wait_for_channel_stop(handle, channel: int, timeout_s: float = 30.0):
    """
    Block until the given channel is no longer moving or timeout is reached.

    Raises:
        TimeoutError: if the stage has not stopped within *timeout_s* seconds.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = ctl.GetProperty_i32(handle, channel, ctl.Property.CHANNEL_STATE)
        if not (state & ctl.ChannelState.ACTIVELY_MOVING):
            return
        time.sleep(0.02)
    raise TimeoutError(f"Channel {channel} did not stop within {timeout_s} s.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def move_relative(dx_m: float = 0.0,
                  dy_m: float = 0.0,
                  dz_m: float = 0.0,
                  velocity_m_s: float = 1e-3,
                  locator: str = None,
                  wait: bool = True):
    """
    Move the stage by the given relative distances (in meters).

    Parameters
    ----------
    dx_m, dy_m, dz_m : float
        Relative displacement in **meters** for X, Y, Z respectively.
        Example: 10e-6  ->  10 um
    velocity_m_s : float
        Maximum move velocity in m/s (default 1 mm/s).
    locator : str, optional
        MCS2 device locator string.  Auto-detected if not given.
    wait : bool
        If True, block until all axes have stopped moving.
    """
    handle = _open_controller(locator)
    try:
        velocity_pm_s = _m_to_pm(velocity_m_s)
        axes = [(AXIS_X, dx_m), (AXIS_Y, dy_m), (AXIS_Z, dz_m)]

        for channel, delta_m in axes:
            if delta_m == 0.0:
                continue
            ctl.SetProperty_i32(handle, channel, ctl.Property.MOVE_MODE,
                                ctl.MoveMode.CL_RELATIVE)
            ctl.SetProperty_i64(handle, channel, ctl.Property.MOVE_VELOCITY,
                                velocity_pm_s)
            ctl.Move(handle, channel, _m_to_pm(delta_m), 0)

        if wait:
            for channel, delta_m in axes:
                if delta_m == 0.0:
                    continue
                _wait_for_channel_stop(handle, channel)
    finally:
        _close_controller(handle)


def move_absolute(x_m: float = None,
                  y_m: float = None,
                  z_m: float = None,
                  velocity_m_s: float = 1e-3,
                  locator: str = None,
                  wait: bool = True):
    """
    Move the stage to the given absolute positions (in meters).

    Parameters
    ----------
    x_m, y_m, z_m : float or None
        Target absolute position in **meters**.  Pass None to skip an axis.
    velocity_m_s : float
        Maximum move velocity in m/s (default 1 mm/s).
    locator : str, optional
        MCS2 device locator string.  Auto-detected if not given.
    wait : bool
        If True, block until all axes have stopped moving.
    """
    handle = _open_controller(locator)
    try:
        velocity_pm_s = _m_to_pm(velocity_m_s)
        axes = [(AXIS_X, x_m), (AXIS_Y, y_m), (AXIS_Z, z_m)]

        for channel, pos_m in axes:
            if pos_m is None:
                continue
            ctl.SetProperty_i32(handle, channel, ctl.Property.MOVE_MODE,
                                ctl.MoveMode.CL_ABSOLUTE)
            ctl.SetProperty_i64(handle, channel, ctl.Property.MOVE_VELOCITY,
                                velocity_pm_s)
            ctl.Move(handle, channel, _m_to_pm(pos_m), 0)

        if wait:
            for channel, pos_m in axes:
                if pos_m is None:
                    continue
                _wait_for_channel_stop(handle, channel)
    finally:
        _close_controller(handle)


def get_position(locator: str = None) -> dict:
    """
    Read the current position of all three axes.

    Returns
    -------
    dict
        {'x': float, 'y': float, 'z': float} in **meters**.
    """
    handle = _open_controller(locator)
    try:
        pos = {}
        for name, channel in [('x', AXIS_X), ('y', AXIS_Y), ('z', AXIS_Z)]:
            pm = ctl.GetProperty_i64(handle, channel, ctl.Property.POSITION)
            pos[name] = _pm_to_m(pm)
        return pos
    finally:
        _close_controller(handle)


def find_reference(locator: str = None):
    """
    Run the reference search (homing) procedure on all three axes.

    Moves each axis to its reference mark and zeros the position.
    Blocks until the procedure is complete.
    """
    handle = _open_controller(locator)
    try:
        for channel in [AXIS_X, AXIS_Y, AXIS_Z]:
            ctl.SetProperty_i32(handle, channel,
                                ctl.Property.REFERENCING_OPTIONS, 0)
            ctl.Reference(handle, channel, 0)
        for channel in [AXIS_X, AXIS_Y, AXIS_Z]:
            _wait_for_channel_stop(handle, channel, timeout_s=120.0)
    finally:
        _close_controller(handle)


# ---------------------------------------------------------------------------
# Quick-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 1. List all devices found on the network/USB ----------------------
    devices = list_devices()
    if not devices:
        print("ERROR: No MCS2 devices found. Check connection and power.")
        raise SystemExit(1)

    print(f"Found {len(devices)} device(s):")
    for d in devices:
        print(f"  {d}")

    # --- 2. Pick the first device that actually responds -------------------
    locator = None
    for candidate in devices:
        try:
            h = ctl.Open(candidate)
            ctl.Close(h)
            locator = candidate
            print(f"\nUsing: {locator}")
            break
        except ctl.Error as e:
            print(f"  {candidate}  -> unreachable (error {e.args[1]:#06x}), skipping")

    if locator is None:
        print("\nERROR: No reachable MCS2 controller found.")
        raise SystemExit(1)

    # --- 3. Try to open and read position ----------------------------------
    try:
        pos = get_position(locator)
        print(f"Current position: x={pos['x']*1e6:.3f} um  "
              f"y={pos['y']*1e6:.3f} um  z={pos['z']*1e6:.3f} um")
    except RuntimeError as e:
        print(f"\nERROR reading position:\n{e}")
        raise SystemExit(1)

    # --- 4. Demo: move +10 um on all axes ----------------------------------
    print("\nMoving +10 um in X, Y, and Z ...")
    try:
        move_relative(dx_m=10e-6, dy_m=10e-6, dz_m=10e-6, locator=locator)
        pos = get_position(locator)
        print(f"Position after move: x={pos['x']*1e6:.3f} um  "
              f"y={pos['y']*1e6:.3f} um  z={pos['z']*1e6:.3f} um")
    except RuntimeError as e:
        print(f"\nERROR during move:\n{e}")
        raise SystemExit(1)
