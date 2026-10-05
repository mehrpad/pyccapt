"""Manual smoke-test for the SmarAct MCS2 stage.

This is a *thin* interactive test harness — it does NOT re-implement any stage
logic.  All the real work lives in the production driver
:mod:`pyccapt.control.smaract_mcs2.mcs2_stage` (the same ``SmarActStage`` class
the Stage / Laser Control GUIs use), so running this script actually exercises
the code that ships.

Run it directly to discover the controllers, pick one, read the position and
do a small ``+10 um`` / ``-10 um`` round-trip on X, Y, Z::

    conda run -n oxcart python -m pyccapt.control.devices_test.stage_smartact

Units: every public driver method takes / returns **meters** (the SDK's native
picometer conversion is handled inside the driver).
"""

from pyccapt.control.smaract_mcs2 import mcs2_stage
from pyccapt.control.smaract_mcs2.mcs2_stage import (
    SmarActStage,
    SmarActStageError,
    list_devices,
)


def _print_position(stage: SmarActStage, label: str) -> None:
    pos = stage.get_position()
    print(
        f"{label:<22}"
        f"x={pos['x'] * 1e6:+10.3f} um  "
        f"y={pos['y'] * 1e6:+10.3f} um  "
        f"z={pos['z'] * 1e6:+10.3f} um"
    )


def _pick_locator(devices: list) -> str:
    """Interactively choose a device locator and verify it can be opened."""
    while True:
        print("\nAvailable devices:")
        for i, d in enumerate(devices, start=1):
            print(f"  [{i}] {d}")
        print("  [c] Enter a custom locator string (e.g. with a specific port)")

        raw = input(f"\nSelect device [1-{len(devices)} / c]: ").strip().lower()

        if raw == "c":
            print("  Locator examples:")
            print("    network:sn:MCS2-00017939          (default port)")
            print("    network:sn:MCS2-00017939:55552    (custom port)")
            print("    network:host:192.168.1.100        (by IP)")
            candidate = input("  Enter locator: ").strip()
            if not candidate:
                print("  Empty input, try again.")
                continue
        else:
            try:
                choice = int(raw)
            except ValueError:
                print("  Invalid input. Enter a number or 'c'.")
                continue
            if not (1 <= choice <= len(devices)):
                print(f"  Please enter a number between 1 and {len(devices)}, or 'c'.")
                continue
            candidate = devices[choice - 1]

        print(f"\nTrying: {candidate} ...")
        try:
            # Open + close just to confirm the controller is reachable; the
            # driver raises SmarActStageError with a human-readable hint on
            # failure (network timeout, busy connection slot, bad locator, ...).
            SmarActStage(candidate).close()
            print("  Connected successfully.")
            return candidate
        except SmarActStageError as exc:
            print(f"  ERROR: Cannot connect to '{candidate}'.\n  {exc}")
            print("  Please select a different device or try a custom locator.")


def main() -> int:
    if mcs2_stage.ctl is None:
        print(
            "\nERROR: 'smaract.ctl' SDK is not installed (it is NOT on PyPI).\n"
            "  Install it from the SmarAct MCS2 SDK, e.g.:\n"
            "    pip install \"C:/SmarAct/MCS2/SDK/Python/packages/smaract.ctl-<ver>.zip\""
        )
        return 1

    devices = list_devices()
    if not devices:
        print("ERROR: No MCS2 devices found. Check connection and power.")
        return 1

    print(f"\nFound {len(devices)} device(s):")
    for i, d in enumerate(devices, start=1):
        print(f"  [{i}] {d}")

    locator = _pick_locator(devices)
    print(f"\nUsing: {locator}")

    try:
        with SmarActStage(locator) as stage:
            _print_position(stage, "Initial position:")

            print("\nStep 1: Moving +10 um (forward) on X, Y, Z ...")
            stage.move_relative(dx_m=10e-6, dy_m=10e-6, dz_m=10e-6)
            _print_position(stage, "After +10 um move:")

            print("\nStep 2: Moving -10 um (backward) on X, Y, Z ...")
            stage.move_relative(dx_m=-10e-6, dy_m=-10e-6, dz_m=-10e-6)
            _print_position(stage, "After -10 um move:")
    except SmarActStageError as exc:
        print(f"\nERROR during stage operation:\n{exc}")
        return 1

    print("\nDone. Stage returned to original position.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
