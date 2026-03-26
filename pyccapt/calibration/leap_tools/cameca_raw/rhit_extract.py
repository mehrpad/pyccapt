"""CLI wrapper for extracting RHIT files to HDF5."""

from pyccapt.calibration.leap_tools.cameca_raw.rhit_tools import _main


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
