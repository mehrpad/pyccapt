"""Extract ranging tables from MATLAB Atom-Probe-Toolbox ``.fig`` files.

The Atom-Probe-Toolbox at ``E:/Atom-Probe-Toolbox`` does its ranging on top of
a mass spectrum plot and saves the figure as ``.fig``. Each range is stored as
a ``specgraph.areaseries`` whose:

- ``XData(1)`` / ``XData(end)`` are the lower / upper m/c bounds
- ``FaceColor`` is the RGB tint (0..1)
- ``UserData.chargeState`` is the integer charge
- ``DisplayName`` is the human-readable ion label, e.g. ``"56Fe +"`` or
  ``"12C 16O2 +2"``

This module reads such a figure and produces a PyCCAPT-compatible range
``DataFrame`` with the 14-column schema used everywhere in the calibration
pipeline (see :func:`pyccapt.calibration.data_tools.data_tools.read_range`).

``UserData.plotType`` and ``UserData.ion`` are stored as MATLAB MCOS opaque
objects (``string`` / ``table`` classes) that can't be decoded without
emulating the MCOS subsystem. We don't need them: the presence of
``chargeState`` in ``UserData`` is itself the range discriminator, and
``DisplayName`` carries the same composition information in plain text.

Usage from a terminal
---------------------

Convert a single figure (writes ``<stem>_range.h5`` next to the input)::

    python -m pyccapt.calibration.leap_tools.matlab_fig_range \
        "E:/ai_driven_apt_3d_reconstruction/data/LEAP_Erik/A718_08676/Massspectrum.fig"

Walk a folder of datasets, converting every ``Massspectrum*.fig`` found::

    python -m pyccapt.calibration.leap_tools.matlab_fig_range \
        "E:/ai_driven_apt_3d_reconstruction/data/LEAP_Erik" --recursive

Only versions matching a glob (e.g. only the "pretty" figures)::

    python -m pyccapt.calibration.leap_tools.matlab_fig_range \
        "E:/datasets" --recursive --pattern "Massspectrum_pretty.fig"

The output ``<stem>_range.h5`` (and a sibling ``<stem>_range.csv``) is picked
up by the PyCCAPT data loader when the dataset path is named ``<stem>.h5``;
for a generic ``Massspectrum.fig`` you'll typically want ``--output`` to
rename it so it sits next to your processed dataset, e.g.
``--output ../R56_08676_range.h5``.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# scipy is already a runtime dependency of pyccapt (data_tools imports it).
import scipy.io


_PYCCAPT_RANGE_COLUMNS = [
    "name",
    "ion",
    "mass",
    "mc",
    "mc_low",
    "mc_up",
    "color",
    "element",
    "complex",
    "isotope",
    "charge",
    "vol",
    "raw_comp",
    "ion_name",
]

_ATOM_TOKEN_RE = re.compile(r"(?P<isotope>\d+)?(?P<element>[A-Z][a-z]?)(?P<count>\d+)?")
_TRAILING_CHARGE_RE = re.compile(r"\s*(?P<block>[+-]+\s*\d*)\s*$")


def _unwrap(value):
    """Unwrap repeated ``(1, 1)`` object arrays produced by ``loadmat``."""
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.flat[0]
    return value


def _node_type(node) -> str:
    """Return the HG node ``type`` string (``'figure'``, ``'axes'``, ...)."""
    raw = getattr(node, "type", None)
    if raw is None:
        return ""
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return ""
        first = raw.flat[0]
        return first if isinstance(first, str) else str(first)
    return str(raw)


def _iter_children(node) -> Iterable:
    """Yield each unwrapped child of a HG ``mat_struct`` node."""
    children = getattr(node, "children", None)
    if children is None or not isinstance(children, np.ndarray) or children.size == 0:
        return []
    return (_unwrap(child) for child in children.ravel())


def _scalar(value):
    """Return the first scalar from a ``(1, N)`` ndarray, or ``value`` itself."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.flat[0]
    return value


def _to_str(value) -> str:
    """Coerce a MATLAB-imported scalar / array to a plain Python str."""
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return str(value.flat[0])
    return str(value)


def _hex_color(face_color) -> str:
    """Convert a MATLAB FaceColor RGB triple (0..1) to ``#RRGGBB``."""
    if face_color is None:
        return "#808080"
    if isinstance(face_color, np.ndarray):
        rgb = face_color.ravel().astype(float)
    else:
        rgb = np.asarray(face_color, dtype=float).ravel()
    if rgb.size < 3:
        return "#808080"
    rgb_255 = np.clip(np.round(rgb[:3] * 255.0), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb_255.tolist())


def parse_display_name(display_name: str) -> dict:
    """Parse a MATLAB-toolbox ``DisplayName`` like ``'56Fe +2'``.

    Returns a dict with ``elements``, ``counts``, ``isotopes``, ``charge`` and
    ``neutral_name`` keys. The charge in ``DisplayName`` is either ``+``
    repetitions (``'+++'`` = 3) or a sign followed by a digit (``'+2'``); both
    are accepted. Unparsable strings yield a row with no elements but the raw
    text stored as ``raw_comp``.
    """
    text = display_name.strip()
    raw = text

    charge = 1
    trailing = _TRAILING_CHARGE_RE.search(text)
    if trailing:
        sign_block = trailing.group("block").strip()
        text = text[: trailing.start()].rstrip()
        digits = re.search(r"\d+", sign_block)
        if digits:
            charge = int(digits.group(0))
        else:
            charge = sign_block.count("+") - sign_block.count("-")
            if charge == 0:
                charge = 1
        if sign_block.startswith("-"):
            charge = -abs(charge)

    elements: list[str] = []
    counts: list[int] = []
    isotopes: list = []

    for token in text.split():
        for match in _ATOM_TOKEN_RE.finditer(token):
            element = match.group("element")
            if not element:
                continue
            isotope = int(match.group("isotope")) if match.group("isotope") else None
            count = int(match.group("count")) if match.group("count") else 1
            elements.append(element)
            counts.append(count)
            isotopes.append(isotope)

    if elements:
        parts = []
        for element, count in zip(elements, counts):
            parts.append(f"{element}{'' if count == 1 else count}")
        neutral_name = "".join(parts)
    else:
        neutral_name = raw

    return {
        "elements": elements,
        "counts": counts,
        "isotopes": isotopes,
        "charge": int(charge),
        "neutral_name": neutral_name,
        "raw_comp": raw,
    }


def _build_pyccapt_ion_label(parsed: dict) -> str:
    """Return a LaTeX-style ion label compatible with the calibration legend."""
    if not parsed["elements"]:
        return parsed["raw_comp"]
    parts = []
    for element, count, isotope in zip(parsed["elements"], parsed["counts"], parsed["isotopes"]):
        iso = f"{{}}^{{{isotope}}}" if isotope else ""
        stoich = "" if count == 1 else f"_{{{count}}}"
        parts.append(f"{iso}{element}{stoich}")
    charge = parsed["charge"]
    if charge == 1:
        charge_token = "^{+}"
    elif charge == -1:
        charge_token = "^{-}"
    elif charge > 0:
        charge_token = f"^{{{charge}+}}"
    else:
        charge_token = f"^{{{abs(charge)}-}}"
    return "$" + "".join(parts) + charge_token + "$"


def _is_range_areaseries(node) -> bool:
    """Return True when this ``specgraph.areaseries`` carries a range payload."""
    if "areaseries" not in _node_type(node).lower():
        return False
    props = _unwrap(getattr(node, "properties", None))
    if props is None or not hasattr(props, "_fieldnames"):
        return False
    user_data = _unwrap(getattr(props, "UserData", None))
    if user_data is None or not hasattr(user_data, "_fieldnames"):
        return False
    return "chargeState" in user_data._fieldnames


def _collect_range_areaseries(node, accumulator: list) -> None:
    """Depth-first collect every range areaseries below ``node``."""
    if _is_range_areaseries(node):
        accumulator.append(node)
        return
    for child in _iter_children(node):
        _collect_range_areaseries(child, accumulator)


def _load_fig_struct(fig_path: Path):
    """Load the top-level ``hgS_*`` figure struct from a saved ``.fig`` file."""
    contents = scipy.io.loadmat(
        str(fig_path),
        squeeze_me=False,
        struct_as_record=False,
    )
    hgs_keys = [k for k in contents if k.startswith("hgS_")]
    if not hgs_keys:
        raise ValueError(
            f"{fig_path} does not look like a MATLAB figure: no 'hgS_*' struct found. "
            "If it was saved with MATLAB v7.3 (HDF5) format this reader does not yet "
            "support it; resave the .fig from MATLAB as '-v7' or earlier."
        )
    return _unwrap(contents[hgs_keys[0]])


def fig_to_range_dataframe(fig_path) -> pd.DataFrame:
    """Return a PyCCAPT-style range dataframe extracted from one ``.fig`` file."""
    fig_path = Path(fig_path).expanduser()
    root = _load_fig_struct(fig_path)

    found: list = []
    _collect_range_areaseries(root, found)

    if not found:
        return pd.DataFrame(columns=_PYCCAPT_RANGE_COLUMNS)

    rows = []
    for node in found:
        props = _unwrap(node.properties)
        user_data = _unwrap(props.UserData)

        x_data = props.XData
        x_flat = np.asarray(x_data).ravel()
        mc_low = float(x_flat[0])
        mc_up = float(x_flat[-1])
        face_color = _hex_color(getattr(props, "FaceColor", None))

        display_name = _to_str(getattr(props, "DisplayName", "")) or _to_str(
            getattr(user_data, "DisplayName", "")
        )
        charge_state = _scalar(user_data.chargeState)
        if display_name:
            parsed = parse_display_name(display_name)
        else:
            parsed = {
                "elements": [],
                "counts": [],
                "isotopes": [],
                "charge": int(charge_state) if charge_state is not None else 1,
                "neutral_name": "",
                "raw_comp": "",
            }
        # UserData.chargeState is authoritative; override the DisplayName guess.
        if charge_state is not None:
            try:
                parsed["charge"] = int(charge_state)
            except (TypeError, ValueError):
                pass

        rows.append({
            "name": parsed["neutral_name"] or display_name,
            "ion": _build_pyccapt_ion_label(parsed),
            "mass": (mc_low + mc_up) / 2.0,
            "mc": (mc_low + mc_up) / 2.0,
            "mc_low": mc_low,
            "mc_up": mc_up,
            "color": face_color,
            "element": list(parsed["elements"]),
            "complex": list(parsed["counts"]),
            "isotope": [iso if iso is not None else 0 for iso in parsed["isotopes"]],
            "charge": int(parsed["charge"]),
            "vol": 0.0,
            "raw_comp": parsed["raw_comp"] or display_name,
            "ion_name": parsed["neutral_name"] or display_name,
        })

    frame = pd.DataFrame(rows, columns=_PYCCAPT_RANGE_COLUMNS)
    frame.sort_values("mc_low", inplace=True, ignore_index=True)
    return frame


def write_pyccapt_range(frame: pd.DataFrame, output_h5, also_csv: bool = True) -> dict:
    """Write the range dataframe as a PyCCAPT-compatible ``.h5`` (+ ``.csv``)."""
    output_h5 = Path(output_h5).expanduser()
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    written: dict = {}
    try:
        frame.to_hdf(output_h5, key="df", mode="w")
    except Exception:
        try:
            output_h5.unlink()
        except OSError:
            pass
        frame.to_hdf(output_h5, key="df", mode="w", format="table")
    written["h5"] = str(output_h5)
    if also_csv:
        csv_path = output_h5.with_suffix(".csv")
        frame.to_csv(csv_path, encoding="utf-8", index=False, sep=";")
        written["csv"] = str(csv_path)
    return written


def _find_figs(root: Path, recursive: bool, pattern: str) -> list:
    glob_pattern = f"**/{pattern}" if recursive else pattern
    return sorted({p.resolve() for p in root.glob(glob_pattern) if p.is_file()})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matlab-fig-range",
        description=(
            "Extract ranging from MATLAB Atom-Probe-Toolbox saved figures (.fig) "
            "and write a PyCCAPT-compatible <stem>_range.h5 next to each input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .fig file, or to a folder containing .fig files.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning a folder.",
    )
    parser.add_argument(
        "--pattern",
        default="*.fig",
        help="Glob pattern for figures when path is a folder (default: '*.fig').",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .h5 path. Only valid when 'path' is a single .fig file. "
            "When omitted, writes <stem>_range.h5 next to each input."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the sibling CSV alongside the .h5 range file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <stem>_range.h5 outputs (otherwise skipped).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full tracebacks for any file that fails to convert.",
    )
    return parser


def _resolve_output_path(fig_path: Path, override) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return fig_path.with_name(f"{fig_path.stem}_range.h5")


def _process_one(fig_path: Path, output_path: Path, also_csv: bool, overwrite: bool) -> dict:
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "h5": str(output_path), "ranges": None}
    frame = fig_to_range_dataframe(fig_path)
    written = write_pyccapt_range(frame, output_path, also_csv=also_csv)
    written["status"] = "ok"
    written["ranges"] = int(len(frame))
    return written


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path = args.path.expanduser().resolve()
    if path.is_file():
        targets = [path]
        output_override = args.output
    elif path.is_dir():
        if args.output is not None:
            parser.error("--output is only valid when 'path' is a single .fig file")
        targets = _find_figs(path, args.recursive, args.pattern)
        if not targets:
            print(f"No figures matching {args.pattern!r} found under {path}")
            return 0
        output_override = None
    else:
        parser.error(f"{path} is neither a file nor a directory")

    print(f"Converting {len(targets)} figure(s) from {path}")
    failed = []
    n_ok = n_skipped = 0
    for index, fig_path in enumerate(targets, start=1):
        output_path = _resolve_output_path(fig_path, output_override)
        print(f"[{index}/{len(targets)}] {fig_path}")
        try:
            result = _process_one(fig_path, output_path, also_csv=not args.no_csv, overwrite=args.overwrite)
        except Exception as exc:
            failed.append((fig_path, f"{type(exc).__name__}: {exc}"))
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
            continue
        if result["status"] == "skipped":
            n_skipped += 1
            print(f"    skipped (output exists): {result['h5']}")
        else:
            n_ok += 1
            print(f"    wrote: {result['h5']}  ({result['ranges']} ranges)")
            if "csv" in result:
                print(f"    wrote: {result['csv']}")

    print(f"\nDone. converted={n_ok}, skipped={n_skipped}, failed={len(failed)}")
    if failed:
        print("Failed figures:")
        for fig_path, reason in failed:
            print(f"  {fig_path}  --  {reason}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
