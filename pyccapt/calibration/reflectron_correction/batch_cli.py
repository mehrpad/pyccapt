"""Batch reflectron correction over a folder of EPOS files.

Walks a folder for ``.epos`` files, applies the affine reflectron-mesh
correction for the chosen instrument preset to each one, and writes a corrected
``.h5`` (and optionally ``.epos``) next to every input file.

Usage from a terminal
---------------------

List the available instrument presets::

    python -m pyccapt.calibration.reflectron_correction.batch_cli --list-instruments

Correct every EPOS file in a folder (non-recursive) with one preset::

    python -m pyccapt.calibration.reflectron_correction.batch_cli \
        "E:/datasets/run_42" \
        --instrument 5000xr_oxford_5083_23091

Recurse into subdirectories, also write a corrected ``.epos`` next to each
``.h5``, and overwrite existing outputs::

    python -m pyccapt.calibration.reflectron_correction.batch_cli \
        "E:/datasets" \
        --instrument 3000xhr_leoben_21_14093 \
        --recursive --save-epos --overwrite

Remove every ``*_corrected.h5`` / ``*_corrected.epos`` previously written under a
folder (e.g. to redo a batch from scratch). Preview first with ``--dry-run``::

    python -m pyccapt.calibration.reflectron_correction.batch_cli "E:/datasets" --clean --recursive --dry-run
    python -m pyccapt.calibration.reflectron_correction.batch_cli "E:/datasets" --clean --recursive

Notes
-----
- The instrument name accepts either the preset key (e.g. ``5000xr_oxford_5083_23091``)
  or the human-readable display name (case/space/dash-insensitive).
- Output files are written beside each input as ``<input_stem>_corrected.h5``
  (and ``<input_stem>_corrected.epos`` when ``--save-epos`` is set).
- Existing outputs are **skipped automatically** on re-runs, so you can re-invoke
  the same command to retry only the files that failed previously. Use
  ``--overwrite`` to force a fresh write.
- Failures on a single EPOS file (corrupt input, pytables error, etc.) are caught
  and logged as one line; the run continues with the next folder. The summary at
  the end lists every failed path. Add ``--verbose`` to also see full tracebacks.
- The HDF5 writer falls back from pytables ``fixed`` to ``table`` format when the
  fixed format raises ``HDF5ExtError`` ("Problems creating the Array"), which
  affects some EPOS files. The fallback file is still readable with
  ``pandas.read_hdf``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

from pyccapt.calibration.leap_tools import ccapt_tools, matlab_fig_range
from pyccapt.calibration.reflectron_correction import core as reflectron_core


# When asked to convert a MATLAB ``.fig`` range alongside an EPOS we walk this
# many parent levels above the EPOS looking for a matching figure. The Atom-
# Probe-Toolbox saves ``Massspectrum.fig`` at the dataset root which sits 2-5
# levels above the actual ``.epos`` (``<dataset>/recons/<name>/default/x.epos``).
_DEFAULT_FIG_SEARCH_DEPTH = 6

# Headroom (in bytes) we want free before attempting a fresh HDF5 write. Pytables
# / HDF5 don't fail gracefully when the disk fills up mid-write; they emit a
# multi-page backtrace ending in errno=28. Bailing out early with a clear message
# is much nicer.
_PER_FILE_DISK_HEADROOM_BYTES = 512 * 1024 * 1024  # 512 MiB


def _find_epos_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.epos" if recursive else "*.epos"
    return sorted({p.resolve() for p in root.glob(pattern) if p.is_file()})


def _find_corrected_outputs(root: Path, recursive: bool) -> list[Path]:
    """Return every ``*_corrected.h5`` and ``*_corrected.epos`` file under ``root``."""
    patterns = (
        ("**/*_corrected.h5", "**/*_corrected.epos")
        if recursive
        else ("*_corrected.h5", "*_corrected.epos")
    )
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path.resolve())
    return sorted(found)


def _clean_corrected_outputs(root: Path, recursive: bool, dry_run: bool) -> int:
    """Remove ``*_corrected.h5``/``*_corrected.epos`` files under ``root``."""
    targets = _find_corrected_outputs(root, recursive)
    if not targets:
        print(f"No *_corrected.h5/.epos files found under {root}")
        return 0

    action = "Would delete" if dry_run else "Deleting"
    print(f"{action} {len(targets)} file(s) under {root}:")
    n_removed = n_failed = 0
    for path in targets:
        if dry_run:
            print(f"  {path}")
            continue
        try:
            path.unlink()
            print(f"  removed: {path}")
            n_removed += 1
        except OSError as exc:
            print(f"  FAILED to remove {path}: {exc}")
            n_failed += 1

    if dry_run:
        print(f"\nDry run. {len(targets)} file(s) would be removed. Re-run without --dry-run to delete.")
        return 0
    print(f"\nDone. removed={n_removed}, failed={n_failed}")
    return 0 if n_failed == 0 else 1


def _free_bytes(path: Path) -> int:
    """Return free bytes on the filesystem hosting ``path`` (or its parent)."""
    target = path
    while not target.exists():
        if target.parent == target:
            break
        target = target.parent
    try:
        return shutil.disk_usage(str(target)).free
    except OSError:
        return -1


def _format_size(n_bytes: int) -> str:
    """Format ``n_bytes`` as a short human-readable string."""
    n = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def _disk_has_room(out_path: Path, source_path: Path) -> tuple[bool, int, int]:
    """Return ``(ok, free_bytes, needed_bytes)`` for a planned write.

    We require roughly the source EPOS size plus a fixed headroom -- the
    corrected HDF5 ends up similar in size to the raw EPOS, and pytables also
    needs scratch room while it builds its chunks.
    """
    try:
        needed = source_path.stat().st_size + _PER_FILE_DISK_HEADROOM_BYTES
    except OSError:
        needed = _PER_FILE_DISK_HEADROOM_BYTES
    free = _free_bytes(out_path)
    if free < 0:
        return True, free, needed  # Can't check; let the write try.
    return free >= needed, free, needed


def _find_matlab_range_fig(epos_path: Path, pattern: str, max_depth: int) -> Path | None:
    """Walk parents of ``epos_path`` looking for a MATLAB toolbox range fig.

    Returns the most-recently-modified match within ``max_depth`` parent levels,
    or ``None`` if nothing matched. Older Atom-Probe-Toolbox runs save figures
    like ``Massspectrum.fig``, ``Massspectrum_correct width.fig`` and
    ``Massspectrum_pretty.fig`` -- prefer ``Massspectrum.fig`` exactly, then
    fall back to whichever ``pattern`` match is newest.
    """
    parents = [epos_path.parent, *epos_path.parents][:max_depth + 1]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for parent in parents:
        if parent in seen or not parent.is_dir():
            continue
        seen.add(parent)
        for match in parent.glob(pattern):
            if match.is_file():
                candidates.append(match.resolve())
    if not candidates:
        return None
    exact = [c for c in candidates if c.name.lower() == "massspectrum.fig"]
    if exact:
        return max(exact, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _write_matlab_range_for(epos_path: Path, h5_path: Path, *, pattern: str, max_depth: int, overwrite: bool) -> dict | None:
    """Convert a sibling MATLAB ``.fig`` (if any) to a pyccapt range next to ``h5_path``.

    Returns a dict describing the work (``{'fig', 'range_h5', 'range_csv',
    'status'}``) or ``None`` when no figure was found.
    """
    fig_path = _find_matlab_range_fig(epos_path, pattern, max_depth)
    if fig_path is None:
        return None
    range_h5 = h5_path.with_name(f"{h5_path.stem}_range.h5")
    if range_h5.exists() and not overwrite:
        return {"status": "skipped", "fig": str(fig_path), "range_h5": str(range_h5)}
    frame = matlab_fig_range.fig_to_range_dataframe(fig_path)
    matlab_fig_range.write_pyccapt_range(frame, range_h5, also_csv=True)
    return {
        "status": "ok",
        "fig": str(fig_path),
        "range_h5": str(range_h5),
        "range_csv": str(range_h5.with_suffix(".csv")),
        "ranges": int(len(frame)),
    }


def _write_dataframe_to_hdf(dataframe, h5_path: Path) -> None:
    """Write a corrected dataframe to ``h5_path`` robustly.

    Pytables' default ``fixed`` format occasionally raises
    ``tables.exceptions.HDF5ExtError`` ("Problems creating the Array") for
    specific EPOS files. The ``table`` format is more permissive and is
    transparently readable by ``pd.read_hdf``, so we fall back to it on any
    pytables failure.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_hdf(h5_path, key="df", mode="w")
        return
    except Exception:
        try:
            h5_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    dataframe.to_hdf(h5_path, key="df", mode="w", format="table")


def _correct_single_file(
    epos_path: Path,
    mesh: reflectron_core.ReflectronMesh,
    *,
    save_epos: bool,
    overwrite: bool,
    with_matlab_range: bool = False,
    matlab_range_pattern: str = "Massspectrum*.fig",
    matlab_range_depth: int = _DEFAULT_FIG_SEARCH_DEPTH,
) -> dict:
    h5_path = epos_path.with_name(f"{epos_path.stem}_corrected.h5")
    epos_out_path = epos_path.with_name(f"{epos_path.stem}_corrected.epos")

    if not overwrite and h5_path.exists() and (not save_epos or epos_out_path.exists()):
        result: dict = {"status": "skipped", "h5": str(h5_path)}
        if with_matlab_range:
            range_info = _write_matlab_range_for(
                epos_path,
                h5_path,
                pattern=matlab_range_pattern,
                max_depth=matlab_range_depth,
                overwrite=overwrite,
            )
            if range_info is not None:
                result["matlab_range"] = range_info
        return result

    # Pre-flight: if there's clearly no room for this write, bail out with a
    # clean message instead of letting pytables die mid-stream with errno=28.
    ok, free, needed = _disk_has_room(h5_path, epos_path)
    if not ok:
        raise OSError(
            f"Disk full: only {_format_size(free)} free on output drive but "
            f"~{_format_size(needed)} required for {h5_path.name}. Free space "
            "and re-run -- already-written outputs are skipped automatically."
        )

    written: dict = {"status": "ok"}

    # When --save-epos is set we still need the full corrected DataFrame to
    # write the binary EPOS, so we fall back to the eager in-memory path. When
    # only the HDF5 output is requested (the common case), use the streaming
    # corrector that keeps peak RAM bounded to a single chunk regardless of
    # the EPOS size.
    if save_epos:
        raw = reflectron_core.load_epos_for_reflectron_correction(epos_path)
        corrected = reflectron_core.apply_reflectron_correction_to_ccapt(raw, mesh)
        if overwrite or not h5_path.exists():
            _write_dataframe_to_hdf(corrected, h5_path)
        if overwrite or not epos_out_path.exists():
            ccapt_tools.ccapt_to_epos(
                corrected,
                path=str(epos_out_path.parent) + "\\",
                name=epos_out_path.name,
            )
            written["epos"] = str(epos_out_path)
    else:
        if overwrite or not h5_path.exists():
            reflectron_core.correct_epos_streaming(epos_path, mesh, h5_path)
    written["h5"] = str(h5_path)

    if with_matlab_range:
        range_info = _write_matlab_range_for(
            epos_path,
            h5_path,
            pattern=matlab_range_pattern,
            max_depth=matlab_range_depth,
            overwrite=overwrite,
        )
        if range_info is not None:
            written["matlab_range"] = range_info

    return written


def _build_parser() -> argparse.ArgumentParser:
    presets = reflectron_core.list_builtin_presets()
    preset_help = "\n".join(f"  {key:<30s}  {name}" for key, name in presets.items())

    parser = argparse.ArgumentParser(
        prog="reflectron-batch",
        description=(
            "Apply reflectron correction to every .epos file in a folder and "
            "save the corrected dataset as <stem>_corrected.h5 beside it."
        ),
        epilog="Available instrument presets:\n" + preset_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Folder to scan for .epos files (omit when using --list-instruments).",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        help="Instrument preset key or display name (see --list-instruments).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for .epos files.",
    )
    parser.add_argument(
        "--save-epos",
        action="store_true",
        help="Also write a corrected .epos file next to the .h5 output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing _corrected outputs (otherwise files are skipped).",
    )
    parser.add_argument(
        "--list-instruments",
        action="store_true",
        help="Print the available instrument presets and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full Python tracebacks for files that fail (default: one-line error).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete every *_corrected.h5 and *_corrected.epos file under the folder "
            "(respects --recursive). No correction is run. Combine with --dry-run "
            "to preview the deletions first."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --clean, list the files that would be removed without deleting them.",
    )
    parser.add_argument(
        "--with-matlab-range",
        action="store_true",
        help=(
            "After writing each *_corrected.h5, search up to "
            f"{_DEFAULT_FIG_SEARCH_DEPTH} parent levels above the EPOS for a MATLAB "
            "Atom-Probe-Toolbox range figure and convert it to a pyccapt-style "
            "<stem>_corrected_range.h5 next to the corrected output."
        ),
    )
    parser.add_argument(
        "--matlab-range-pattern",
        default="Massspectrum*.fig",
        help=(
            "Glob pattern (in each parent folder) for the MATLAB range figure when "
            "--with-matlab-range is set. Default: 'Massspectrum*.fig'."
        ),
    )
    parser.add_argument(
        "--matlab-range-depth",
        type=int,
        default=_DEFAULT_FIG_SEARCH_DEPTH,
        help=(
            "How many parent levels above each EPOS to search for the MATLAB range "
            f"figure. Default: {_DEFAULT_FIG_SEARCH_DEPTH}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_instruments:
        for key, name in reflectron_core.list_builtin_presets().items():
            print(f"{key}\t{name}")
        return 0

    if args.clean:
        if args.folder is None:
            parser.error("--clean requires a folder argument")
        clean_root = args.folder.expanduser().resolve()
        if not clean_root.is_dir():
            parser.error(f"{clean_root} is not a directory")
        return _clean_corrected_outputs(clean_root, args.recursive, dry_run=args.dry_run)

    if args.folder is None or args.instrument is None:
        parser.error("folder and --instrument are required (or use --list-instruments / --clean)")

    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    try:
        mesh = reflectron_core.load_builtin_preset(args.instrument)
    except ValueError as exc:
        parser.error(str(exc))

    epos_files = _find_epos_files(root, args.recursive)
    if not epos_files:
        print(f"No .epos files found under {root}")
        return 0

    print(f"Instrument preset: {mesh.preset.display_name}")
    print(f"Found {len(epos_files)} EPOS file(s) under {root}")
    free = _free_bytes(root)
    if free >= 0:
        print(f"Free space on output drive: {_format_size(free)}")

    failed_files: list[tuple[Path, str]] = []
    n_ok = n_skipped = n_range_ok = 0
    for index, epos_path in enumerate(epos_files, start=1):
        print(f"[{index}/{len(epos_files)}] {epos_path}")
        try:
            result = _correct_single_file(
                epos_path,
                mesh,
                save_epos=args.save_epos,
                overwrite=args.overwrite,
                with_matlab_range=args.with_matlab_range,
                matlab_range_pattern=args.matlab_range_pattern,
                matlab_range_depth=args.matlab_range_depth,
            )
        except Exception as exc:
            failed_files.append((epos_path, f"{type(exc).__name__}: {exc}"))
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
            continue

        if result["status"] == "skipped":
            n_skipped += 1
            print(f"    skipped (output exists): {result['h5']}")
        else:
            n_ok += 1
            print(f"    wrote: {result['h5']}")
            if "epos" in result:
                print(f"    wrote: {result['epos']}")
        range_info = result.get("matlab_range")
        if range_info is not None:
            if range_info.get("status") == "ok":
                n_range_ok += 1
                print(
                    "    matlab range: "
                    f"{range_info['fig']} -> {range_info['range_h5']} "
                    f"({range_info['ranges']} ranges)"
                )
            else:
                print(
                    "    matlab range: skipped (output exists) "
                    f"{range_info['range_h5']}"
                )
        elif args.with_matlab_range:
            print("    matlab range: no .fig found within search depth")

    print(
        f"\nDone. corrected={n_ok}, skipped={n_skipped}, "
        f"failed={len(failed_files)}, matlab_range_written={n_range_ok}"
    )
    if failed_files:
        print("Failed files:")
        for path, reason in failed_files:
            print(f"  {path}  --  {reason}")
        print(
            "Re-run with the same command to retry only the failed files "
            "(already-written .h5 outputs are skipped automatically)."
        )
    return 0 if not failed_files else 1


if __name__ == "__main__":
    sys.exit(main())
