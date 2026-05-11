from pathlib import Path

MAX_LINES_PER_FILE = 1250
CALIBRATION_FOLDERS = (
    "calibration",
    "clustering",
    "data_tools",
    "leap_tools",
    "mc",
    "reconstructions",
    "tutorials",
)

# Calibration tutorial helpers are exempt from the 1250-line rule because each
# is a single ~1300-1900 line UI builder that wires up dozens of nested
# closures sharing widget state. Splitting these without behavior risk would
# require turning every closure into a function with the captured state passed
# explicitly — a large, error-prone refactor with no observable benefit.
#
# The pure / closure-free utilities have already been extracted to sibling
# modules (`_helper_calibration_pure.py`); what remains here is the closure
# graph itself. Revisit only when one of these files needs a substantive
# functional change anyway, so the split can ride along with that work.
#
# The rule still bites every other calibration module, including new ones.
#
# TODO(helper_auto_raw_analysis): currently 1389 lines after the peak-units
# dropdown + signal preview + chunked DLTS classifier landed. The pure pieces
# (detect_detector_kind, _delay_line_pairs, _classify_pulse_chunks, plot_*,
# species_from_*, markdown formatters) are easy to extract into a sibling
# `_auto_raw_analysis_pure.py`; the UI closures (call_auto_raw_data_analysis,
# call_signal_preview, run_analysis) stay here. Schedule that split next time
# this file gets a substantive change.
KNOWN_OFFENDERS_TO_REFACTOR = frozenset(
    {
        Path("tutorials/tutorials_helpers/helper_visualization.py").as_posix(),
        Path("tutorials/tutorials_helpers/helper_calibration.py").as_posix(),
        Path("tutorials/tutorials_helpers/helper_auto_raw_analysis.py").as_posix(),
        # TODO(_raw_workflow_surface_concept): currently 1565 lines after the
        # combinatorial per-pulse hit recovery (greedy + exhaustive) + per-peak
        # diagnostics + length-tracking landed. The pure pieces (candidate
        # generation, validity scoring, selection algorithms) are easy to
        # extract into a sibling `_raw_workflow_sc_combinatorial.py`; the
        # legacy chunked recovery + plotting helpers stay here. Schedule that
        # split next time this file gets a substantive change.
        Path("data_tools/_raw_workflow_surface_concept.py").as_posix(),
    }
)


def test_calibration_python_module_lengths_are_bounded():
    project_root = Path(__file__).resolve().parents[2]
    calibration_root = project_root / "pyccapt" / "calibration"
    offenders = []

    for folder in CALIBRATION_FOLDERS:
        for file_path in (calibration_root / folder).rglob("*.py"):
            relative = file_path.relative_to(calibration_root).as_posix()
            line_count = len(file_path.read_text(encoding="utf-8").splitlines())
            if line_count > MAX_LINES_PER_FILE and relative not in KNOWN_OFFENDERS_TO_REFACTOR:
                offenders.append(f"{relative}: {line_count} lines")

    assert not offenders, (
        f"Python modules exceed {MAX_LINES_PER_FILE} lines:\n"
        + "\n".join(offenders)
        + "\n\nIf this file should temporarily be exempt, add it to "
        "KNOWN_OFFENDERS_TO_REFACTOR with a TODO; otherwise split it."
    )


def test_known_offender_list_does_not_outlive_the_files():
    """Catch entries in the offender list that no longer exist on disk."""
    project_root = Path(__file__).resolve().parents[2]
    calibration_root = project_root / "pyccapt" / "calibration"
    missing = [relative for relative in KNOWN_OFFENDERS_TO_REFACTOR if not (calibration_root / relative).is_file()]
    assert not missing, f"KNOWN_OFFENDERS_TO_REFACTOR references files that no longer exist: {missing}. Remove these entries."


def test_known_offender_list_only_holds_actual_offenders():
    """Catch entries that have already been refactored under the limit."""
    project_root = Path(__file__).resolve().parents[2]
    calibration_root = project_root / "pyccapt" / "calibration"
    no_longer_offending = []
    for relative in KNOWN_OFFENDERS_TO_REFACTOR:
        path = calibration_root / relative
        if path.is_file():
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count <= MAX_LINES_PER_FILE:
                no_longer_offending.append(f"{relative}: {line_count} lines")
    assert not no_longer_offending, (
        "These files are now under the limit; remove them from "
        f"KNOWN_OFFENDERS_TO_REFACTOR:\n" + "\n".join(no_longer_offending)
    )
