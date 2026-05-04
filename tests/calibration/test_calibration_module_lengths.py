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

# Two calibration tutorial helpers are exempt from the 1250-line rule because
# each is a single ~1300-1900 line UI builder that wires up dozens of nested
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
KNOWN_OFFENDERS_TO_REFACTOR = frozenset({
    Path("tutorials/tutorials_helpers/helper_visualization.py").as_posix(),
    Path("tutorials/tutorials_helpers/helper_calibration.py").as_posix(),
})


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
    missing = [
        relative for relative in KNOWN_OFFENDERS_TO_REFACTOR
        if not (calibration_root / relative).is_file()
    ]
    assert not missing, (
        f"KNOWN_OFFENDERS_TO_REFACTOR references files that no longer exist: "
        f"{missing}. Remove these entries."
    )


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
