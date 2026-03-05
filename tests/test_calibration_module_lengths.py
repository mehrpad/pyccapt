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


def test_calibration_python_module_lengths_are_bounded():
    project_root = Path(__file__).resolve().parents[1]
    calibration_root = project_root / "pyccapt" / "calibration"
    offenders = []

    for folder in CALIBRATION_FOLDERS:
        for file_path in (calibration_root / folder).rglob("*.py"):
            line_count = len(file_path.read_text(encoding="utf-8").splitlines())
            if line_count > MAX_LINES_PER_FILE:
                offenders.append(f"{file_path}: {line_count} lines")

    assert not offenders, (
        f"Python modules exceed {MAX_LINES_PER_FILE} lines:\n" + "\n".join(offenders)
    )
