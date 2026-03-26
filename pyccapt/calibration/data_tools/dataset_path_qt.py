from PyQt6.QtWidgets import QApplication, QFileDialog


DATASET_FILTER = (
    "Dataset files (*.epos *.pos *.apt *.h5 *.ato *.csv);;"
    "LEAP (*.epos *.pos *.apt);;"
    "PyCCAPT / HDF5 (*.h5);;"
    "ATO (*.ato);;"
    "CSV (*.csv);;"
    "All Files (*)"
)
RANGE_FILTER = (
    "Range files (*.h5 *.rrng *.rng);;"
    "PyCCAPT / HDF5 (*.h5);;"
    "LEAP RRNG (*.rrng);;"
    "LEAP RNG (*.rng);;"
    "All Files (*)"
)
GENERIC_FILTER = (
    "Dataset / range files (*.epos *.pos *.apt *.h5 *.ato *.csv *.rrng *.rng);;"
    "All Files (*)"
)


def gui_fname(initial_directory, file_kind="any"):
    """
    Select a file via a dialog and return the file name.

    Args:
        initial_directory (str): path to the initial directory.

    Returns:
        chosen_file (str): path to the chosen file.
    """

    app = QApplication.instance() or QApplication([initial_directory])
    selected_filter = {
        "dataset": DATASET_FILTER,
        "range": RANGE_FILTER,
    }.get(str(file_kind).lower(), GENERIC_FILTER)
    fname = QFileDialog.getOpenFileName(
        None,
        "Select a file...",
        initial_directory,
        filter=selected_filter,
    )
    chosen_file = fname[0]

    if chosen_file:
        return chosen_file
    else:
        return None
