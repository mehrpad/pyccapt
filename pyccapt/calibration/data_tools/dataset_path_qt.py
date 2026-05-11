"""Qt file/directory pickers for the dataset-path Browse buttons.

PyQt6 is imported lazily inside the dialog functions so that headless CI
environments (which don't ship PyQt6 by default) can still import this module
to read the filter constants -- the dialog itself is only ever invoked from
the Jupyter notebooks, where PyQt6 is present.
"""

DATASET_FILTER = (
    "Dataset files (*.epos *.pos *.apt *.h5 *.ato *.csv *.rhit *.RHIT *.str *.STR *.hits *.HITS);;"
    "LEAP (*.epos *.pos *.apt);;"
    "LEAP CAMECA RHIT (*.rhit *.RHIT);;"
    "LEAP CAMECA STR / HITS (*.str *.STR *.hits *.HITS);;"
    "PyCCAPT / HDF5 (*.h5);;"
    "ATO (*.ato);;"
    "CSV (*.csv);;"
    "All Files (*)"
)
RANGE_FILTER = "Range files (*.h5 *.rrng *.rng);;PyCCAPT / HDF5 (*.h5);;LEAP RRNG (*.rrng);;LEAP RNG (*.rng);;All Files (*)"
CAMECA_RAW_FILTER = (
    "LEAP CAMECA raw (*.rhit *.RHIT *.str *.STR *.hits *.HITS *.epos *.EPOS);;"
    "LEAP CAMECA RHIT (*.rhit *.RHIT);;"
    "LEAP CAMECA STR / HITS (*.str *.STR *.hits *.HITS);;"
    "LEAP EPOS (*.epos *.EPOS);;"
    "All Files (*)"
)
GENERIC_FILTER = (
    "Dataset / range files (*.epos *.pos *.apt *.h5 *.ato *.csv "
    "*.rhit *.RHIT *.str *.STR *.hits *.HITS *.rrng *.rng);;"
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
    from PyQt6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance() or QApplication([initial_directory])
    selected_filter = {
        "dataset": DATASET_FILTER,
        "range": RANGE_FILTER,
        "cameca_raw": CAMECA_RAW_FILTER,
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


def gui_dirname(initial_directory):
    """Select an existing directory via a dialog and return its path."""
    from PyQt6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance() or QApplication([initial_directory])
    chosen_dir = QFileDialog.getExistingDirectory(
        None,
        "Select a directory...",
        initial_directory,
    )
    return chosen_dir or None
