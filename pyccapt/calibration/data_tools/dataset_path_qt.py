from PyQt6.QtWidgets import QApplication, QFileDialog


def gui_fname(initial_directory):
    """
    Select a file via a dialog and return the file name.

    Args:
        initial_directory (str): path to the initial directory.

    Returns:
        chosen_file (str): path to the chosen file.
    """

    app = QApplication([initial_directory])
    fname = QFileDialog.getOpenFileName(None, "Select a file...", initial_directory,
                                        filter="PyCCAPT data / saved range (*.h5 *.rrng *.rng);;"
                                               "LEAP (*.pos *.epos *.apt);;"
                                               "APT (*.ato);;"
                                               "CSV (*.csv);;"
                                               "All Files (*)")
    chosen_file = fname[0]

    if chosen_file:
        return chosen_file
    else:
        return None
