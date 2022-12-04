from sys import executable, argv
from subprocess import check_output
from PyQt6.QtWidgets import QFileDialog, QApplication
import os

def gui_fname(directory='./'):
    """Open a file dialog, starting in the given directory, and return
    the chosen filename"""
    # run this exact file in a separate process, and grab the result
    file = check_output([executable, __file__, directory])
    return file.strip()


if __name__ == "__main__":
    os.chdir('..//..//..//..//')
    directory = argv[1]
    app = QApplication([directory])
    fname = QFileDialog.getOpenFileName(None, "Select a file...",
            directory, filter="All files (*)")
    print(fname[0])