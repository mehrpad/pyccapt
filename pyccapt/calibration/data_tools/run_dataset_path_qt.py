import sys

from pyccapt.calibration.data_tools.dataset_path_qt import gui_fname

if __name__ == "__main__":
    folder_path = sys.argv[1]
    file_kind = sys.argv[2] if len(sys.argv) > 2 else "any"
    result = gui_fname(folder_path, file_kind=file_kind)
    if result:
        print(result)
    else:
        print("No file chosen")
