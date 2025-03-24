import os

import numpy as np
import pandas as pd


def get_project_path():
    """
    Get the absolute path of the current Python script

    Args:
        None
    Returns:
        script_path (str): The absolute path of the current Python script.
    """
    script_path = os.path.abspath(__file__)

    # Traverse up the directory tree until reaching the project's main folder
    while not os.path.isfile(os.path.join(script_path, 'setup.py')):
        script_path = os.path.dirname(script_path)

        # Break the loop if we reached the root directory
        if script_path == os.path.dirname(script_path):
            break

    return script_path

class Variables:
    """
    A class that represents all shared variables.

    Attributes:
        pulse_mode (str): The pulse mode.
        selected_x_fdm (int): The value of selected_x_fdm.
        selected_y_fdm (int): The value of selected_y_fdm.
        roi_fdm (int): The value of roi_fdm.
        selected_x1 (int): The value of selected_x1.
        selected_x2 (int): The value of selected_x2.
        selected_y1 (int): The value of selected_y1.
        selected_y2 (int): The value of selected_y2.
        selected_z1 (int): The value of selected_z1.
        selected_z2 (int): The value of selected_z2.
        list_material (list): List that stores mass/weights of added elements.
        charge (list): List of charges.
        element (list): List of elements.
        isotope (list): List of isotopes.
        peaks_x_selected (list): List of peaks indices.
        peaks_index_list (list): List of peak indices.
        result_path (str): The result path.
        path (str): The path.
        dataset_name (str): The dataset name.
        result_data_path (str): The result data path.
        result_data_name (str): The result data name.
        dld_t (numpy.ndarray): Array for dld_t.
        self.dld_t_c (numpy.ndarray): Array for dld_t calibration.
        dld_x_det (numpy.ndarray): Array for dld_x_det.
        dld_y_det (numpy.ndarray): Array for dld_y_det.
        dld_high_voltage (numpy.ndarray): Array for dld_high_voltage.
        dld_t_calib (numpy.ndarray): Array for dld_t calibration.
        dld_t_calib_backup (numpy.ndarray): Backup array for dld_t calibration.
        mc (numpy.ndarray): Array for mc.
        self.mc_c (numpy.ndarray): Array for mc calibration.
        mc_calib (numpy.ndarray): Array for mc calibration.
        mc_calib_backup (numpy.ndarray): Backup array for mc calibration.
        max_peak (int): The maximum peak value.
        max_tof (int): The maximum tof value.
        peak_x (list): List of peaks.
        peak_y (list): List of peak y-values.
        peak_widths (list): List of peak widths.
        x_hist (numpy.ndarray): Array for x histogram.
        y_hist (numpy.ndarray): Array for y histogram.
        h_line_pos (list): List of horizontal line positions.
        plotly_3d_reconstruction (plotly.graph_objs._figure.Figure): Plotly 3D reconstruction.
        data (data frame): dataset of the experiment.
        data_backup (data frame): backup dataset of the experiment.
        range_data (data frame): range dataset.
        range_data_backup (data frame): Backup range dataset.
        last_directory (str): The last directory.
        animation_detector_html (str): The animation detector html.
    """

    def __init__(self):
        """
        Initializes all the attributes of MyClass.
        """
        self.pulse_mode = ''

        self.selected_x_fdm = 0
        self.selected_y_fdm = 0
        self.roi_fdm = 0

        self.selected_x1 = 0
        self.selected_x2 = 0
        self.selected_y1 = 0
        self.selected_y2 = 0
        self.selected_z1 = 0
        self.selected_z2 = 0
        self.h_line_pos = []

        self.list_material = []
        self.charge = []
        self.element = []
        self.isotope = []

        self.peaks_x_selected = []
        self.peaks_index_list = []

        self.result_path = ''
        self.path = ''
        self.dataset_name = ''
        self.result_data_path = ''
        self.result_data_name = ''

        self.dld_t = np.zeros(0)
        self.dld_t_c = np.zeros(0)
        self.dld_x_det = np.zeros(0)
        self.dld_y_det = np.zeros(0)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.z = np.zeros(0)
        self.dld_high_voltage = np.zeros(0)
        self.dld_pulse_v = np.zeros(0)
        self.dld_pulse_l = np.zeros(0)
        self.dld_t_calib = np.zeros(0)
        self.dld_t_calib_backup = np.zeros(0)
        self.mc = np.zeros(0)
        self.mc_uc = np.zeros(0)
        self.mc_calib = np.zeros(0)
        self.mc_calib_backup = np.zeros(0)
        self.max_peak = 0
        self.max_tof = 0
        self.peaks_index = 0
        self.peak_x = []
        self.peak_y = []
        self.peak_widths = []
        self.x_hist = None
        self.y_hist = None
        self.AptHistPlotter = None
        self.ions_list_data = None
        self.last_directory = get_project_path()  # You can set a default directory here

        self.plotly_3d_reconstruction = None
        self.data = None
        self.data_backup = None
        self.max_mc = 400
        self.max_tof = None
        self.flight_path_length = None
        self.pulse_mode = None
        self.mask = None
        # Create an empty DataFrame with the specified columns
        self.range_data = pd.DataFrame({"name": ["unranged0"], "ion": ['un'], "mass": [0], "mc": [0], "mc_low": 0,
                                "mc_up": 400, "color": '#000000', "element": [['unranged']],
                                "complex": [[0]], "isotope": [[0]], "charge": [0]})

        # Set data types for the columns
        self.range_data = self.range_data.astype({'name': 'str',
                                                  'ion': 'str',
                                                  'mass': 'float64',
                                                  'mc': 'float64',
                                                  'mc_low': 'float64',
                                                  'mc_up': 'float64',
                                                  'color': 'str',
                                                  'element': 'object',
                                                  'complex': 'object',
                                                  'isotope': 'object',
                                                  'charge': 'uint32'})

        self.range_data_backup = None
        self.animation_detector_html = None
