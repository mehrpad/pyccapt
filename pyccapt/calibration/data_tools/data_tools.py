import h5py
import numpy as np
import pandas as pd
import scipy.io

# Local module and scripts
from pyccapt.calibration.data_tools import ato_tools, data_loadcrop, data_tools
from pyccapt.calibration.leap_tools import ccapt_tools
from pyccapt.calibration.mc import tof_tools
from pyccapt.calibration.leap_tools import leap_tools


def read_hdf5(filename: "type: string - Path to hdf5(.h5) file") -> "type: dataframe":
    """
    This function differs from reading pandas dataframe as it does not assume that
    the contents of the HDF5 file as argument was created using pandas. It could have been
    created using other tools like h5py/MATLAB.
    """

    try:
        dataframeStorage = {}
        groupDict = {}

        with h5py.File(filename, 'r') as hdf:
            groups = list(hdf.keys())

            for item in groups:
                groupDict[item] = list(hdf[item].keys())
            print(groupDict)
            for key, value in groupDict.items():
                for item in value:
                    dataset = pd.DataFrame(np.array(hdf['{}/{}'.format(key, item)]), columns=['values'])
                    dataframeStorage["{}/{}".format(key, item)] = dataset

            return dataframeStorage
    except FileNotFoundError as error:
        print("[*] HDF5 File could not be found")

    except IndexError as error:
        print("[*] No Group keys could be found in HDF5 File")


def read_range(filename: "type:string - Path to hdf5(.h5) file") -> "type: dataframe - Pandas Dataframe":
    """
    This function is different from read_hdf5 function. As it assumes, the content 
    of the HDF5 file passed as argument was created using the Pandas library.

        Attributes:
            filename: Path to the hdf5 file. (type: string)
        Return:
            hdf5_file_response:  content of hdf5 file (type: dataframe)       
    """
    try:
        # check the file extension
        if filename.endswith('.h5'):
            range_data = pd.read_hdf(filename, mode='r')
        elif filename.endswith('.rrng'):
            range_data = leap_tools.read_rrng(filename)
        return range_data
    except FileNotFoundError as error:
        print("[*] HDF5 File could not be found")
        print(error)
        raise FileNotFoundError


def read_mat_files(filename: "type:string - Path to .mat file") -> " type: dict - Returns the content .mat file":
    """
        This function read data from .mat files.
        Attributes:
            filename: Path to the .mat file. (type: string)
        Return:
            hdf5_file_response:  content of hdf5 file (type: dict)  
    """
    try:
        hdf5_file_response = scipy.io.loadmat(filename)
        return hdf5_file_response
    except FileNotFoundError as error:
        print("[*] Mat File could not be found")


def convert_mat_to_df(hdf5_file_response: "type: dict - content of .mat file"):
    """
        This function converts converts contents read from the .mat file
        to pandas dataframe.
        Attributes:
            hdf5_file_response: contents of .mat file (type: dict)
        Returns:
            pd_dataframe: converted dataframes (type: pandas dataframe)
    """
    pd_dataframe = pd.DataFrame(hdf5_file_response['None'])
    key = 'dataframe/isotope'
    filename = 'isotopeTable.h5'
    store_df_to_hdf(filename, pd_dataframe, key)
    return pd_dataframe


def store_df_to_hdf(dataframe: "dataframe which is to be stored in h5 file",
                    key: "DirectoryStructure/columnName of content",
                    filename: "type: string - name of hdf5 file"):
    """
        This function stores dataframe to hdf5 file.

        Attributes:
            filename: filename of hdf5 where dataframes needs to stored
            dataframe: dataframe that needs to be stored.
            key: Key that defines hierarchy of the hdf5
        Returns:
            Does not return anything
    """
    dataframe.to_hdf(filename, key, mode='w')


def store_df_to_csv(data, path):
    """
        This function stores dataframe to csv file.

        Attributes:
            path: filename of hdf5 where dataframes needs to stored
            data: data that needs to be stored.
        Returns:
            Does not return anything
    """

    data.to_csv(path, encoding='utf-8', index=False, sep=';')


def remove_invalid_data(dld_group_storage, max_tof):
    """
    Removes the data with time-of-flight (TOF) values greater than max_tof or lower than 0.

    Args:
        dld_group_storage (pandas.DataFrame): DataFrame containing the DLD group storage data.
        max_tof (float): Maximum allowable TOF value.

    Returns:
        None. The DataFrame is modified in-place.

    """
    # Create a mask for data with TOF values greater than max_tof
    mask_1 = dld_group_storage['t (ns)'].to_numpy() > max_tof

    mask_2 = (dld_group_storage['t (ns)'].to_numpy() < 50)  # Remove data with TOF values less than 50 ns

    mask_3 = ((dld_group_storage['x_det (cm)'].to_numpy() == 0) & (dld_group_storage['y_det (cm)'].to_numpy() == 0) &
              (dld_group_storage['t (ns)'].to_numpy() == 0))

    mask_4 = (dld_group_storage['high_voltage (V)'].to_numpy() < 0)

    mask_5 = (dld_group_storage['x_det (cm)'].to_numpy() == 0) & (dld_group_storage['y_det (cm)'].to_numpy() == 0)

    mask_f_1 = np.logical_or(mask_1, mask_2)
    mask_f_2 = np.logical_or(mask_3, mask_4)
    mask_f_2 = np.logical_or(mask_f_2, mask_5)
    mask = np.logical_or(mask_f_1, mask_f_2)

    # Calculate the number of data points over max_tof
    num_over_max_tof = len(mask[mask])

    # Remove data points with TOF values greater than max_tof
    dld_group_storage.drop(np.where(mask)[0], inplace=True)

    # Reset the index of the DataFrame
    dld_group_storage.reset_index(inplace=True, drop=True)

    # Print the number of data points over max_tof
    print('The number of data that is removed:', num_over_max_tof)

    return dld_group_storage


def save_data(data, variables, name=None, hdf=True, epos=False, pos=False, csv=False, temp=False):
    """
    save data in different formats

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        vsriables (class): class containing the variables.
        name (string): name of the dataset.
        hdf (bool): save data as hdf5 file.
        epos (bool): save data as epos file.
        pos (bool): save data as pos file.
        csv (bool): save data as csv file.
        temp (bool): save data as temporary file.

    Returns:
        None. The DataFrame is modified in-place.

    """
    if name is not None:
        data_name = name
    else:
        if temp:
            data_name = variables.result_data_name + '_temp'
        else:
            data_name = variables.data_name

    if hdf:
        # save the dataset to hdf5 file
        hierarchyName = 'df'
        store_df_to_hdf(data, hierarchyName, variables.result_data_path + '//' + data_name + '.h5')
    if epos:
        # save data as epos file
        ccapt_tools.ccapt_to_epos(data, path=variables.result_path,
                                  name=data_name + '.epos')
    if pos:
        # save data in pos format
        ccapt_tools.ccapt_to_pos(data, path=variables.result_path, name=data_name + '.pos')
    if csv:
        # save data in csv format
        store_df_to_csv(data, variables.result_path + variables.result_data_name + '.csv')


def load_data(dataset_path, data_type, mode='processed'):
    """
    save data in different formats

    Args:
        dataset_path (string): path to the dataset.
        data_type (string): type of the dataset.
        mode (string): mode of the dataset.

    Returns:
        data (pandas.DataFrame): DataFrame containing the data.

    """
    if data_type == 'leap_pos' or data_type == 'leap_epos':
        if data_type == 'leap_epos':
            data = ccapt_tools.epos_to_ccapt(dataset_path)
        else:
            print('The dataset should contains at least epos information to use all possible analysis')
            data = ccapt_tools.pos_to_ccapt(dataset_path)
    elif data_type == 'leap_apt':
        data = ccapt_tools.apt_to_ccapt(dataset_path)
    elif data_type == 'ato_v6':
        data = ato_tools.ato_to_ccapt(dataset_path, moed='pyccapt')
    elif data_type == 'pyccapt' and mode == 'raw':
        data = data_loadcrop.fetch_dataset_from_dld_grp(dataset_path)
    elif data_type == 'pyccapt' and mode == 'processed':
        data = pd.read_hdf(dataset_path, mode='r')
        if 'pulse_v (V)' not in data.columns:
            # create a new colum at the index of 'pulse' and remove the old one
            data.insert(data.columns.get_loc('pulse') + 1, 'pulse_v (V)', data['pulse'])
            data.drop('pulse', axis=1, inplace=True)
        if 'pulse_l (pJ)' not in data.columns:
            data.insert(data.columns.get_loc('pulse_v (V)') + 1, 'pulse_l (pJ)', np.zeros_like(data['high_voltage (V)']))
    return data


def extract_data(data, variables, flightPathLength_d, max_mc):
    """
    exctract data from the dataset

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        variables (class): class containing the variables.
        flightPathLength_d (float): flight path length in m.
        t0_d (float): time of flight offset in ns.
        max_mc (float): maximum time of flight in ns.
    Returns:

    """

    variables.dld_high_voltage = data['high_voltage (V)'].to_numpy()
    if 'pulse_v (V)' not in data.columns:
        variables.dld_pulse_v = data['pulse'].to_numpy()
    else:
        variables.dld_pulse_v = data['pulse_v (V)'].to_numpy()
    if 'pulse_l (pJ)' not in data.columns:
        variables.dld_pulse_l = np.zeros_like(variables.dld_high_voltage)
    else:
        variables.dld_pulse_l = data['pulse_l (pJ)'].to_numpy()
    variables.dld_t = data['t (ns)'].to_numpy()
    variables.dld_x_det = data['x_det (cm)'].to_numpy()
    variables.dld_y_det = data['y_det (cm)'].to_numpy()
    if 'mc (Da)' in data.columns:
        variables.mc = data['mc (Da)'].to_numpy()
    if 't_c (ns)' in data.columns:
        variables.dld_t_c = data['t_c (ns)'].to_numpy()
    if 'mc_uc (Da)' in data.columns:
        variables.mc_calib = data['mc_uc (Da)'].to_numpy()
        variables.mc_calib_backup = data['mc_uc (Da)'].to_numpy()
        variables.mc_uc = data['mc_uc (Da)'].to_numpy()

    # Calculate the maximum possible time of flight (TOF)
    variables.max_tof = int(tof_tools.mc2tof(max_mc, 1000, 0, 0, flightPathLength_d))
    variables.dld_t_calib = data['t (ns)'].to_numpy()
    variables.dld_t_calib_backup = data['t (ns)'].to_numpy()

    if 'x (nm)' in data.columns and 'y (nm)' in data.columns and 'z (nm)' in data.columns:
        variables.x = data['x (nm)'].to_numpy()
        variables.y = data['y (nm)'].to_numpy()
        variables.z = data['z (nm)'].to_numpy()
    print('The maximum possible time of flight is:', variables.max_tof)


def pyccapt_raw_to_processed(data):
    """
    process the raw data

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
    Returns:
        data (pandas.DataFrame): DataFrame containing the processed data.

    """
    # create pandas an empty dataframe
    data_processed = pd.DataFrame()
    data_processed['x (nm)'] = np.zeros(len(data))
    data_processed['y (nm)'] = np.zeros(len(data))
    data_processed['z (nm)'] = np.zeros(len(data))
    data_processed['mc (Da)'] = np.zeros(len(data))
    data_processed['mc_uc (Da)'] = np.zeros(len(data))
    data_processed['high_voltage (V)'] = data['high_voltage (V)'].to_numpy()
    data_processed['pulse_v (V)'] = data['pulse_v (V)'].to_numpy()
    data_processed['pulse_l (pJ)'] = data['pulse_l (pJ)'].to_numpy()
    data_processed['t (ns)'] = data['t (ns)'].to_numpy()
    data_processed['t_c (ns)'] = np.zeros(len(data))
    data_processed['x_det (cm)'] = data['x_det (cm)'].to_numpy()
    data_processed['y_det (cm)'] = data['y_det (cm)'].to_numpy()
    data_processed['delta_p'] = np.zeros(len(data))
    data_processed['multi'] = np.zeros(len(data))
    data_processed['start_counter'] = data['start_counter'].to_numpy()

    return data_processed

def save_range(variables):
    """
    Save the range data to the file.

    Args:
        variables:

    Returns:
        None
    """
    # save the new data
    name_save_file = variables.result_data_path + '/' + variables.dataset_name + '_range' + '.h5'
    data_tools.store_df_to_hdf(variables.range_data, 'df', name_save_file)
    # save data in csv format
    name_save_file = variables.result_data_path + '/' + variables.dataset_name + '_range' + '.csv'
    data_tools.store_df_to_csv(variables.range_data, name_save_file)
