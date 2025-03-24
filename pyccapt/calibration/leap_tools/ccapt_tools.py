import struct
from itertools import chain

import numpy as np
import pandas as pd

# Local module and scripts
from pyccapt.calibration.leap_tools import leap_tools


def ccapt_to_pos(data, path=None, name=None):
    """
    Convert CCAPT data to POS format.

    Args:
        data (pandas.DataFrame): CCAPT data.
        path (str): Optional. Path to save the POS file.
        name (str): Optional. Name of the POS file.

    Returns:
        bytes: POS data.

    """
    dd = data[['x (nm)', 'y (nm)', 'z (nm)', 'mc (Da)']]
    dd = dd.astype(np.single)
    records = dd.to_records(index=False)
    list_records = list(records)
    d = tuple(chain(*list_records))
    pos = struct.pack('>' + 'ffff' * len(dd), *d)
    if name is not None:
        with open(path + name, 'w+b') as f:
            f.write(pos)
    return pos


def ccapt_to_epos(data, path=None, name=None, chunk_size=1_000_000):
    """
    Convert CCAPT data to EPOS format, processing in chunks to avoid memory errors.

    Args:
        data (pandas.DataFrame): CCAPT data.
        path (str): Optional. Path to save the EPOS file.
        name (str): Optional. Name of the EPOS file.
        chunk_size (int): Number of rows to process in each chunk.

    Returns:
        None: Writes EPOS data to file if path and name are provided.
    """

    dd = data[
        ['x (nm)', 'y (nm)', 'z (nm)', 'mc (Da)', 't (ns)', 'high_voltage (V)', 'pulse_v (V)', 'x_det (cm)',
         'y_det (cm)', 'delta_p', 'multi']]

    dd = dd.astype(np.single)
    dd = dd.astype({'delta_p': np.uintc})
    dd = dd.astype({'multi': np.uintc})

    if name is not None:
        with open(path + name, 'w+b') as f:
            for i in range(0, len(dd), chunk_size):
                chunk = dd.iloc[i:i + chunk_size]
                records = chunk.to_records(index=False)
                list_records = list(records)
                d = tuple(chain(*list_records))
                epos_chunk = struct.pack('>' + 'fffffffffII' * len(chunk), *d)
                f.write(epos_chunk)
    else:
        epos = b''
        for i in range(0, len(dd), chunk_size):
            chunk = dd.iloc[i:i + chunk_size]
            records = chunk.to_records(index=False)
            list_records = list(records)
            d = tuple(chain(*list_records))
            epos_chunk = struct.pack('>' + 'fffffffffII' * len(chunk), *d)
            epos += epos_chunk
        return epos



def pos_to_ccapt(file_path):
    """
    Convert POS data to CCAPT format.

    Args:
        file_path: POS data file_path.

    Returns:
        pandas.DataFrame: CCAPT data.

    """
    pos = leap_tools.read_pos(file_path)
    length = len(pos['m/n (Da)'].to_numpy())
    ccapt = pd.DataFrame({'x (nm)': pos['x (nm)'].to_numpy(),
                          'y (nm)': pos['y (nm)'].to_numpy(),
                          'z (nm)': pos['z (nm)'].to_numpy(),
                          'mc (Da)': pos['m/n (Da)'].to_numpy(),
                          'mc_uc (Da)': np.zeros(length),
                          'high_voltage (V)': np.zeros(length),
                          'pulse': np.zeros(length),
                          't (ns)': np.zeros(length),
                          't_c (ns)': np.zeros(length),
                          'x_det (cm)': np.zeros(length),
                          'y_det (cm)': np.zeros(length),
                          'delta_p': np.zeros(length, dtype=int),
                          'multi': np.zeros(length, dtype=int),
                          'start_counter': np.zeros(length, dtype=int),
                          })
    return ccapt


def epos_to_ccapt(file_path):
    """
    Convert EPOS data to PyCCAPT format.

    Args:
        file_path: EPOS data file path.

    Returns:
        pandas.DataFrame: CCAPT data.

    """
    epos = leap_tools.read_epos(file_path)
    length = len(epos['m/n (Da)'].to_numpy())
    ccapt = pd.DataFrame({'x (nm)': epos['x (nm)'].to_numpy(),
                          'y (nm)': epos['y (nm)'].to_numpy(),
                          'z (nm)': epos['z (nm)'].to_numpy(),
                          'mc (Da)': epos['m/n (Da)'].to_numpy(),
                          'mc_uc (Da)': np.zeros(length),
                          'high_voltage (V)': epos['HV_DC (V)'].to_numpy(),
                          'pulse_v (V)': epos['pulse (V)'].to_numpy(),
                          'pulse_l (pJ)': np.zeros(length),
                          't (ns)': epos['TOF (ns)'].to_numpy(),
                          't_c (ns)': np.zeros(length),
                          'x_det (cm)': epos['det_x (mm)'].to_numpy() / 10,
                          'y_det (cm)': epos['det_y (mm)'].to_numpy() / 10,
                          'delta_p': epos['pslep'].to_numpy(),
                          'multi': epos['ipp'].to_numpy(),
                          'start_counter': np.zeros(length, dtype=int),
                          })
    return ccapt


def apt_to_ccapt(file_path):
    """
    Convert APT data to PyCCAPT format.

    Args:
        file_path: APT data file path.

    Returns:
        pandas.DataFrame: CCAPT data.
    """

    data = leap_tools.read_apt(file_path)

    data['zs'] = data['zs'] * -1 # Invert z-axis
    length_data = len(data["Mass"])

    # Define the keys and corresponding column names
    key_mappings = {
        'xs': 'x (nm)',
        'ys': 'y (nm)',
        'zs': 'z (nm)',
        'Mass': 'mc (Da)',
        'Voltage': 'high_voltage (V)',
        'Vap': 'pulse_v (V)',
        'laserpower': 'pulse_l (pJ)',
        'Epos ToF': 't (ns)',
        'tofc': 't_c (ns)',
        'XDet_mm': 'x_det (cm)',
        'YDet_mm': 'y_det (cm)',
        'Delta Pulse': 'delta_p',
        'Multiplicity': 'multi',
        'tElapsed': 'start_counter'
    }

    # Create a dictionary with zero-filled arrays for missing keys
    data_dict = {column_name: data.get(key, np.zeros(length_data)) for key, column_name in key_mappings.items()}
    data_dict['start_counter'] = np.zeros(length_data, dtype=int)

    df = pd.DataFrame(data_dict)
    df.insert(loc=4, column='mc_uc (Da)', value=np.zeros(length_data))
    # df.insert(loc=8, column='t_c (ns)', value=data['tofc '].to_numpy())
    df['x_det (cm)'] = df['x_det (cm)'] / 10
    df['y_det (cm)'] = df['y_det (cm)'] / 10

    return df
