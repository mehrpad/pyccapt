import os

import h5py
import numpy as np


def rename_subcategory(hdf5_file_path, old_name, new_name):
    """
        rename subcategory

        Args:
            hdf5_file_path: path to the hdf5 file
            old_name: old name of the subcategory
            new_name: new name of the subcategory

        Returns:
            None
    """

    with h5py.File(hdf5_file_path, 'r+') as file:
        # data_x = file['dld/x']
        # del file[new_name]
        # file.create_dataset(new_name, data=np.zeros(len(data_x)), dtype=np.int64)
        if old_name in file:
            file[new_name] = file[old_name]
            del file[old_name]
            print(f"Subcategory '{old_name}' renamed to '{new_name}'")
        else:
            print(f"Subcategory '{old_name}' not found in the HDF5 file.")


def correct_surface_concept_old_data(hdf5_file_path):
    """
        correct surface concept old data

        Args:
            hdf5_file_path: path to the hdf5 file

        Returns:
            None
    """
    # surface concept tdc specific binning and factors
    TOFFACTOR = 27.432 / (1000.0 * 4.0)  # 27.432 ps/bin, tof in ns, data is TDC time sum
    DETBINS = 4900.0
    BINNINGFAC = 2.0
    XYFACTOR = 80.0 / DETBINS * BINNINGFAC  # XXX mm/bin
    XYBINSHIFT = DETBINS / BINNINGFAC / 2.0  # to center detector

    with h5py.File(hdf5_file_path, 'r+') as file:
        data_x = file['dld/x']
        data_y = file['dld/y']
        data_t = file['dld/t']

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_t = np.array(data_t)

        modified_t = (data_t.astype(np.float64) * TOFFACTOR)
        del file['dld/t']
        file.create_dataset('dld/t', data=modified_t, dtype=np.float64)
        modified_x = ((data_x.astype(np.float64) - XYBINSHIFT) * XYFACTOR) / 10.0
        del file['dld/x']
        file.create_dataset('dld/x', data=modified_x, dtype=np.float64)
        modified_y = ((data_y.astype(np.float64) - XYBINSHIFT) * XYFACTOR) / 10.0
        del file['dld/y']
        file.create_dataset('dld/y', data=modified_y, dtype=np.float64)


def copy_npy_to_hdf_surface_concept(path, hdf5_file_name):
    """
        copy npy data to hdf5 file for surface concept TDC

        Args:
            path: path to the npy files
            hdf5_file_name: name of the hdf5 file

        Returns:
            None
    """
    # TOFFACTOR = 27.432 / (1000 * 4)  # 27.432 ps/bin, tof in ns, data is TDC time sum
    # DETBINS = 4900
    # BINNINGFAC = 2
    # XYFACTOR = 80 / DETBINS * BINNINGFAC  # XXX mm/bin
    # XYBINSHIFT = DETBINS / BINNINGFAC / 2  # to center detector

    hdf5_file_path = path + hdf5_file_name
    high_voltage = np.load(path + 'voltage_data.npy')
    voltage_pulse = np.load(path + 'voltage_pulse_data.npy')
    laser_pulse = np.load(path + 'laser_pulse_data.npy')
    start_counter = np.load(path + 'start_counter.npy')
    t = np.load(path + 't_data.npy')
    x_det = np.load(path + 'x_data.npy')
    y_det = np.load(path + 'y_data.npy')

    channel = np.load(path + 'channel_data.npy')
    high_voltage_tdc = np.load(path + 'voltage_data_tdc.npy')
    voltage_pulse_tdc = np.load(path + 'voltage_pulse_data_tdc.npy')
    laser_pulse_tdc = np.load(path + 'laser_pulse_data_tdc.npy')
    start_counter_tdc = np.load(path + 'tdc_start_counter.npy')
    time_data = np.load(path + 'time_data.npy')

    # xx_tmp = (((x_det - XYBINSHIFT) * XYFACTOR) * 0.1)  # from mm to in cm by dividing by 10
    # yy_tmp = (((y_det - XYBINSHIFT) * XYFACTOR) * 0.1)  # from mm to in cm by dividing by 10
    # tt_tmp = (t * TOFFACTOR)  # in ns

    with h5py.File(hdf5_file_path, 'r+') as file:
        del file['dld/t']
        del file['dld/x']
        del file['dld/y']
        del file['dld/voltage_pulse']
        del file['dld/laser_pulse']
        del file['dld/high_voltage']
        del file['dld/start_counter']
        file.create_dataset('dld/t', data=t, dtype=np.float64)
        file.create_dataset('dld/x', data=x_det, dtype=np.float64)
        file.create_dataset('dld/y', data=y_det, dtype=np.float64)
        file.create_dataset('dld/voltage_pulse', data=voltage_pulse, dtype=np.float64)
        file.create_dataset('dld/laser_pulse', data=laser_pulse, dtype=np.float64)
        file.create_dataset('dld/high_voltage', data=high_voltage, dtype=np.float64)
        file.create_dataset('dld/start_counter', data=start_counter, dtype=np.uint64)

        del file['tdc/channel']
        del file['tdc/high_voltage']
        del file['tdc/voltage_pulse']
        del file['tdc/laser_pulse']
        del file['tdc/start_counter']
        del file['tdc/time_data']
        file.create_dataset('tdc/channel', data=channel, dtype=np.uint32)
        file.create_dataset('tdc/high_voltage', data=high_voltage_tdc, dtype=np.float64)
        file.create_dataset('tdc/voltage_pulse', data=voltage_pulse_tdc, dtype=np.float64)
        file.create_dataset('tdc/laser_pulse', data=laser_pulse_tdc, dtype=np.float64)
        file.create_dataset('tdc/start_counter', data=start_counter_tdc, dtype=np.uint64)
        file.create_dataset('tdc/time_data', data=time_data, dtype=np.uint64)


def load_and_copy_chunks_to_hdf(path, hdf5_file_path, chunk_id):
    with h5py.File(hdf5_file_path, 'r+') as hdf_file:
        # Delete existing datasets (if needed)
        for group in ['dld', 'tdc']:
            if group in hdf_file:
                for name in list(hdf_file[group].keys()):
                    del hdf_file[f'{group}/{name}']

        # Create empty datasets with appropriate shapes and dtypes
        def create_empty_dataset(group_name, chunk_name, dataset_name, dtype):
            total_size = 0
            for i in range(1, chunk_id + 1):
                chunk_file = path + f"/{chunk_name}_chunk_{i}.npy"
                if os.path.exists(chunk_file):
                    chunk_data = np.load(chunk_file)
                    total_size += chunk_data.shape[0]
            if total_size > 0:
                hdf_file.create_dataset(f'{group_name}/{dataset_name}', (total_size,), dtype=dtype)

        create_empty_dataset('dld', 't_data', 't', np.float64)
        create_empty_dataset('dld', 'x_data', 'x', np.float64)
        create_empty_dataset('dld', 'y_data', 'y', np.float64)
        create_empty_dataset('dld', 'voltage_pulse_data', 'voltage_pulse', np.float64)
        create_empty_dataset('dld', 'laser_pulse_data', 'laser_pulse', np.float64)
        create_empty_dataset('dld', 'voltage_data', 'high_voltage', np.float64)
        create_empty_dataset('dld', 'start_counter', 'start_counter', np.uint64)

        create_empty_dataset('tdc', 'channel_data', 'channel', np.uint32)
        create_empty_dataset('tdc', 'voltage_data_tdc', 'high_voltage', np.float64)
        create_empty_dataset('tdc', 'voltage_pulse_data_tdc', 'voltage_pulse', np.float64)
        create_empty_dataset('tdc', 'laser_pulse_data_tdc', 'laser_pulse', np.float64)
        create_empty_dataset('tdc', 'tdc_start_counter', 'start_counter', np.uint64)
        create_empty_dataset('tdc', 'time_data', 'time_data', np.uint64)

        # Write data chunk by chunk
        def write_chunked_data(group_name, dataset_name, dataset_name_new):
            offset = 0
            for i in range(1, chunk_id + 1):
                chunk_file = path + f"/{dataset_name_new}_chunk_{i}.npy"
                if os.path.exists(chunk_file):
                    chunk_data = np.load(chunk_file)
                    chunk_size = chunk_data.shape[0]
                    hdf_file[f'{group_name}/{dataset_name}'][offset:offset + chunk_size] = chunk_data
                    offset += chunk_size
                else:
                    print(f"File '{chunk_file}' not found.")
            print(f"Written {dataset_name} data.")

        write_chunked_data('dld', 't', 't_data')
        write_chunked_data('dld', 'x', 'x_data')
        write_chunked_data('dld', 'y', 'y_data')
        write_chunked_data('dld', 'voltage_pulse', 'voltage_pulse_data')
        write_chunked_data('dld', 'laser_pulse', 'laser_pulse_data')
        write_chunked_data('dld', 'high_voltage', 'voltage_data')
        write_chunked_data('dld', 'start_counter', 'start_counter')

        write_chunked_data('tdc', 'channel', 'channel_data')
        write_chunked_data('tdc', 'high_voltage', 'voltage_data_tdc')
        write_chunked_data('tdc', 'voltage_pulse', 'voltage_pulse_data_tdc')
        write_chunked_data('tdc', 'start_counter', 'tdc_start_counter')
        write_chunked_data('tdc', 'time_data', 'time_data')
        write_chunked_data('tdd', 'laser_pulse', 'laser_pulse_data_tdc')


def crop_dataset_to_new_file(original_path, new_path, num_of_samples):
    """
    Crop dataset and save to a new file.

    Args:
        original_path: Path to the original dataset.
        new_path: Path to save the cropped dataset.
        num_of_samples: Number of samples to keep.

    Returns:
        None
    """
    with h5py.File(original_path, 'r') as original_file, h5py.File(new_path, 'w') as new_file:
        num_events = original_file['apt/num_events']
        num_raw_signals = original_file['apt/num_raw_signals']
        assert len(num_events) == len(num_raw_signals), "Length of num_events and num_raw_signals should be the same."

        count = 0
        count_raw = 0
        index = None
        index_event = None
        index_raw = None

        for i in range(len(num_events)):
            count += num_events[i]
            count_raw += num_raw_signals[i]
            if count > num_of_samples:
                index = i
                index_event = count
                index_raw = count_raw
                break

        if index is not None:
            # Copy cropped data to the new file
            for key in original_file['apt']:
                cropped_data = original_file['apt/%s' % key][:index + 1]
                new_file.create_dataset(f'apt/{key}', data=cropped_data, dtype=original_file['apt/%s' % key].dtype)

            for key in original_file['dld']:
                cropped_data = original_file['dld/%s' % key][:index_event + 1]
                new_file.create_dataset(f'dld/{key}', data=cropped_data, dtype=original_file['dld/%s' % key].dtype)

            for key in original_file['tdc']:
                cropped_data = original_file['tdc/%s' % key][:index_raw + 1]
                new_file.create_dataset(f'tdc/{key}', data=cropped_data, dtype=original_file['tdc/%s' % key].dtype)

            print("Cropped dataset written to the new file.")
        else:
            print("Number of samples requested exceeds the dataset size. No cropping performed.")



if __name__ == '__main__':
    name = '2423_Mar-14-2025_14-52_NiC9_Al2'
    path = 'T:/Ott/APT_Measurements/Oxcart/%s/' % name
    new_path = 'C:/Users/LokalAdmin/Downloads//%s' % 'cropped_' + name
    name = '%s.h5' % name
    # copy_npy_to_hdf(path, name)

    # rename_subcategory(path + name, old_name='dld', new_name='dld_1')
    # copy_npy_to_hdf_surface_concept(path+'/temp_data/', name)
    # rename_subcategory(path + name, old_name='tdc/voltage_laser', new_name='tdc/laser_pulse')
    load_and_copy_chunks_to_hdf(path + '/temp_data/chunks/', path + name, 1220)
    # crop_dataset_to_new_file(path, new_path, 500000)
    print('Done')
