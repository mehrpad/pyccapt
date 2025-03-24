import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.interpolate import interp1d


def copy_xytof_from_cobold_txt_to_hdf5(txt_path, save_path):
    """
    Copy x, y, tof, multi event data from Cobold text file to an existing HDF5 file.

    Args:
        txt_path (str): Path to the Cobold text file.
        save_path (str): Path to the save file.

    Returns:
        None
    """

    def append_number_at_index(array, index, number):
        return np.concatenate((array[:index], [number], array[index:]))
    # Read data from text file
    with open(txt_path, 'r') as f:
        data = np.loadtxt(f)

    xx = data[:, 6] / 10
    yy = data[:, 7] / 10
    tof = data[:, 8]

    xx_2 = data[:, 10] / 10
    yy_2 = data[:, 11] / 10
    tof_2 = data[:, 12]

    xx_3 = data[:, 14] / 10
    yy_3 = data[:, 15] / 10
    tof_3 = data[:, 16]

    xx_4 = data[:, 18] / 10
    yy_4 = data[:, 19] / 10
    tof_4 = data[:, 20]
    with h5py.File(save_path, 'r+') as file:
        del file['dld/x']
        del file['dld/y']
        del file['dld/t']
        # del file['dld/start_counter']
        del file['dld/AbsoluteTimeStamp']
        laser_intensity =  file['dld/laser_intensity'][:]
        del file['dld/laser_intensity']

        file.create_dataset('dld/x', data=xx)
        file.create_dataset('dld/y', data=yy)
        file.create_dataset('dld/t', data=tof)
        file.create_dataset('dld/start_counter', data=np.arange(len(xx)), dtype='i')
        file.create_dataset('dld/pulse', data=laser_intensity, dtype='i')

        x = file['dld/x'][:]
        y = file['dld/y'][:]
        t = file['dld/t'][:]
        dc_v = file['dld/high_voltage'][:]
        # pulse = file['dld/pulse'][:]
        pulse = file['dld/pulse'][:]
        sc = file['dld/start_counter'][:]
    print('multi event length:', len(tof_2[tof_2 != 0]) + len(tof_3[tof_3 != 0]) + len(tof_4[tof_4 != 0]))
    index = 0
    for i in range(len(tof_2)):
        if tof_2[i] != 0:
            x = append_number_at_index(x, index + 1, xx_2[i])
            y = append_number_at_index(y, index + 1, yy_2[i])
            t = append_number_at_index(t, index + 1, tof_2[i])
            dc_v = append_number_at_index(dc_v, index + 1, dc_v[index])
            pulse = append_number_at_index(pulse, index + 1, pulse[index])
            sc = append_number_at_index(sc, index + 1, sc[index])
            index = index + 1
        if tof_3[i] != 0:
            x = append_number_at_index(x, index + 1, xx_3[i])
            y = append_number_at_index(y, index + 1, yy_3[i])
            t = append_number_at_index(t, index + 1, tof_3[i])
            dc_v = append_number_at_index(dc_v, index + 1, dc_v[index])
            pulse = append_number_at_index(pulse, index + 1, pulse[index])
            sc = append_number_at_index(sc, index + 1, sc[index])
            index = index + 1
        if tof_4[i] != 0:
            x = append_number_at_index(x, index + 1, xx_4[i])
            y = append_number_at_index(y, index + 1, yy_4[i])
            t = append_number_at_index(t, index + 1, tof_4[i])
            dc_v = append_number_at_index(dc_v, index + 1, dc_v[index])
            pulse = append_number_at_index(pulse, index + 1, pulse[index])
            sc = append_number_at_index(sc, index + 1, sc[index])
        index = index + 1
        if ((i + 1) / len(tof_2)) * 100 % 10 == 0:
            print("Processed {} percent".format(int(((i + 1) / len(tof_2)) * 100)))

    with h5py.File(save_path, 'r+') as file:
        del file['dld/x']
        del file['dld/y']
        del file['dld/t']
        del file['dld/high_voltage']
        del file['dld/pulse']
        del file['dld/start_counter']

        file.create_dataset('dld/x', data=x)
        file.create_dataset('dld/y', data=y)
        file.create_dataset('dld/t', data=t)
        file.create_dataset('dld/high_voltage', data=dc_v)
        file.create_dataset('dld/pulse', data=pulse)
        file.create_dataset('dld/start_counter', data=sc)

    print('finish')



def rename_a_category(file_path, old_name, new_name):
    with h5py.File(file_path, 'r+') as data:
        temp = data[old_name][:]
        del data[old_name]
        data.create_dataset(new_name, data=temp)


def laser_pulse_energy_from_mat_file(mat_path, source_file, target_file):
    # pulse energy = ref_laser_intensity * pulse duration * area
    # pulse_ref_energy = ref_laser_intensity * 1e2 * 1e2 * 12e-15 * 4e-6 * 4e-6 * np.pi
    # pulse_ref_energy = pulse_ref_energy * 1e12  # pJ
    # angle = ref_angle + 270 * np.log10(pulse_energy / pulse_ref_energy)

    laser_table = scipy.io.loadmat(mat_path)

    angle_val = laser_table['angle_vals'].flatten()
    P_L_val = laser_table['P_L_vals'].flatten()

    # Take the logarithm of both angle and P_L
    log_angle_val = angle_val
    # log_P_L_val = np.log(P_L_val)

    # Create an interpolation function
    interp_func = interp1d(log_angle_val, P_L_val, fill_value="extrapolate")

    # Generate points for plotting
    x_values = np.linspace(log_angle_val.min(), log_angle_val.max(), 100)
    y_values = interp_func(x_values)

    # Plot original data and interpolation
    plt.figure(figsize=(8, 6))
    plt.scatter(log_angle_val, P_L_val, label='Original Data')
    plt.plot(x_values, y_values, label='Interpolation')
    plt.xlabel('Log Angle Value')
    plt.ylabel('Pulse Energy')
    plt.title('Interpolation of Pulse Energy vs. Log Angle')
    plt.legend()
    plt.grid(True)
    plt.show()


    with h5py.File(source_file, 'r') as data:
        laser_angle = data['dld/laser_intensity'][:]
    # laser_P_L = interp_func(laser_angle)
    # Apply interpolation on the logarithmic scale
    laser_P_L = interp_func(laser_angle)
    # Exponentiate to get back to the original scale
    b = np.unique(laser_P_L)
    # laser_P_L = np.exp(laser_P_L)
    laser_P_L = laser_P_L * 1e-3 / 2 / 100e3
    laser_P_L = laser_P_L * 1e12

    with h5py.File(target_file, 'r+') as data:
        del data['dld/pulse']
        data.create_dataset("dld/pulse", data=laser_P_L, dtype='f')





if __name__ == "__main__":
    txt_path = 'E:/Jonas/227_Feb-18-2025_08-57_W_poly/output_DAn.txt'
    save_path = 'E:/Jonas/227_Feb-18-2025_08-57_W_poly/data_227_Feb-18-2025_08-57_W_poly.h5'
    copy_xytof_from_cobold_txt_to_hdf5(txt_path, save_path)
    # mat_path = 'T:/Monajem/physics_atom_probe_data/Backup_data/Power_vals_calibration.mat'
    # source_file = ('T:/Monajem/physics_atom_probe_data/Backup_data/Measurements_2024_02_01/'
    #                '207_Feb-01-2024_13-08_Powersweep/data_207_Feb-01-2024_13-08_Powersweep.h5')
    # target_file = '../../../tests/data/physics_experiment/data_207_Feb-01-2024_13-08_Powersweep.h5'
    #
    # laser_pulse_energy_from_mat_file(mat_path, source_file, target_file)

    # file_path = '../../../tests/data/physics_experiment/data_130_Sep-19-2023_14-58_W_12fs.h5'
    # (at 242°) corresponds to an intensity of 1.4e13 W/cm^2.
    # 170 fs the highest intensity is at 3.4e13 W/cm^2
    # Energy per pulse (J) = Power Density (W/cm^2) * Area (cm^2) * Pulse Duration (s)
    # ref_angle = 242
    # ref_laser_intensity = 3.4e13 * 12e-15 * 4e-4 * 4e-4 * np.pi / 1e-12
    # # ref_laser_intensity = 1.4 * 12 * 4 * 4 * np.pi
    # print(ref_laser_intensity)
    # convert_ND_angle_to_laser_intensity(file_path, ref_laser_intensity, ref_angle)

    # file_path = '../../../tests/data/physics_experiment/data_207_Feb-01-2024_13-08_Powersweep.h5'
    # rename_a_category(file_path, 'dld/AbsoluteTimeStamp', 'dld/start_counter')
    # rename_a_category(file_path, 'dld/laser_intensity', 'dld/pulse')
    # fill_a_category(file_path, 'dld/start_counter')
    # pulse_energy_calculator_list(242, 1.4e13, 5000)
    print('Done')

#########

# ref_angle = 260
# ref_laser_intensity = 1.4e13
# pulse_energy = ref_laser_intensity * (10 ** ((angle - ref_angle) / (270 * 0.5)))
# # 1 μm^2 =1×10^−8 cm^2
# #Energy per pulse (J) = Power Density (W/cm^2) * Area (cm^2) * Pulse Duration (s)
# pulse_energy = pulse_energy * 6e-8 *  12e-15 * 1e12
# variables.data['pulse'] = pulse_energy
