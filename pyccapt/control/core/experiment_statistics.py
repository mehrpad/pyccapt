import datetime


def save_statistics_apt(variables, conf):
    """
    Save setup parameters and run statistics in a text file.

    Args:
        variables (object): An object containing experiment variables.
        conf (dict): A dictionary containing the configuration file parameters.

    Returns:
        None
    """
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Create a header with additional information
    header = f"""
Experiment Parameters and Statistics
-------------------------------------------
Experiment Timestamp: {current_datetime}
Username: {variables.user_name}
Experiment Name: {variables.ex_name}
Electrode Name: {variables.electrode}
Maximum Experiment Time: {variables.ex_time} seconds
Maximum Number of Ions: {variables.max_ions}
Control Refresh Frequency: {variables.ex_freq} Hz
Specimen DC Voltage Range (Min-Max): {variables.vdc_min} V - {variables.vdc_max} V
K_p Upwards: {variables.vdc_step_up}
K_p Downwards: {variables.vdc_step_down}
Control Algorithm: {variables.control_algorithm}
Pulse Mode: {variables.pulse_mode}
    """
    if variables.pulse_mode == 'Voltage':
        header += f"""
Pulse Voltage Range (Min-Max): {variables.v_p_min} V - {variables.v_p_max} V
Stop Criteria:
Criteria Time: {variables.criteria_time}
Criteria DC Voltage: {variables.criteria_vdc}
Criteria Ions: {variables.criteria_ions}
"""

    header += f"""
Pulse Fraction: {variables.pulse_fraction * 100} %
Pulse Frequency: {variables.pulse_frequency} kHz
Detection Rate: {variables.detection_rate} %
Counter Source: {variables.counter_source}
Email: {variables.email}
-----------------------------------------------------
Device name: {conf['device_name']}
t_0_laser (Sec): {conf['t_0_laser']}
t_0_voltage (Sec): {conf['t_0_voltage']}
flight path distance (cm): {conf['flight_path_length']}
TDC model: {conf['tdc_model']}
"""
    if variables.pulse_mode == 'Voltage':
        statistics = f"""
Experiment Elapsed Time (Sec): {variables.elapsed_time:.3f}
Experiment Total Ions: {variables.total_ions}
Specimen Max Achieved Voltage (V): {variables.specimen_voltage:.3f}

Specimen Max Achieved Pulse Voltage (V): {variables.pulse_voltage:.3f}
Last detection rate: {variables.detection_rate_current_plot:.3f}%
-----------------------------------------------------
"""
    elif variables.pulse_mode == 'Laser':
        statistics = f"""
Experiment Elapsed Time (Sec): {variables.elapsed_time:.3f}
Experiment Total Ions: {variables.total_ions}
Specimen Max Achieved Voltage (V): {variables.specimen_voltage:.3f}
Laser pulse energy (): {0.0:.3f}
Laser average power (mW): {variables.laser_average_power:.3f}
Laser pulse frequency (kHz): {variables.laser_freq}
Laser power (mW): {variables.laser_power:.3f}
Laser division factor: {variables.laser_division_factor:.3f}
Last detection rate: {variables.detection_rate_current_plot:.3f}%
-----------------------------------------------------
"""
    software_info = "Created by PyCCAPT software."

    with open(variables.path + '\\parameters.txt', 'w') as f:
        f.write(header)
        f.write(statistics)
        f.write(software_info)
