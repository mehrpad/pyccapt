import datetime
from pathlib import Path


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
    elif variables.pulse_mode in ('Laser', 'VoltageLaser'):
        # variables.laser_freq is stored in Hz; the GUI's repetition-rate
        # combo selects 400 000, 500 000, ..., 1 000 000 Hz. Convert to kHz
        # for the human-readable summary so the label and value agree.
        # Effective output rate = base / division.
        try:
            base_freq_khz = float(variables.laser_freq) / 1000.0
            div = max(int(variables.laser_division_factor or 1), 1)
            output_rate_khz = base_freq_khz / div
        except (TypeError, ValueError, ZeroDivisionError):
            base_freq_khz = float('nan')
            output_rate_khz = float('nan')
        # Pulse energy is now computed and tracked by the laser GUI; if it
        # was never set (laser disabled / never connected), fall back to 0.
        pulse_energy_nJ = float(getattr(variables, 'laser_pulse_energy', 0) or 0)
        statistics = f"""
Experiment Elapsed Time (Sec): {variables.elapsed_time:.3f}
Experiment Total Ions: {variables.total_ions}
Specimen Max Achieved Voltage (V): {variables.specimen_voltage:.3f}
Specimen Max Achieved Pulse Voltage (V): {variables.pulse_voltage:.3f}
Laser pulse energy (nJ): {pulse_energy_nJ:.3f}
Laser average power (mW): {variables.laser_average_power:.3f}
Laser base pulse frequency (kHz): {base_freq_khz:.3f}
Laser division factor: {variables.laser_division_factor}
Laser output pulse frequency (kHz): {output_rate_khz:.3f}
Laser IR power setpoint (W): {variables.laser_power:.3f}
Last detection rate: {variables.detection_rate_current_plot:.3f}%
-----------------------------------------------------
"""
    else:
        # Defensive default so the file write never raises
        # UnboundLocalError when pulse_mode is something unexpected.
        statistics = f"""
Experiment Elapsed Time (Sec): {variables.elapsed_time:.3f}
Experiment Total Ions: {variables.total_ions}
Specimen Max Achieved Voltage (V): {variables.specimen_voltage:.3f}
Pulse Mode: {variables.pulse_mode!r} (no specialised statistics block)
Last detection rate: {variables.detection_rate_current_plot:.3f}%
-----------------------------------------------------
"""
    software_info = "Created by PyCCAPT software."

    output_path = Path(variables.path) / 'parameters.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(statistics)
        f.write(software_info)
