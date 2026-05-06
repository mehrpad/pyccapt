import csv
import os
import time
from datetime import datetime

import serial.tools.list_ports

from pyccapt.control.core import runtime
from pyccapt.control.devices.edwards_tic import EdwardsAGC
from pyccapt.control.devices.pfeiffer_gauges import TPG362


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _available_serial_ports_text() -> str:
    ports = sorted(port.device for port in serial.tools.list_ports.comports() if getattr(port, "device", ""))
    return ", ".join(ports) if ports else "none detected"


def _format_port_error(device_label, port, exc) -> str:
    return (
        f"{device_label} unavailable on {port}: {exc}. "
        f"Available serial ports: {_available_serial_ports_text()}"
    )


def _serial_port_present(port):
    if not port:
        return False
    try:
        ports = {getattr(p, "device", "") for p in serial.tools.list_ports.comports()}
    except Exception:
        return True
    return port in ports


def _open_cryovac_serial(port):
    """Open the Cryovac serial port with a few retries for transient enumeration glitches.

    On Windows the USB-to-serial bridge for the TIC 500 sometimes briefly
    drops out of the COM port table during boot — opening it with no retry
    fails with OSError 22 ("A device which does not exist was specified")
    even though the device itself is fine and shows up a moment later.
    """

    last_exc = None
    for attempt in range(3):
        if _serial_port_present(port) or attempt > 0:
            try:
                return serial.Serial(
                    port=port,
                    baudrate=9600,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.5,
                    write_timeout=1.0,
                )
            except Exception as e:
                last_exc = e
        time.sleep(0.6)
    if last_exc is None:
        last_exc = OSError(f"port '{port}' not in serial enumeration")
    raise last_exc


def command_cryovac(cmd, com_port_cryovac):
    """
    Execute a command on Cryovac through serial communication.

    Vendor / origin
    ---------------
    The CLI command set used here (``getOutput``, ``getOutputNames``,
    ``Out1Cryo.PID.Setpoint``, ``Out2LL.PID.Setpoint``, ``e_mlp``-style
    queries, etc.) is defined and owned by **CryoVac GmbH** for their
    TIC 500 programmable temperature controller. This function and its
    siblings (``initialize_cryovac`` etc.) are thin wrappers around
    that ASCII serial protocol; the protocol itself is CryoVac's
    property. Refer to the CryoVac TIC 500 manual (e.g. document
    5984591) for the authoritative reference.

    Args:
        cmd: Command to be executed.
        com_port_cryovac: Serial communication object.

    Returns:
        Response code after executing the command.
    """
    com_port_cryovac.write((cmd + '\r\n').encode())
    time.sleep(0.1)
    response = ''
    while com_port_cryovac.in_waiting > 0:
        response = com_port_cryovac.readline()
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    return response


def command_edwards(conf, variables, cmd, E_AGC, status=None):
    """
    Execute commands and set flags based on parameters.

    Args:
        conf: Configuration parameters.
        variables: Variables instance.
        cmd: Command to be executed.
        E_AGC: EdwardsAGC instance.
        status: Status of the lock.

    Returns:
        Response code after executing the command.
    """

    response = -1
    try:
        if variables.flag_pump_load_lock_click and variables.flag_pump_load_lock and status == 'load_lock':
            if conf['pump_ll'] == "on":
                E_AGC.comm('!C910 0')
                E_AGC.comm('!C904 0')
            variables.flag_pump_load_lock_click = False
            variables.flag_pump_load_lock = False
            variables.flag_pump_load_lock_led = False
            time.sleep(1)
        elif variables.flag_pump_load_lock_click and not variables.flag_pump_load_lock and status == 'load_lock':
            if conf['pump_ll'] == "on":
                E_AGC.comm('!C910 1')
                E_AGC.comm('!C904 1')
            variables.flag_pump_load_lock_click = False
            variables.flag_pump_load_lock = True
            variables.flag_pump_load_lock_led = True
            time.sleep(1)

        if variables.flag_pump_cryo_load_lock_click and variables.flag_pump_cryo_load_lock and status == 'cryo_load_lock':
            if conf['pump_cll'] == "on":
                E_AGC.comm('!C910 0')
                E_AGC.comm('!C904 0')
            variables.flag_pump_cryo_load_lock_click = False
            variables.flag_pump_cryo_load_lock = False
            variables.flag_pump_cryo_load_lock_led = False
            time.sleep(1)
        elif (variables.flag_pump_cryo_load_lock_click and not variables.flag_pump_cryo_load_lock and
              status == 'cryo_load_lock'):
            if conf['pump_cll'] == "on":
                E_AGC.comm('!C910 1')
                E_AGC.comm('!C904 1')
            variables.flag_pump_cryo_load_lock_click = False
            variables.flag_pump_cryo_load_lock = True
            variables.flag_pump_cryo_load_lock_led = True
            time.sleep(1)

        if conf['COM_PORT_gauge_ll'] != "off" or conf['COM_PORT_gauge_cll'] != "off":
            if cmd == 'pressure':
                response_tmp = E_AGC.comm('?V911')
                response_tmp = float(response_tmp.replace(';', ' ').split()[1])

                if response_tmp < 90 and status == 'load_lock':
                    variables.flag_pump_load_lock_led = False
                elif response_tmp >= 90 and status == 'load_lock':
                    variables.flag_pump_load_lock_led = True
                if response_tmp < 90 and status == 'cryo_load_lock':
                    variables.flag_pump_cryo_load_lock_led = False
                elif response_tmp >= 90 and status == 'cryo_load_lock':
                    variables.flag_pump_cryo_load_lock_led = True
                response = E_AGC.comm('?V940')
            else:
                print('Unknown command for Edwards TIC Load Lock')
    except Exception:
        response = -1  # Set response to -1 indicate an error

    return response


def initialize_cryovac(com_port_cryovac, variables):
    """
    Initialize the communication port of Cryovac.

    Args:
        com_port_cryovac: Serial communication object.
        variables: Variables instance.

    Returns:
        None
    """
    # The TIC 500 firmware can fault on screen with "A system crash occurred /
    # Diagnostic code: bpt0" if the first command we send arrives concatenated
    # with stale bytes left in its macro parser (e.g. a partial line from a
    # previous session, or line transitions caused by the host opening the
    # port). Drain both directions, send a bare line terminator so any
    # in-flight half-line is flushed as an empty macro, then drain again
    # before issuing the real query.
    time.sleep(0.2)
    try:
	    com_port_cryovac.reset_input_buffer()
	    com_port_cryovac.reset_output_buffer()
	    com_port_cryovac.write(b'\r\n')
	    time.sleep(0.15)
	    com_port_cryovac.reset_input_buffer()
    except Exception:
	    pass

    output = ''
    for _ in range(5):
	    output = command_cryovac('getOutput?', com_port_cryovac)
	    if output and output.split():
		    break
	    time.sleep(0.2)

    try:
	    variables.temperature = float(output.split()[0].replace(',', ''))
    except (ValueError, IndexError):
	    variables.temperature = -1


def initialize_edwards_tic_load_lock(conf, variables):
    """
    Initialize TIC load lock parameters.

    Args:
        conf: Configuration parameters.
        variables: Variables instance.

    Returns:
        None
    """

    E_AGC_ll = EdwardsAGC(variables.COM_PORT_gauge_ll)
    response = command_edwards(conf, variables, 'pressure', E_AGC=E_AGC_ll)
    variables.vacuum_load_lock = float(response.replace(';', ' ').split()[2]) * 0.01
    variables.vacuum_load_lock_backing = float(response.replace(';', ' ').split()[4]) * 0.01


def initialize_edwards_tic_cryo_load_lock(conf, variables):
    """
    Initialize TIC cryo load lock parameters.

    Args:
        conf: Configuration parameters.
        variables: Variables instance.

    Returns:
        None
    """
    E_AGC_cll = EdwardsAGC(variables.COM_PORT_gauge_cll)
    response = command_edwards(conf, variables, 'pressure', E_AGC=E_AGC_cll)
    variables.vacuum_cryo_load_lock = float(response.replace(';', ' ').split()[2]) * 0.01
    variables.vacuum_cryo_load_lock_backing = float(response.replace(';', ' ').split()[4]) * 0.01


def initialize_edwards_tic_buffer_chamber(conf, variables):
    """
    Initialize TIC buffer chamber parameters.

    Args:
        conf: Configuration parameters.
        variables: Variables instance.

    Returns:
        None
    """
    E_AGC_bc = EdwardsAGC(variables.COM_PORT_gauge_bc)
    response = command_edwards(conf, variables, 'pressure', E_AGC=E_AGC_bc)
    variables.vacuum_buffer_backing = float(response.replace(';', ' ').split()[2]) * 0.01


def initialize_pfeiffer_gauges(variables):
    """
    Initialize Pfeiffer gauge parameters.

    Args:
        variables: Variables instance.

    Returns:
        None
    """
    tpg = TPG362(port=variables.COM_PORT_gauge_mc)
    value, _ = tpg.pressure_gauge(2)
    variables.vacuum_main = float(value)
    value, _ = tpg.pressure_gauge(1)
    variables.vacuum_buffer = float(value)


def state_update(conf, variables, emitter):
    """
    Read gauge parameters and update variables.

    Args:
        conf: Configuration parameters.
        variables: Variables instance.
        emitter: Emitter instance.

    Returns:
        None
    """

    tpg = None
    E_AGC_bc = None
    E_AGC_ll = None
    E_AGC_cll = None
    reported_runtime_issues: set[str] = set()

    def report_once(issue_key: str, message: str) -> None:
        if issue_key not in reported_runtime_issues:
            print(message)
            reported_runtime_issues.add(issue_key)

    def clear_issue(issue_key: str) -> None:
        reported_runtime_issues.discard(issue_key)

    if conf['gauges'] == "on":
        if conf['COM_PORT_gauge_mc'] != "off":
            try:
                tpg = TPG362(port=variables.COM_PORT_gauge_mc)
            except Exception as e:
                print(
                    f"{bcolors.FAIL}"
                    f"{_format_port_error('Analysis chamber gauge', variables.COM_PORT_gauge_mc, e)}"
                    f"{bcolors.ENDC}"
                )
                tpg = None

        if conf['COM_PORT_gauge_bc'] != "off":
            try:
                E_AGC_bc = EdwardsAGC(variables.COM_PORT_gauge_bc, variables)
            except Exception as e:
                print(
                    f"{bcolors.FAIL}"
                    f"{_format_port_error('Buffer chamber gauge', variables.COM_PORT_gauge_bc, e)}"
                    f"{bcolors.ENDC}"
                )
                E_AGC_bc = None

        if conf['COM_PORT_gauge_ll'] != "off":
            try:
                E_AGC_ll = EdwardsAGC(variables.COM_PORT_gauge_ll, variables)
            except Exception as e:
                print(
                    f"{bcolors.FAIL}"
                    f"{_format_port_error('Load-lock gauge', variables.COM_PORT_gauge_ll, e)}"
                    f"{bcolors.ENDC}"
                )
                E_AGC_ll = None

        if conf['COM_PORT_gauge_cll'] != "off":
            try:
                E_AGC_cll = EdwardsAGC(variables.COM_PORT_gauge_cll, variables)
            except Exception as e:
                print(
                    f"{bcolors.FAIL}"
                    f"{_format_port_error('Cryo load-lock gauge', variables.COM_PORT_gauge_cll, e)}"
                    f"{bcolors.ENDC}"
                )
                E_AGC_cll = None

    if conf['cryo'] == "off":
        print('The cryo temperature monitoring is off')
    else:
        try:
            com_port_cryovac = _open_cryovac_serial(variables.COM_PORT_cryo)
            initialize_cryovac(com_port_cryovac, variables)
        except Exception as e:
            com_port_cryovac = None
            print(_format_port_error('Cryovac', variables.COM_PORT_cryo, e))

        cryovac_reconnect_interval = 30.0
        last_cryovac_reconnect_attempt = time.time()

        start_time = time.time()
        log_time_time_interval = conf['log_time_time_interval']
        vacuum_main = -1.0
        vacuum_buffer = -1.0
        vacuum_buffer_backing = -1.0
        vacuum_load_lock = -1.0
        vacuum_load_lock_backing = -1.0
        vacuum_cryo_load_lock = -1.0
        vacuum_cryo_load_lock_backing = -1.0
        set_temperature_tmp_cryo = 0
        set_temperature_tmp_ll = 0
        while emitter.bool_flag_while_loop:
            if conf['cryo'] == "on" and com_port_cryovac is None:
                now = time.time()
                if now - last_cryovac_reconnect_attempt >= cryovac_reconnect_interval:
                    last_cryovac_reconnect_attempt = now
                    if _serial_port_present(variables.COM_PORT_cryo):
                        try:
                            com_port_cryovac = _open_cryovac_serial(variables.COM_PORT_cryo)
                            initialize_cryovac(com_port_cryovac, variables)
                            clear_issue("cryovac_reconnect")
                            print(f"Cryovac reconnected on {variables.COM_PORT_cryo}")
                        except Exception as e:
                            com_port_cryovac = None
                            report_once(
                                "cryovac_reconnect",
                                _format_port_error('Cryovac', variables.COM_PORT_cryo, e),
                            )
            if conf['cryo'] == "on" and com_port_cryovac is not None:
                try:
                    output = command_cryovac('getOutput', com_port_cryovac)
                except Exception as e:
                    print(e)
                    print("cannot read the cryo temperature")
                    output = '0'
                try:
                    # Output indices are driven by config.toml cryo_sensor_X_index keys.
                    # Change those values to remap which device output feeds each sensor slot.
                    words = output.split()
                    idx1 = conf.get('cryo_sensor_1_index', 2)  # cryo_head_outside
                    idx2 = conf.get('cryo_sensor_2_index', 1)  # cryo_head_inside
                    idx3 = conf.get('cryo_sensor_3_index', 0)  # stage
                    idx4 = conf.get('cryo_sensor_4_index', 3)  # load_lock
                    temperature_cryo_head = float(words[idx1].replace(',', ''))
                    temperature_cryo_head_inside = float(words[idx2].replace(',', ''))
                    temperature_stage = float(words[idx3].replace(',', ''))
                    temperature_ll = float(words[idx4].replace(',', ''))
                except Exception as e:
                    com_port_cryovac = None
                    temperature_cryo_head = -1
                    temperature_cryo_head_inside = -1
                    temperature_stage = -1
                    print(e)
                    # Handle the case where response is not a valid float
                    temperature = -1
                variables.temperature = temperature_stage
                emitter.temp_stage.emit(temperature_stage)
                emitter.temp_cryo_head_inside.emit(temperature_cryo_head_inside)
                emitter.temp_cryo_head.emit(temperature_cryo_head)
                emitter.temp_ll.emit(temperature_ll - 273.15)  # convert from kelvin to celsius

                if variables.set_temperature_flag_cryo:
                    if variables.set_temperature_cryo != set_temperature_tmp_cryo:
                        try:
                            res = command_cryovac(f'Out1Cryo.PID.Setpoint {variables.set_temperature_cryo}',
                                                  com_port_cryovac)
                            print(res)
                            set_temperature_tmp_cryo = variables.set_temperature_cryo
                        except Exception as e:
                            print(e)
                            print("cannot set the cryo temperature")
                elif variables.set_temperature_flag_cryo == False:
                    variables.set_temperature_cryo = 0
                    res = command_cryovac(f'Out1Cryo.PID.Setpoint {variables.set_temperature_cryo}', com_port_cryovac)
                    variables.set_temperature_flag_cryo = None

                if variables.set_temperature_flag_ll:
                    if variables.set_temperature_ll != set_temperature_tmp_ll:
                        try:
                            # convert from celcius to kelvin
                            set_temperature_ll = variables.set_temperature_ll + 273.15
                            res = command_cryovac(f'Out2LL.PID.Setpoint {set_temperature_ll}', com_port_cryovac)
                            print(res)
                            set_temperature_tmp_ll = variables.set_temperature_ll
                        except Exception as e:
                            print(e)
                            print("cannot set the load lock temperature")
                elif variables.set_temperature_flag_ll == False:
                    variables.set_temperature_ll = 0
                    res = command_cryovac(f'Out2LL.PID.Setpoint {variables.set_temperature_ll}', com_port_cryovac)
                    variables.set_temperature_flag_ll = None
            if conf['COM_PORT_gauge_mc'] != "off" and tpg is not None:
                value, _ = tpg.pressure_gauge(2)
                try:
                    vacuum_main = float(value)
                    clear_issue("vacuum_main")
                except Exception as e:
                    report_once("vacuum_main", f"Error reading Temperature:{e}")
                    # Handle the case where response is not a valid float
                    vacuum_main = -1.0
                variables.vacuum_main = vacuum_main
                emitter.vacuum_main.emit(float(vacuum_main))
                value, _ = tpg.pressure_gauge(1)
                try:
                    vacuum_buffer = float(value)
                    clear_issue("vacuum_buffer")
                except Exception as e:
                    report_once("vacuum_buffer", f"Error reading BC:{e}")
                    tpg = None
                    # Handle the case where response is not a valid float
                    vacuum_buffer = -1.0
                variables.vacuum_buffer = vacuum_buffer
                emitter.vacuum_buffer.emit(float(vacuum_buffer))
            if conf['pump_ll'] != "off" and E_AGC_ll is not None:
                response = command_edwards(conf, variables, 'pressure', E_AGC=E_AGC_ll, status='load_lock')

                try:
                    if isinstance(response, str):
                        vacuum_load_lock = float(response.replace(';', ' ').split()[2]) * 0.01
                        clear_issue("vacuum_load_lock")
                    else:
                        raise ValueError("device returned no pressure payload")
                except Exception as e:
                    report_once("vacuum_load_lock", f"Error reading LL:{e}")
                    E_AGC_ll = None
                    # Handle the case where response is not a valid float
                    vacuum_load_lock = -1
                try:
                    if isinstance(response, str):
                        vacuum_load_lock_backing = float(response.replace(';', ' ').split()[4]) * 0.01
                        clear_issue("vacuum_load_lock_backing")
                    else:
                        raise ValueError("device returned no backing-pressure payload")
                except Exception as e:
                    report_once("vacuum_load_lock_backing", f"Error reading LL backing:{e}")
                    E_AGC_ll = None
                    # Handle the case where response is not a valid float
                    vacuum_load_lock_backing = -1
                variables.vacuum_load_lock = vacuum_load_lock
                variables.vacuum_load_lock_backing = vacuum_load_lock_backing
                emitter.vacuum_load_lock.emit(vacuum_load_lock)
                emitter.vacuum_load_lock_back.emit(vacuum_load_lock_backing)

            if conf['pump_cll'] != "off" and E_AGC_cll is not None:
                response = command_edwards(conf, variables, 'pressure', E_AGC=E_AGC_cll, status='cryo_load_lock')

                try:
                    if isinstance(response, str):
                        vacuum_cryo_load_lock = float(response.replace(';', ' ').split()[2]) * 0.01
                        clear_issue("vacuum_cryo_load_lock")
                    else:
                        raise ValueError("device returned no pressure payload")
                except Exception as e:
                    report_once("vacuum_cryo_load_lock", f"Error reading CLL:{e}")
                    E_AGC_cll = None
                    # Handle the case where response is not a valid float
                    vacuum_cryo_load_lock = -1
                try:
                    if isinstance(response, str):
                        vacuum_cryo_load_lock_backing = float(response.replace(';', ' ').split()[4]) * 0.01
                        clear_issue("vacuum_cryo_load_lock_backing")
                    else:
                        raise ValueError("device returned no backing-pressure payload")
                except Exception as e:
                    report_once("vacuum_cryo_load_lock_backing", f"Error reading CLL backing:{e}")
                    # Handle the case where response is not a valid float
                    vacuum_cryo_load_lock_backing = -1
                variables.vacuum_cryo_load_lock = vacuum_cryo_load_lock
                variables.vacuum_cryo_load_lock_backing = vacuum_cryo_load_lock_backing
                emitter.vacuum_cryo_load_lock.emit(vacuum_cryo_load_lock)
                emitter.vacuum_cryo_load_lock_back.emit(vacuum_cryo_load_lock_backing)

            if conf['COM_PORT_gauge_bc'] != "off" and E_AGC_bc is not None:
                response = command_edwards(conf, variables, 'pressure', E_AGC=E_AGC_bc)
                try:
                    if isinstance(response, str):
                        vacuum_buffer_backing = float(response.replace(';', ' ').split()[2]) * 0.01
                        clear_issue("vacuum_buffer_backing")
                    else:
                        raise ValueError("device returned no pressure payload")
                except Exception as e:
                    report_once("vacuum_buffer_backing", f"Error reading BC backing:{e}")
                    # Handle the case where response is not a valid float
                    vacuum_buffer_backing = -1
                variables.vacuum_buffer_backing = vacuum_buffer_backing
                emitter.vacuum_buffer_back.emit(vacuum_buffer_backing)

            elapsed_time = time.time() - start_time
            # Every 30 minutes, log the vacuum levels
            if elapsed_time > log_time_time_interval:
                start_time = time.time()
                try:
                    log_vacuum_levels(vacuum_main, vacuum_buffer, vacuum_buffer_backing, vacuum_load_lock,
                                      vacuum_load_lock_backing, vacuum_cryo_load_lock, vacuum_cryo_load_lock_backing)
                except Exception as e:
                    print(e)
                    print("cannot log the vacuum levels")
            time.sleep(1)


def log_vacuum_levels(main_chamber, buffer_chamber, buffer_chamber_pre, load_lock, load_lock_pre,
                      cryo_load_lock, cryo_load_lock_pre):
    """
    Log vacuum levels to a text file and a CSV file.

    Args:
        main_chamber (float): Vacuum level of the main chamber.
        buffer_chamber (float): Vacuum level of the buffer chamber.
        buffer_chamber_pre (float): Vacuum level of the buffer chamber backing pump.
        load_lock (float): Vacuum level of the load lock.
        load_lock_pre(float): Vacuum level of the load lock backing pump.
        cryo_load_lock (float): Vacuum level of the cryo load lock.
        cryo_load_lock_pre (float): Vacuum level of the cryo load lock backing pump.

    Returns:
        None
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_month = datetime.now().strftime("%Y-%m")
    path = runtime.project_path("files", "logs", "vacuum")
    os.makedirs(path, mode=0o777, exist_ok=True)
    txt_file_path = path / f"vacuum_log_{current_month}.txt"
    csv_file_path = path / f"vacuum_log_{current_month}.csv"

    with open(txt_file_path, "a") as log_file:
        log_file.write(f"{timestamp}: Main Chamber={main_chamber}, Buffer Chamber={buffer_chamber}, "
                       f"Buffer Chamber Pre={buffer_chamber_pre}, Load Lock={load_lock}, "
                       f"Load Lock Pre={load_lock_pre}, Cryo Load Lock={cryo_load_lock}, "
                       f"Cryo Load Lock Pre={cryo_load_lock_pre}\n")

    row = [timestamp, main_chamber, buffer_chamber, buffer_chamber_pre, load_lock, load_lock_pre, cryo_load_lock,
           cryo_load_lock_pre]
    header = ["Timestamp", "Main Chamber", "Buffer Chamber", "Buffer Chamber Backing Pump", "Load Lock",
              "Load Lock Backing", 'Cryo Load Lock', 'Cryo Load Lock Backing']

    file_empty = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0

    # Write to CSV file
    with open(csv_file_path, "a", newline='') as log_file:
        csv_writer = csv.writer(log_file)

        # Write the header if the file is empty
        if file_empty:
            csv_writer.writerow(header)

        # Write the data row
        csv_writer.writerow(row)
