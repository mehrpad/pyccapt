import serial.tools.list_ports

from pyccapt.control.devices import signal_generator, email_send


def initialization_signal_generator(variables, log_apt):
    """
    Initialize the signal generator.

    Args:
            signal_generator: The class object of the SignalGenerator class.
            variables: The class object of the Variables class.
            log_apt: The logger object.

    Returns:
            initialization_error: The boolean flag to indicate if the initialization is successful.
    """
    # Initialize the signal generator
    try:
        signal_generator.initialize_signal_generator(variables, variables.pulse_frequency)
        log_apt.info('Signal generator is initialized')
        initialization_error = False
    except Exception as e:
        log_apt.info('Signal generator is not initialized')
        print('Can not initialize the signal generator')
        print('Make the signal_generator off in the config file or fix the error below')
        print(e)
        variables.stop_flag = True
        initialization_error = True
        log_apt.info('Experiment is terminated')
    return initialization_error


def command_v_p(com_port_v_p, cmd):
    """
    Send commands to the pulser.

    This method sends commands to the pulser over the COM port and reads the response.

    Args:
            com_port_v_p (serial.Serial): The COM port object for the pulser.
            cmd (str): The command to send.

    Returns:
            str: The response received from the device.
    """
    if cmd == 'close':
        com_port_v_p.close()
    else:
        cmd = cmd + '\r\n'
        com_port_v_p.write(cmd.encode())
    # response = self.com_port_v_p.readline().decode().strip()
    # return response


def command_v_dc(com_port_v_dc, cmd):
    """
    Send commands to the high voltage parameter: v_dc.

    This method sends commands to the V_dc source over the COM port and reads the response.

    Args:
            com_port_v_dc (serial.Serial): The COM port object for the V_dc source.
            cmd (str): The command to send.

    Returns:
            str: The response received from the device.
    """

    if cmd == 'close':
        com_port_v_dc.close()
    else:
        com_port_v_dc.write((cmd + '\r\n').encode())
    # response = ''
    # try:
    #     while self.com_port_v_dc.in_waiting > 0:
    #         response = self.com_port_v_dc.readline()
    # except Exception as error:
    #     print(error)
    #
    # if isinstance(response, bytes):
    #     response = response.decode("utf-8")

    # return response


def initialization_v_dc(com_port_v_dc, log_apt, variables):
    """
    Initialize the high voltage.

    Args:
            com_port_v_dc: The COM port object for the high voltage.
            log_apt: The logger object.
            variables: The class object of the Variables class.

    Returns:
            initialization_error: The boolean flag to indicate if the initialization is successful.
    """

    try:
        # Initialize high voltage
        if com_port_v_dc.is_open:
            com_port_v_dc.flushInput()
            com_port_v_dc.flushOutput()

            cmd_list = [">S1 3.0e-4", ">S0B 0", ">S0 %s" % variables.vdc_min, "F0", ">S0?", ">DON?", ">S0A?"]
            for cmd in range(len(cmd_list)):
                command_v_dc(com_port_v_dc, cmd_list[cmd])
        else:
            print("Couldn't open Port!")
            exit()
        log_apt.info('High voltage is initialized')
        initialization_error = False
    except Exception as e:
        log_apt.info('High voltage is  not initialized')
        print('Can not initialize the high voltage')
        print('Make the v_dc off in the config file or fix the error below')
        print(e)
        variables.stop_flag = True
        initialization_error = True
        log_apt.info('Experiment is terminated')
    return initialization_error


def initialization_v_p(com_port_v_p, log_apt, variables):
    """
    Initialize the pulser.

    Args:
            com_port_v_p: The COM port object for the pulser.
            log_apt: The logger object.
            variables: The class object of the Variables class.

    Return:
            initialization_error: The boolean flag to indicate if the initialization is successful.
    """

    try:
        command_v_p(com_port_v_p, '*RST')
        log_apt.info('Pulser is initialized')
        initialization_error = False
    except Exception as e:
        log_apt.info('Pulser is not initialized')
        print('Can not initialize the pulser')
        print('Make the v_p off in the config file or fix the error below')
        print(e)
        variables.stop_flag = True
        initialization_error = True
        log_apt.info('Experiment is terminated')
    return initialization_error


def send_info_email(log_apt, variables, conf, interim=False):
    """
    Send the information email.

    The body carries the same run-statistics-on-top / setup-parameters
    report that is written to the experiment folder, so the recipient sees
    everything inline without having to open the attachment. The latest
    Visualization snapshot is attached / inlined by ``email_send``.

    Args:
            log_apt: The logger object.
            variables: The class object of the Variables class.
            conf: The configuration dictionary (needed to build the report).
            interim: When True this is a periodic progress e-mail sent mid-run
                    (every ``email_interval_events`` ions) rather than the final
                    end-of-experiment report; only the subject line differs.

    Returns:
            None
    """
    # Imported lazily so this device-side helper does not pull in the core
    # package at import time.
    from pyccapt.control.core import experiment_statistics

    if interim:
        subject = 'Experiment {} Progress ({} ions) on {}'.format(
            variables.hdf5_data_name, variables.total_ions, variables.start_time
        )
    else:
        subject = 'Experiment {} Report on {}'.format(variables.hdf5_data_name, variables.start_time)
    elapsed_time_temp = float("{:.3f}".format(variables.elapsed_time))
    message = (
        'The experiment was started at: {}\n'
        'The experiment was ended at: {}\n'
        'Experiment duration: {}\n'
        'Total number of ions: {}\n\n'.format(
            variables.start_time, variables.end_time, elapsed_time_temp, variables.total_ions
        )
    )

    # Full report: run statistics on top, separated by a line from the
    # setup parameters. Same content as the attached experiment-details file.
    message += experiment_statistics.build_statistics_text(variables, conf)

    # Pass variables through so the email module can attach apt.log and
    # parameters.txt from this experiment's folder. Any failure (missing
    # credentials, SMTP error, attachment IO error) raises and is caught
    # by the caller's try/except in apt_exp_control.run_experiment.
    attached = email_send.send_email(variables.email, subject, message, variables=variables, interim=interim)
    if attached:
        log_apt.info('Email is sent (attachments: %s)', ', '.join(attached))
    else:
        log_apt.info('Email is sent (no attachments)')
