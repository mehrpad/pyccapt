try:
    import thorlabs_apt.core as apt
except Exception as e:
    print('Thorlabs APT library could not be imported')
    print(e)


def thorlab(conf, degree, step_increase=False, initialize=False):
    """
    Initialize the Thorlabs motor controller and move it to the specified degree.

    Args:
            conf (dict): Configuration settings.
                    Should contain 'COM_PORT_thorlab_motor' for the COM port number.
            degree (float): Degree value to move the motor to.
            step_increase (bool, optional): Whether to move the motor in steps if degree > 180.
            initialize (bool, optional): Whether to initialize the motor and move to home position.

    Returns:
            None
    """
    if conf['thorlab_motor'] == 'on':
        motor = apt.Motor(int(conf['COM_PORT_thorlab_motor']))

        if initialize:
            motor.set_move_home_parameters(2, 1, 10, 4)
            motor.move_home(True)
            if degree != 0:
                motor.move_by(degree)
        elif step_increase:
            if degree > 180:
                motor.move_by(int(degree / 2), blocking=True)
                motor.move_by(degree - int(degree / 2), blocking=True)
            else:
                motor.move_by(degree)
        else:
            if degree > 180:
                motor.move_to(int(degree / 2), blocking=True)
                motor.move_to(degree, blocking=True)
            else:
                motor.move_to(degree)
