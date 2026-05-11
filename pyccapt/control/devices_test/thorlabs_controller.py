import multiprocessing
import time


def thorlab(degree, initialize=False):
    """
    Initialize the Thorlabs controller and set it to the specified degree.

    Args:
            degree (float): The degree to move the motor to.
            initialize (bool): Whether to perform motor initialization.
    """
    import thorlabs_apt.core as apt

    motor = apt.Motor(27261754)

    if initialize:
        motor.move_home(True)
    else:
        motor.move_by(degree * 2, blocking=True)
        time.sleep(3)
        motor.move_to(degree, blocking=True)


if __name__ == '__main__':
    process1 = multiprocessing.Process(target=thorlab, args=(10, True))
    process1.start()
    process1.join()

    process2 = multiprocessing.Process(target=thorlab, args=(10, False))
    process2.start()
    process2.join()
