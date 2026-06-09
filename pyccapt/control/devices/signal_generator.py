import time

import pyvisa


def _open_session(device_resource):
    """Open a pyvisa session as a context manager.

    Previously each call constructed a fresh ``ResourceManager`` and
    ``open_resource`` and never closed either, leaking VISA sessions
    every time the user pressed start/stop. Use a tiny helper so each
    function shares the same open-then-close lifecycle.
    """
    resources = pyvisa.ResourceManager()
    return resources, resources.open_resource(device_resource)


def _close_session(resources, wave_generator):
    """Close VISA handles, swallowing per-call errors so cleanup never raises."""
    try:
        wave_generator.close()
    except Exception:
        pass
    try:
        resources.close()
    except Exception:
        pass


def initialize_signal_generator(variables, freq):
    """
    Initialize the signal generator.

    Args:
            variables: Instance of variables class.
            freq: Frequency at which signal needs to be generated.

    Returns:
            None
    """
    freq1_command = 'C1:BSWV FRQ,%s' % (freq * 1000)
    freq2_command = 'C2:BSWV FRQ,%s' % (freq * 1000)

    device_resource = variables.COM_PORT_signal_generator
    resources, wave_generator = _open_session(device_resource)
    try:
        wave_generator.write('C1:OUTP OFF')  # Turn off channel 1
        time.sleep(0.01)
        wave_generator.write(freq1_command)  # Set output frequency on channel 1
        time.sleep(0.01)
        wave_generator.write('C1:BSWV DUTY,1')  # Set 30% duty cycle on channel 1
        time.sleep(0.01)
        wave_generator.write('C1:BSWV RISE,0.000000002')  # Set 0.2ns rising edge on channel 1
        time.sleep(0.01)
        wave_generator.write('C1:BSWV DLY,0')  # Set 0 second delay on channel 1
        time.sleep(0.01)
        wave_generator.write('C1:BSWV HLEV,5')  # Set 5v high level on channel 1
        time.sleep(0.01)
        wave_generator.write('C1:BSWV LLEV,0')  # Set 0v low level on channel 1
        time.sleep(0.01)
        wave_generator.write('C1:OUTP LOAD,50')  # Set 50 ohm load on channel 1
        time.sleep(0.01)
        wave_generator.write('C1:OUTP ON')  # Turn on channel 1

        wave_generator.write('C2:OUTP OFF')  # Turn off channel 2
        time.sleep(0.01)
        wave_generator.write(freq2_command)  # Set output frequency on channel 2
        time.sleep(0.01)
        wave_generator.write('C2:BSWV DUTY,1')  # Set 30% duty cycle on channel 2
        time.sleep(0.01)
        wave_generator.write('C2:BSWV RISE,0.000000002')  # Set 0.2ns rising edge on channel 2
        time.sleep(0.01)
        wave_generator.write('C2:BSWV DLY,0')  # Set 0 second delay on channel 2
        time.sleep(0.01)
        wave_generator.write('C2:BSWV HLEV,5')  # Set 5v high level on channel 2
        time.sleep(0.01)
        wave_generator.write('C2:BSWV LLEV,0')  # Set 0v low level on channel 2
        time.sleep(0.01)
        wave_generator.write('C2:OUTP LOAD,50')  # Set 50 ohm load on channel 2
        time.sleep(0.01)
        wave_generator.write('C2:OUTP ON')  # Turn on channel 2
    finally:
        _close_session(resources, wave_generator)


def change_frequency_signal_generator(variables, freq):
    """
    Change the frequency of the signal generator.

    Args:
            variables: Instance of variables class.
            freq: Frequency at which signal needs to be generated.

    Returns:
            None
    """
    freq1_command = 'C1:BSWV FRQ,%s' % (freq * 1000)
    freq2_command = 'C2:BSWV FRQ,%s' % (freq * 1000)

    device_resource = variables.COM_PORT_signal_generator
    resources, wave_generator = _open_session(device_resource)
    try:
        wave_generator.write(freq1_command)  # Set output frequency on channel 1
        time.sleep(0.01)
        wave_generator.write(freq2_command)  # Set output frequency on channel 2
        time.sleep(0.01)
        print(f"Frequency changed to {freq} kHz")
    finally:
        _close_session(resources, wave_generator)


def turn_off_signal_generator(variables=None):
    """
    Turn off the signal generator.

    Args:
            variables: Instance of variables class. The VISA resource
              address is taken from ``variables.COM_PORT_signal_generator``
              so different rigs can use different Siglent / generic
              wave generators. The previous hardcoded
              ``USB0::0xF4EC::0x1101::SDG6XBAD2R0601::INSTR`` was the
              serial number of one specific unit -- ``turn_off_*`` was
              silently a no-op on every other machine because
              ``open_resource`` raised and the exception was unhandled.

    Returns:
            None
    """
    if variables is None or not getattr(variables, 'COM_PORT_signal_generator', ''):
        print(
            'turn_off_signal_generator: no signal generator VISA address '
            'available; ensure variables.COM_PORT_signal_generator is set '
            'from config.toml.'
        )
        return

    device_resource = variables.COM_PORT_signal_generator
    resources, wave_generator = _open_session(device_resource)
    try:
        wave_generator.write('C2:OUTP OFF')  # Turn off channel 2
        time.sleep(0.01)
        wave_generator.write('C1:OUTP OFF')  # Turn off channel 1
        time.sleep(0.01)
    finally:
        _close_session(resources, wave_generator)
