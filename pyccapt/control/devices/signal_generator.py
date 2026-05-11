import time

import pyvisa


def initialize_signal_generator(variables, freq):
    """
    Initialize the signal generator.

    Args:
            variables: Instance of variables class.
            freq: Frequency at which signal needs to be generated.

    Returns:
            None
    """
    resources = pyvisa.ResourceManager()

    freq1_command = 'C1:BSWV FRQ,%s' % (freq * 1000)
    freq2_command = 'C2:BSWV FRQ,%s' % (freq * 1000)

    device_resource = variables.COM_PORT_signal_generator

    wave_generator = resources.open_resource(device_resource)

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


def change_frequency_signal_generator(variables, freq):
    """
    Change the frequency of the signal generator.

    Args:
            variables: Instance of variables class.
            freq: Frequency at which signal needs to be generated.

    Returns:
            None
    """
    resources = pyvisa.ResourceManager()

    freq1_command = 'C1:BSWV FRQ,%s' % (freq * 1000)
    freq2_command = 'C2:BSWV FRQ,%s' % (freq * 1000)

    device_resource = variables.COM_PORT_signal_generator
    wave_generator = resources.open_resource(device_resource)
    wave_generator.write(freq1_command)  # Set output frequency on channel 1
    time.sleep(0.01)
    wave_generator.write(freq2_command)  # Set output frequency on channel 2
    time.sleep(0.01)

    print(f"Frequency changed to {freq} kHz")


def turn_off_signal_generator():
    """
    Turn off the signal generator.

    Returns:
            None
    """
    resources = pyvisa.ResourceManager()

    device_resource = "USB0::0xF4EC::0x1101::SDG6XBAD2R0601::INSTR"

    wave_generator = resources.open_resource(device_resource)

    wave_generator.write('C2:OUTP OFF')  # Turn off channel 2
    time.sleep(0.01)
    wave_generator.write('C1:OUTP OFF')  # Turn off channel 1
    time.sleep(0.01)
