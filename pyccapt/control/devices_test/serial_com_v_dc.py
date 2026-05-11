import time

import serial.tools.list_ports

# Get available COM ports and store as a list
com_ports = list(serial.tools.list_ports.comports())
print(com_ports)

# Get the number of available COM ports
no_com_ports = len(com_ports)

if __name__ == '__main__':
    # Print out user information
    if no_com_ports > 0:
        print("Total number of available COM ports: " + str(no_com_ports))

        # Show all available COM ports
        for idx, curr in enumerate(com_ports):
            print("  " + str(idx) + ".)  " + curr.description)

        # User chooses COM port to connect to
        while True:
            try:
                com_port_idx = int(input("Enter the number of COM port to connect to (0 - " + str(no_com_ports - 1) + "): "))
            except:
                print("Incorrect value for COM port! Enter a Number (0 - " + str(no_com_ports - 1) + ")")
                continue
            else:
                if 0 <= com_port_idx < no_com_ports:
                    break

        # Configure the COM port to talk to. Default values: 115200, 8, N, 1
        com_port = serial.Serial(
            port=com_ports[com_port_idx].device,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )

        if com_port.is_open:
            com_port.flushInput()
            com_port.flushOutput()
            print("Opened Port: " + com_ports[com_port_idx].device)

            # Define a command function
            def command(cmd):
                com_port.write((cmd + '\r\n').encode())
                time.sleep(0.001)
                response = ''

                while com_port.in_waiting > 0:
                    response = com_port.readline()

                if response:
                    print("<<: " + response.decode("utf-8"))
                else:
                    print("<< Error, no Response!")

            # List of commands to send
            cmd_list = [">S1 3.0e-4"]
            for cmd in cmd_list:
                command(cmd)

            for i in range(20):
                time.sleep(2)
                command(">S0A?")
                command(">S1A?")
            command('F0')

        else:
            print("Couldn't open Port!")
            exit()
        com_port.close()
    else:
        print("No COM ports available!")
        exit()
