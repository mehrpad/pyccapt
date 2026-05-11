import time

import serial.tools.list_ports


def main():
    # Get available COM ports and store as a list
    com_ports = list(serial.tools.list_ports.comports())
    no_com_ports = len(com_ports)

    # Print out user information
    if no_com_ports > 0:
        print("Total number of available COM ports:", no_com_ports)

        # Show all available COM ports
        for idx, curr in enumerate(com_ports):
            print("  " + str(idx) + ".)  " + curr.description)

        # User chooses COM port to connect to
        while True:
            try:
                com_port_idx = int(
                    input("Enter the number of the COM port to connect to (0 - " + str(no_com_ports - 1) + "): ")
                )
            except ValueError:
                print("Incorrect value for COM port! Enter a number (0 - " + str(no_com_ports - 1) + ")")
                continue
            else:
                if not (0 <= com_port_idx < no_com_ports):
                    continue
                break

        # Configure the COM port to communicate with
        chosen_com_port = com_ports[com_port_idx].device
        com_port = serial.Serial(
            port=chosen_com_port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )

        if com_port.is_open:
            com_port.flushInput()
            com_port.flushOutput()
            print("Opened Port:", chosen_com_port)

            # Loop until the entered cmd is "exit"
            while True:
                cmd = input(">>: ")  # Get the cmd to send

                if cmd == 'exit':
                    break
                else:
                    com_port.write((cmd + '\r\n').encode())
                    time.sleep(0.001)

                response = ''

                while com_port.in_waiting > 0:
                    response = com_port.readline()

                if response:
                    print("<<:", response.decode("utf-8"))
                else:
                    print("<< Error, no Response!")

        else:
            print("Couldn't open Port!")
            exit()

        com_port.close()
    else:
        print("No COM ports available!")
        exit()


if __name__ == '__main__':
    main()
