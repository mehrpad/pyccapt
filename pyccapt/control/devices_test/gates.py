import nidaqmx
import time

if __name__ == '__main__':
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan('Dev2/port0/line1')

        task.start()
        task.write([False])
        time.sleep(0.5)

# Uncomment and modify the lines below as needed
# task.write([False, False, False, False, False, False])
# time.sleep(5)
# task.write([False, True, False, False, False, False])
# time.sleep(0.5)
# task.write([False, False, False, False, False, False])
# task.write(True)
# time.sleep(1)
# task.write(False)
# time.sleep(1)
# task.write([0, 0, 8, 0])
# time.sleep(0.002)
# task.write([0, 0, 0, 8])
# time.sleep(0.002)
