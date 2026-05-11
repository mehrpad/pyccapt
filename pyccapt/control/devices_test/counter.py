try:
    import nidaqmx
except ImportError:
    print('Please install nidaqmx')

import time

if __name__ == '__main__':
    try:
        task_counter = nidaqmx.Task()
        task_counter.ci_channels.add_ci_count_edges_chan("Dev1/ctr0")

        # Reference the terminal you want to use for the counter here
        task_counter.ci_channels[0].ci_count_edges_term = "PFI0"
        task_counter.start()

        for _ in range(10):
            time.sleep(1)
            data = task_counter.read(number_of_samples_per_channel=1)
            print(data)

    except nidaqmx.errors.DaqError as e:
        # Error handling for DAQ tasks
        print("An exception occurred:", e)

    finally:
        if task_counter.is_task_done():
            task_counter.stop()
        task_counter.close()
