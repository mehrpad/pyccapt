import time
import nidaqmx
from nidaqmx.constants import LineGrouping

CHANNEL = "Dev3/port1/line7"   # USB-6501 P1.7 = physical pin 19
STEP_SECONDS = 5
CYCLES = 10

with nidaqmx.Task() as task:
    task.do_channels.add_do_chan(
        CHANNEL,
        line_grouping=LineGrouping.CHAN_PER_LINE
    )

    print(f"Testing {CHANNEL}")
    print("Measure between physical pin 19 and NI GND, for example pin 17 or pin 24.")

    for i in range(CYCLES):
        print(f"Cycle {i+1}/{CYCLES}: HIGH")
        task.write(True)
        time.sleep(STEP_SECONDS)

        print(f"Cycle {i+1}/{CYCLES}: LOW")
        task.write(False)
        time.sleep(STEP_SECONDS)

    print("Leaving P1.7 LOW before closing task.")
    task.write(False)
    time.sleep(1)

print("Task closed.")