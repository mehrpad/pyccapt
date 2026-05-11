import time
import timeit
from queue import Queue

import numpy as np

from pyccapt.control.tdc_surface_concept import scTDC

NR_OF_MEASUREMENTS = 10
EXPOSURE_MS = 1000

QUEUE_DATA = 0
QUEUE_ENDOFMEAS = 1


class BufDataCB4(scTDC.buffered_data_callbacks_pipe):
    def __init__(self, lib, dev_desc, data_field_selection, dld_events, max_buffered_data_len=500000):
        super().__init__(lib, dev_desc, data_field_selection, max_buffered_data_len, dld_events)
        self.queue = Queue()
        self.end_of_meas = False

    def on_data(self, d):
        dcopy = {}
        for k in d.keys():
            if isinstance(d[k], np.ndarray):
                dcopy[k] = d[k].copy()
            else:
                dcopy[k] = d[k]
        self.queue.put((QUEUE_DATA, dcopy))
        if self.end_of_meas:
            self.end_of_meas = False
            self.queue.put((QUEUE_ENDOFMEAS, None))

    def on_end_of_meas(self):
        self.end_of_meas = True
        return True


def check4():
    device = scTDC.Device(autoinit=False)

    retcode, errmsg = device.initialize()
    if retcode < 0:
        print("error during init:", retcode, errmsg)
        return -1
    else:
        print("successfully initialized")

    DATA_FIELD_SEL = (
        scTDC.SC_DATA_FIELD_DIF1 | scTDC.SC_DATA_FIELD_DIF2 | scTDC.SC_DATA_FIELD_TIME | scTDC.SC_DATA_FIELD_START_COUNTER
    )
    DATA_FIELD_SEL_raw = scTDC.SC_DATA_FIELD_TIME | scTDC.SC_DATA_FIELD_CHANNEL | scTDC.SC_DATA_FIELD_START_COUNTER
    bufdatacb = BufDataCB4(device.lib, device.dev_desc, DATA_FIELD_SEL, dld_events=True)
    bufdatacb_raw = BufDataCB4(device.lib, device.dev_desc, DATA_FIELD_SEL_raw, dld_events=False)

    def errorcheck(retcode):
        if retcode < 0:
            print(device.lib.sc_get_err_msg(retcode))
            bufdatacb.close()
            device.deinitialize()
            return -1
        else:
            return 0

    start = timeit.default_timer()

    retcode = bufdatacb.start_measurement(EXPOSURE_MS)
    if errorcheck(retcode) < 0:
        return -1

    meas_remaining = NR_OF_MEASUREMENTS
    while True:
        eventtype, data = bufdatacb.queue.get()
        eventtype_raw, data_raw = bufdatacb_raw.queue.get()
        if eventtype == QUEUE_DATA:
            print(len(data["start_counter"]))
            a = np.array((data["start_counter"], data["dif1"], data["dif2"], data["time"]))
            b = np.array((data_raw["channel"], data_raw["start_counter"], data_raw["time"]))
        elif eventtype == QUEUE_ENDOFMEAS:
            print('Length data', len(a[0]))
            print(a)
            print('--------------------------')
            print('Length raw data', len(b[0]))
            print(b)
            print('==========================')
            print(meas_remaining)
            meas_remaining -= 1
            if meas_remaining > 0:
                retcode = bufdatacb.start_measurement(EXPOSURE_MS)
                if errorcheck(retcode) < 0:
                    return -1
            else:
                break
        else:
            break

    end = timeit.default_timer()
    print("\ntime elapsed:", end - start, "s")

    time.sleep(0.1)
    bufdatacb.close()
    device.deinitialize()

    return 0


if __name__ == "__main__":
    check4()
