#  Python wrapper for surface Concept TDC

This folder contains dlls and python wrapper for Surface Concept tdc.

## Voltage, laser energy and detection rate per event

* The SDK callback (`BufDataCB4.on_data`) stamps every packet with its arrival time and the DC voltage,
  pulse voltage and laser energy of that moment.
* The acquisition loop takes packets from a queue, possibly seconds later (slow disk, plotting, a burst).
  It labels the events with the stamp, not with the value when the packet is taken.
* The detection rate for the voltage controller uses the arrival times, so a drained queue backlog is
  not a rate spike.
* If the metadata can't be read in the callback, the loop reads it itself. If that also fails, the packet
  is kept under `temp_data/unattributed`.
