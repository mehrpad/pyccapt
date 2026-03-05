Control Configuration
=====================

PyCCAPT control runtime is configured with a TOML file:

- default path: ``pyccapt/config.toml``
- supported format: ``.toml`` only
- JSON configuration is no longer supported

Runtime loading follows this order:

1. Explicit config path (if provided by the caller)
2. ``config.toml`` in the detected project root

Electrode List Configuration
----------------------------

The control GUI electrode dropdown is loaded from:

- ``pyccapt/control/electrode.toml``

Example:

.. code-block:: toml

   [electrodes]
   names = [
       "NiC1", # Nickel capillary
       "CuC1",
       "NC",   # Not categorized
   ]

TOML supports comments, so lab-specific notes can be kept directly in this file.

Configuration Groups
--------------------

The control configuration file is organized into a few logical groups.

Visualization and Calibration Constants
---------------------------------------

These settings define plotting mode and conversion constants used during control
and visualization (for example, flight-path length and timing offsets).

Examples:

- ``visualization = "mc"``
- ``flight_path_length``
- ``t_0_voltage``
- ``t_0_laser``
- ``max_mass``
- ``max_tof``

Safety Limits and Control Bounds
--------------------------------

These values define operating ranges and guardrails for voltages, pulse behavior,
laser power, and cryogenic temperatures.

Examples:

- ``max_vdc``
- ``min_vp`` / ``max_vp``
- ``pulse_fraction_max``
- ``max_laser_power``
- ``max_temperature_cryo`` / ``min_temperature_cryo``

Device Enable Flags
-------------------

Hardware blocks can be enabled or disabled using ``"on"`` / ``"off"`` flags.

Examples:

- ``tdc``, ``camera``, ``gates``, ``laser``, ``stage``
- ``pump_ll``, ``pump_cll``, ``gauges``, ``cryo``
- ``v_dc``, ``v_p``, ``signal_generator``, ``DAQ``

Connection Identifiers
----------------------

Connection endpoints are specified using ``COM_PORT_*`` keys.
Depending on the device, values can be serial COM ports, NI-DAQ paths, VISA
resource strings, or hardware IDs.

Examples:

- ``COM_PORT_laser = "COM9"``
- ``COM_PORT_gates = "Dev2/port0/"``
- ``COM_PORT_signal_generator = "USB0::...::INSTR"``

Best Practices
--------------

- Start from ``pyccapt/config.toml`` and keep it under version control.
- Add comments directly in the TOML file to document local lab settings.
- Validate hardware port values before experiment runs.
- Keep conservative bounds for safety-critical settings.
