"""Shared process-safe variables used by the control module.

Variable ownership
------------------
Every entry in :attr:`Variables._OWNERSHIP` lists *who writes* and *who
reads* it.  This is the authoritative map - if you add a new shared
field, document it there too.  Variables not listed are flagged as
``("?",  ("?",))`` so they're easy to find with ``grep`` and audit.

Process keys (single letters keep the table narrow):
  main   - main GUI process (also hosts the Gates, Pumps&Vacuum,
           Baking, Laser, Stage sub-windows that don't have their own
           subprocess)
  exp    - experiment subprocess (apt_exp_control.run_experiment)
  tdc    - TDC subprocess (tdc_surface_concept / tdc_roentdek /
           tdc_surface_concept_t_*)
  drs    - DRS digitiser subprocess
  cam    - camera subprocess (gui_cameras.run_camera_window)
  viz    - visualization subprocess (gui_visualization.run_visualization_window)
  pump   - pumps & vacuum reader thread (lives inside main)
  ?      - ownership not yet audited
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterable, Mapping
from typing import Any


class Variables:
    """Expose shared experiment state through a manager-backed namespace.

    The class keeps backward compatibility with existing `variables.<name>` access
    while removing thousands of lines of repetitive property code.

    See module docstring for the writer / reader ownership map
    (``_OWNERSHIP``).  Plot data (``x_plot`` / ``y_plot`` / ``t_plot`` /
    ``main_v_dc_plot``) is NOT on this namespace - it lives in
    :class:`pyccapt.control.core.shared_ring_buffer.SharedRingBuffer`
    instances passed explicitly as arguments to subprocess targets.
    """

    # ------------------------------------------------------------------
    # Writer / reader map.  ("writer", ("reader1", "reader2", ...)).
    # Processes that only *snapshot* a value (e.g. for logging) are not
    # listed as readers; only data-flow consumers are.
    # ------------------------------------------------------------------
    _OWNERSHIP = {
        # --- Setup parameters (GUI inputs the experiment loop reads) -----
        "ex_time": ("main", ("exp",)),
        "max_ions": ("main", ("exp",)),
        "ex_freq": ("main", ("exp",)),
        "user_name": ("main", ("exp",)),
        "ex_name": ("main", ("exp",)),
        "exp_name": ("main", ("exp",)),
        "electrode": ("main", ("exp",)),
        "email": ("main", ("exp",)),
        "vdc_min": ("main", ("exp",)),
        "vdc_max": ("main", ("exp",)),
        "vdc_step_up": ("main", ("exp",)),
        "vdc_step_down": ("main", ("exp",)),
        "v_p_min": ("main", ("exp",)),
        "v_p_max": ("main", ("exp",)),
        "pulse_fraction": ("main", ("exp",)),
        "pulse_frequency": ("main", ("exp",)),
        "pulse_mode": ("main", ("exp",)),
        "control_algorithm": ("main", ("exp",)),
        "criteria_time": ("main", ("exp",)),
        "criteria_ions": ("main", ("exp",)),
        "criteria_vdc": ("main", ("exp",)),
        "criteria_laser": ("main", ("exp",)),
        "detection_rate": ("main", ("exp",)),
        "hit_display": ("main", ("viz",)),
        "fixed_laser": ("main", ("exp",)),
        "laser_num_ions_per_step": ("main", ("exp",)),
        "laser_increase_per_step": ("main", ("exp",)),
        "laser_start": ("main", ("exp",)),
        "laser_stop": ("main", ("exp",)),
        "counter_source": ("main", ("exp",)),
        "access_override_enabled": ("main", ("exp",)),
        "override_disabled_devices": ("main", ("exp",)),
        # --- Live experiment statistics (exp writes, GUI/viz read) -------
        "elapsed_time": ("exp", ("main",)),
        "total_ions": ("exp", ("main", "viz")),
        "total_raw_signals": ("exp", ("main",)),
        "specimen_voltage": ("exp", ("main", "viz")),
        "pulse_voltage": ("exp", ("main", "viz")),
        "detection_rate_current": ("exp", ("main", "viz")),
        "specimen_voltage_plot": ("exp", ("viz",)),
        "detection_rate_current_plot": ("exp", ("viz",)),
        "count": ("exp", ("main",)),
        "count_last": ("exp", ()),
        "count_temp": ("exp", ()),
        "avg_n_count": ("exp", ()),
        "counter": ("exp", ("main",)),
        "start_time": ("main", ("exp", "viz")),
        "end_time": ("exp", ("main",)),
        # --- Lifecycle / flow-control flags ------------------------------
        "start_flag": ("main", ("exp",)),
        "stop_flag": ("main", ("exp", "tdc")),
        "end_experiment": ("exp", ("main",)),
        "flag_end_experiment": ("exp", ("main",)),
        "flag_visualization_start": ("exp", ("viz",)),
        "flag_pumps_vacuum_start": ("main", ("pump",)),
        "flag_finished_tdc": ("tdc", ("exp",)),
        "flag_stop_tdc": ("exp", ("tdc",)),
        "flag_tdc_failure": ("tdc", ("exp", "main")),
        # --- Visualization controls / clear handshakes -------------------
        "vdc_hold": ("viz", ("exp",)),
        "reset_heatmap": ("viz", ("viz",)),  # internal to viz
        "plot_clear_flag": ("main", ("viz",)),
        "flag_visualization_win_show": ("main", ("viz",)),  # being replaced by Queue
        "flag_new_min_voltage": ("main", ("exp",)),
        # --- Camera handshakes --------------------------------------------
        "camera_0_ExposureTime": ("main", ("cam",)),
        "camera_1_ExposureTime": ("main", ("cam",)),
        "flag_camera_grab": ("main", ("cam",)),
        "flag_camera_win_show": ("main", ("cam",)),  # being replaced by Queue
        "flag_cameras_take_screenshot": ("main", ("cam", "viz")),
        "last_screen_shot": ("main", ("cam", "viz")),
        "light": ("main", ("cam",)),
        "light_switch": ("main", ("cam",)),
        "alignment_window": ("main", ("cam",)),
        # --- Laser GUI fields (main process, no separate subprocess) ----
        "laser_pulse_energy": ("main", ("exp",)),
        "laser_power": ("main", ("exp",)),
        "laser_freq": ("main", ("exp",)),
        "laser_division_factor": ("main", ("exp",)),
        "laser_average_power": ("main", ("exp",)),
        # Set by the laser GUI on every CLI session open/close so the main
        # GUI status bar can show a "laser disconnected" warning, and so
        # the experiment subprocess can refuse to start in laser pulse
        # mode when the laser was never reached on CLI.
        "flag_laser_connected": ("main", ("exp",)),
        # --- Gates ---------------------------------------------------------
        "flag_main_gate": ("main", ("exp",)),
        "flag_load_gate": ("main", ("exp",)),
        "flag_cryo_gate": ("main", ("exp",)),
        # --- Pumps & vacuum (pump thread writes, GUIs read) --------------
        "temperature": ("pump", ("main", "exp")),
        "vacuum_main": ("pump", ("main", "exp", "viz")),
        "vacuum_buffer": ("pump", ("main",)),
        "vacuum_buffer_backing": ("pump", ("main",)),
        "vacuum_load_lock": ("pump", ("main",)),
        "vacuum_load_lock_backing": ("pump", ("main",)),
        "vacuum_cryo_load_lock": ("pump", ("main",)),
        "vacuum_cryo_load_lock_backing": ("pump", ("main",)),
        "set_temperature_cryo": ("main", ("pump",)),
        "set_temperature_ll": ("main", ("pump",)),
        "set_temperature_flag_cryo": ("main", ("pump",)),
        "set_temperature_flag_ll": ("main", ("pump",)),
        "flag_pump_load_lock": ("main", ("pump",)),
        "flag_pump_load_lock_click": ("main", ("pump",)),
        "flag_pump_load_lock_led": ("pump", ("main",)),
        "flag_pump_cryo_load_lock": ("main", ("pump",)),
        "flag_pump_cryo_load_lock_click": ("main", ("pump",)),
        "flag_pump_cryo_load_lock_led": ("pump", ("main",)),
        # --- Path / metadata fields --------------------------------------
        "path": ("exp", ("exp", "viz", "main")),
        "path_meta": ("exp", ("exp", "viz", "main")),
        "log_path": ("main", ("exp",)),
        "hdf5_path": ("exp", ("main",)),
        "hdf5_data_name": ("exp", ("main",)),
        "data": ("exp", ("exp",)),  # catch-all output dict
        # --- Counters used by viz / exp (incremented in place) -----------
        "index_save_image": ("viz", ("viz",)),
        "index_plot": ("viz", ("viz",)),
        "index_plot_save": ("viz", ("viz",)),
        "index_wait_on_plot_start": ("viz", ("viz",)),
        "clear_index_save_image": ("main", ("viz",)),
        "index_warning_message": ("main", ("main",)),
        "index_line": ("main", ("main",)),
        "number_of_experiment_in_text_line": ("main", ("main",)),
        "index_experiment_in_text_line": ("main", ("main",)),
        # --- TDC list-typed fields (TDC writes per-event, exp/viz drain) -
        # Where readership is unclear the entry is "?" - please audit
        # before adding new dependencies.
        "main_counter": ("exp", ("exp",)),
        "main_raw_counter": ("exp", ("exp",)),
        "main_temperature": ("exp", ("exp",)),
        "main_chamber_vacuum": ("exp", ("exp",)),
        "laser_degree": ("?", ("?",)),
        "laser_intensity": ("exp", ("exp",)),
        "x": ("tdc", ("exp", "viz")),
        "y": ("tdc", ("exp", "viz")),
        "t": ("tdc", ("exp", "viz")),
        "dld_start_counter": ("tdc", ("exp",)),
        "time_stamp": ("tdc", ("exp",)),
        "main_v_dc_dld": ("tdc", ("exp",)),
        "main_v_p_dld": ("tdc", ("exp",)),
        "main_l_p_dld": ("tdc", ("exp",)),
        "main_v_dc_tdc": ("tdc", ("exp",)),
        "main_v_p_tdc": ("tdc", ("exp",)),
        "main_l_p_tdc": ("tdc", ("exp",)),
        "main_v_dc_hsd": ("tdc", ("exp",)),
        "main_v_p_hsd": ("tdc", ("exp",)),
        "main_l_p_hsd": ("tdc", ("exp",)),
        "main_v_dc_drs": ("drs", ("exp",)),
        "main_v_p_drs": ("drs", ("exp",)),
        "main_l_p_drs": ("drs", ("exp",)),
        "main_v_p": ("?", ("?",)),
        "main_p_tdc_roentdek": ("tdc", ("exp",)),
        "channel": ("tdc", ("viz",)),
        "time_data": ("tdc", ("viz",)),
        "tdc_start_counter": ("tdc", ("viz",)),
        "ch0_time": ("tdc", ("viz",)),
        "ch0_wave": ("tdc", ("viz",)),
        "ch1_time": ("tdc", ("viz",)),
        "ch1_wave": ("tdc", ("viz",)),
        "ch2_time": ("tdc", ("viz",)),
        "ch2_wave": ("tdc", ("viz",)),
        "ch3_time": ("tdc", ("viz",)),
        "ch3_wave": ("tdc", ("viz",)),
        "ch4_time": ("tdc", ("viz",)),
        "ch4_wave": ("tdc", ("viz",)),
        "ch5_time": ("tdc", ("viz",)),
        "ch5_wave": ("tdc", ("viz",)),
        "ch0": ("tdc", ("viz",)),
        "ch1": ("tdc", ("viz",)),
        "ch2": ("tdc", ("viz",)),
        "ch3": ("tdc", ("viz",)),
        "ch4": ("tdc", ("viz",)),
        "ch5": ("tdc", ("viz",)),
        "ch6": ("tdc", ("viz",)),
        "ch7": ("tdc", ("viz",)),
    }

    _REQUIRED_CONFIG_KEYS = (
        "COM_PORT_cryo",
        "COM_PORT_V_dc",
        "COM_PORT_V_p",
        "COM_PORT_gauge_mc",
        "COM_PORT_gauge_bc",
        "COM_PORT_gauge_ll",
        "COM_PORT_gauge_cll",
        "COM_PORT_signal_generator",
        "COM_PORT_thorlab_motor",
        "save_meta_interval_camera",
        "save_meta_interval_visualization",
        "pulse_amp_per_supply_voltage",
        "max_laser_power",
    )

    _ALIASES = {
        "flag_cryo_pump_load_lock_led": "flag_pump_cryo_load_lock_led",
        "vdc_steps_up": "vdc_step_up",
        "vdc_steps_down": "vdc_step_down",
        "vp_min": "v_p_min",
        "vp_max": "v_p_max",
        "ex_user": "user_name",
        "detection_rate_init": "detection_rate",
        "hit_displayed": "hit_display",
    }

    _LIST_FIELDS = {
        "main_counter",
        "main_raw_counter",
        "main_temperature",
        "main_chamber_vacuum",
        "laser_degree",
        "x",
        "y",
        "t",
        "dld_start_counter",
        "time_stamp",
        "laser_intensity",
        "main_v_dc_dld",
        "main_v_p_dld",
        "main_l_p_dld",
        "main_v_dc_tdc",
        "main_v_p_tdc",
        "main_l_p_tdc",
        "main_v_dc_hsd",
        "main_v_p_hsd",
        "main_l_p_hsd",
        "main_v_dc_drs",
        "main_v_p_drs",
        "main_l_p_drs",
        "main_v_p",
        # (x_plot, y_plot, t_plot, main_v_dc_plot were here as Manager
        # lists - replaced by SharedRingBuffer instances passed as
        # explicit args; see runtime.create_shared_context.)
        "main_p_tdc_roentdek",
        "override_disabled_devices",
        "channel",
        "time_data",
        "tdc_start_counter",
        "ch0_time",
        "ch0_wave",
        "ch1_time",
        "ch1_wave",
        "ch2_time",
        "ch2_wave",
        "ch3_time",
        "ch3_wave",
        "ch4_time",
        "ch4_wave",
        "ch5_time",
        "ch5_wave",
        "ch0",
        "ch1",
        "ch2",
        "ch3",
        "ch4",
        "ch5",
        "ch6",
        "ch7",
    }

    _EXP_FIELDS = {
        "elapsed_time",
        "total_ions",
        "total_raw_signals",
        "specimen_voltage",
        "detection_rate_current",
        "pulse_voltage",
    }

    _DATA_PLOT_FIELDS = {
        "specimen_voltage_plot",
        "detection_rate_current_plot",
    }

    _VACUUM_FIELDS = {
        "temperature",
        "set_temperature_cryo",
        "set_temperature_ll",
        "set_temperature_flag_cryo",
        "set_temperature_flag_ll",
        "vacuum_main",
        "vacuum_buffer",
        "vacuum_buffer_backing",
        "vacuum_load_lock",
        "vacuum_load_lock_backing",
        "vacuum_cryo_load_lock",
        "vacuum_cryo_load_lock_backing",
    }

    _DEFAULTS = {
        "counter_source": "pulse_counter",
        "counter": 0,
        "count": 0,
        "ex_time": 0,
        "max_ions": 0,
        "ex_freq": 0,
        "user_name": "",
        "electrode": "",
        "ex_name": "",
        "hdf5_data_name": "",
        "vdc_min": 0,
        "vdc_max": 0,
        "vdc_step_up": 0,
        "vdc_step_down": 0,
        "v_p_min": 0,
        "v_p_max": 0,
        "pulse_fraction": 0,
        "pulse_frequency": 0,
        "hdf5_path": "",
        "flag_main_gate": False,
        "flag_load_gate": False,
        "flag_cryo_gate": False,
        "email": "",
        "light": False,
        "alignment_window": False,
        "light_switch": False,
        "vdc_hold": False,
        "reset_heatmap": False,
        "last_screen_shot": False,
        "camera_0_ExposureTime": 2000,
        "camera_1_ExposureTime": 2000,
        "path": "",
        "path_meta": "",
        "index_save_image": 0,
        "index_plot": 0,
        "index_wait_on_plot_start": 0,
        "index_plot_save": 0,
        "flag_pump_load_lock": True,
        "flag_pump_load_lock_click": False,
        "flag_pump_load_lock_led": None,
        "flag_pump_cryo_load_lock": True,
        "flag_pump_cryo_load_lock_click": False,
        "flag_pump_cryo_load_lock_led": None,
        "flag_camera_grab": False,
        "flag_camera_win_show": False,
        "flag_visualization_win_show": False,
        "flag_end_experiment": False,
        "flag_new_min_voltage": False,
        "flag_visualization_start": False,
        "flag_pumps_vacuum_start": False,
        "criteria_time": True,
        "criteria_ions": True,
        "criteria_vdc": True,
        "criteria_laser": True,
        "exp_name": "",
        "log_path": "",
        "fixed_laser": 0,
        "laser_num_ions_per_step": 0,
        "laser_increase_per_step": 0,
        "laser_start": 0,
        "laser_stop": 0,
        "elapsed_time": 0.0,
        "start_time": "",
        "end_time": "",
        "total_ions": 0,
        "total_raw_signals": 0,
        "specimen_voltage": 0.0,
        "specimen_voltage_plot": 0.0,
        "detection_rate": 0.0,
        "detection_rate_current": 0.0,
        "detection_rate_current_plot": 0.0,
        "pulse_voltage": 0.0,
        "control_algorithm": "",
        "pulse_mode": "",
        "count_last": 0,
        "count_temp": 0,
        "avg_n_count": 0,
        "index_warning_message": 0,
        "index_line": 0,
        "stop_flag": False,
        "end_experiment": False,
        "start_flag": False,
        "flag_stop_tdc": False,
        "flag_finished_tdc": False,
        "flag_tdc_failure": False,
        "plot_clear_flag": False,
        "clear_index_save_image": False,
        "number_of_experiment_in_text_line": 0,
        "index_experiment_in_text_line": 0,
        "flag_cameras_take_screenshot": False,
        "access_override_enabled": False,
        "temperature": 0,
        "set_temperature_cryo": 0,
        "set_temperature_ll": 0,
        "set_temperature_flag_cryo": None,
        "set_temperature_flag_ll": None,
        "vacuum_main": 0,
        "vacuum_buffer": 0,
        "vacuum_buffer_backing": 0,
        "vacuum_load_lock": 0,
        "vacuum_load_lock_backing": 0,
        "vacuum_cryo_load_lock": 0,
        "vacuum_cryo_load_lock_backing": 0,
        "laser_pulse_energy": 0,
        "laser_power": 0,
        "laser_freq": 0,
        "laser_division_factor": 0,
        "laser_average_power": 0,
        # Set True by the laser GUI when a CLI session is open and
        # responsive; False (the default) means the laser is either not
        # configured, the COM port is unavailable, or the laser is
        # currently in NKTPBus mode. The main GUI reads this flag to
        # show a status-bar warning.
        "flag_laser_connected": False,
        "hit_display": 0,
        "data": {},
    }

    _INTERNAL_ATTRS = {
        "ns",
        "lock",
        "lock_lists",
        "lock_data_plot",
        "lock_exp",
        "lock_vacuum_tmp",
        "lock_data",
        "lock_setup_parameters",
        "lock_statistics",
        "lock_experiment_variables",
        "_known_fields",
    }

    def __init__(self, conf: Mapping[str, Any], namespace: Any) -> None:
        if not isinstance(conf, Mapping):
            raise TypeError("`conf` must be a mapping with configuration values.")

        missing = [key for key in self._REQUIRED_CONFIG_KEYS if key not in conf]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise KeyError(f"Configuration is missing required keys: {missing_text}")

        object.__setattr__(self, "ns", namespace)
        object.__setattr__(self, "lock", multiprocessing.Lock())
        object.__setattr__(self, "lock_lists", multiprocessing.Lock())
        object.__setattr__(self, "lock_data_plot", multiprocessing.Lock())
        object.__setattr__(self, "lock_exp", multiprocessing.Lock())
        object.__setattr__(self, "lock_vacuum_tmp", multiprocessing.Lock())

        # Backward-compatible lock aliases used in older code/comments.
        object.__setattr__(self, "lock_data", self.lock_lists)
        object.__setattr__(self, "lock_setup_parameters", self.lock)
        object.__setattr__(self, "lock_statistics", self.lock_exp)
        object.__setattr__(self, "lock_experiment_variables", self.lock_lists)

        defaults = dict(self._DEFAULTS)
        defaults.update(
            {
                "COM_PORT_cryo": conf["COM_PORT_cryo"],
                "COM_PORT_V_dc": conf["COM_PORT_V_dc"],
                "COM_PORT_V_p": conf["COM_PORT_V_p"],
                "COM_PORT_gauge_mc": conf["COM_PORT_gauge_mc"],
                "COM_PORT_gauge_bc": conf["COM_PORT_gauge_bc"],
                "COM_PORT_gauge_ll": conf["COM_PORT_gauge_ll"],
                "COM_PORT_gauge_cll": conf["COM_PORT_gauge_cll"],
                "COM_PORT_signal_generator": conf["COM_PORT_signal_generator"],
                "COM_PORT_thorlab_motor": conf["COM_PORT_thorlab_motor"],
                "save_meta_interval_camera": conf["save_meta_interval_camera"],
                "save_meta_interval_visualization": conf["save_meta_interval_visualization"],
                "pulse_amp_per_supply_voltage": conf["pulse_amp_per_supply_voltage"],
                "max_laser_power": conf["max_laser_power"],
            }
        )

        list_defaults = {name: [] for name in self._LIST_FIELDS}
        defaults.update(list_defaults)

        for name, value in defaults.items():
            setattr(self.ns, name, value)

        object.__setattr__(self, "_known_fields", set(defaults))

    def _resolve_field_name(self, name: str) -> str:
        return self._ALIASES.get(name, name)

    def _lock_for_field(self, field: str):
        if field in self._LIST_FIELDS:
            return self.lock_lists
        if field in self._DATA_PLOT_FIELDS:
            return self.lock_data_plot
        if field in self._EXP_FIELDS:
            return self.lock_exp
        if field in self._VACUUM_FIELDS:
            return self.lock_vacuum_tmp
        return self.lock

    def __getattr__(self, name: str) -> Any:
        field = self._resolve_field_name(name)
        try:
            namespace = object.__getattribute__(self, "ns")
        except AttributeError as exc:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from exc

        # PERF: the namespace is a multiprocessing.Manager proxy, so every
        # attribute touch is an IPC round-trip to the Manager process. The
        # previous ``hasattr(...)`` then ``getattr(...)`` did TWO round-trips
        # per read; this is the single hottest path in the control app
        # (the TDC drain loop reads many flags per tick). Do a single
        # ``getattr`` under the lock and translate a missing field into the
        # same AttributeError -- one IPC instead of two, identical semantics.
        lock = self._lock_for_field(field)
        with lock:
            try:
                return getattr(namespace, field)
            except AttributeError:
                raise AttributeError(
                    f"{type(self).__name__!s} has no attribute {name!r}"
                )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._INTERNAL_ATTRS or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        field = self._resolve_field_name(name)
        lock = self._lock_for_field(field)
        with lock:
            setattr(self.ns, field, value)
            self._known_fields.add(field)

    def extend_to(self, variable_name: str, value: Iterable[Any]) -> None:
        """Extend a shared list attribute with iterable values.

        NOTE: each call fetches the WHOLE list from the Manager, extends a
        local copy, and ships it all back -- O(current length) IPC per
        call. For multi-million-element acquisition arrays this makes the
        Manager process the throughput bottleneck. The bulk acquisition
        arrays should live in shared memory / on-disk npy chunks rather
        than the Manager namespace; only short scalars/status belong here.
        Tracked as a follow-up refactor (CC2).
        """
        field = self._resolve_field_name(variable_name)

        if isinstance(value, (str, bytes)):
            raise TypeError("value must be an iterable of elements, not a string.")
        if not isinstance(value, list):
            try:
                value = list(value)
            except TypeError as exc:
                raise TypeError("value must be iterable.") from exc

        with self.lock_lists:
            # Single getattr round-trip (was hasattr + getattr = two).
            try:
                current_value = getattr(self.ns, field)
            except AttributeError:
                raise ValueError(f"{variable_name!r} is not an attribute of the namespace.")
            if not isinstance(current_value, list):
                raise TypeError(f"{variable_name!r} is not a list.")
            current_value.extend(value)
            setattr(self.ns, field, current_value)

    def clear_to(self, variable_name: str) -> None:
        """Clear a shared list attribute by replacing it with an empty list."""
        field = self._resolve_field_name(variable_name)
        with self.lock_lists:
            # Single getattr round-trip (was hasattr + setattr).
            try:
                getattr(self.ns, field)
            except AttributeError:
                raise ValueError(f"{variable_name!r} is not an attribute of the namespace.")
            setattr(self.ns, field, [])

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow snapshot of all currently known shared fields."""
        return {name: getattr(self.ns, name) for name in sorted(self._known_fields)}
