"""Centralised hover-tooltips for the PyCCAPT control GUIs.

How it works
------------
Each GUI window has a top-level dict here whose keys are the widget
attribute names on the ``Ui_*`` object (the same names used in the .py and
.ui files - for example ``stage_speed_x``, ``laser_home``, ``superuser``).
The values are the strings shown on hover.

A single helper, :func:`apply_tooltips`, walks the dict and calls
``setToolTip`` on every widget that exists.  Missing keys are silently
ignored so this module can be edited independently of the GUI source.

Style guide
-----------
* Keep each tooltip to 1-2 short sentences.
* Always answer two questions: *what does this control do?* and *how do I
  change it?* (e.g. "drag the slider", "edit X in config.toml", "click",
  "type a number").
* Mention the underlying config.toml key when one exists, so the user
  can find the setting outside the GUI.

Add or edit entries here - no GUI source changes are needed for the new
text to appear on the next launch.
"""

from __future__ import annotations

from typing import Mapping


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def apply_tooltips(ui_object, tooltips: Mapping[str, str]) -> None:
    """Set tooltips on every named widget that exists on ``ui_object``.

    ``ui_object`` is typically a ``Ui_*`` instance after its ``setupUi``
    has run.  Widgets that don't exist on the object (e.g. removed in a
    refactor) are silently skipped, so the tooltip dictionary is
    forward-compatible with GUI changes.
    """
    for attr_name, text in tooltips.items():
        widget = getattr(ui_object, attr_name, None)
        if widget is None:
            continue
        try:
            widget.setToolTip(text)
        except Exception:
            pass


# ===========================================================================
# Main control window  (pyccapt.control.gui.gui_main)
# ===========================================================================

MAIN_TOOLTIPS = {
    # --- Sub-window launcher buttons --------------------------------------
    "pumps_vaccum": "Open the combined Gates & Pumps window (gate valves, chamber pressures, pumping and temperatures).",
    "camears": "Open the Cameras & Alignment window (live side / top / angle camera streams, light, exposure).",
    "laser_control": "Open the Laser Control window (NKT laser settings, AOM, and the laser focusing stage).",
    "stage_control": "Open the Stage Control window (sample / specimen stage, SmarAct MCS2).",
    "visualization": "Open the live Visualization window (detector hitmap, mass spectrum, statistics).",
    "baking": "Open the Baking window (chamber bake-out logging).",
    # --- Experiment metadata ----------------------------------------------
    "ex_number": "Experiment number, used in the output filename.  Type a new integer to bump the counter.",
    "ex_user": "User name written into the metadata of every saved file.",
    "ex_name": "Free-text experiment name; becomes part of the saved filename.",
    "email": "Notification email address (if email reports are enabled).",
    "electrode": "Electrode configuration selector - picks an entry from electrode.toml.",
    # --- Stop conditions --------------------------------------------------
    "max_ions": "Stop the experiment after this many detected ions.  Type an integer.",
    "ex_time": "Stop the experiment after this many seconds, regardless of ion count.",
    "criteria_ions": "Enable/disable the 'stop at max ions' criterion.",
    "criteria_time": "Enable/disable the 'stop at max time' criterion.",
    "criteria_vdc": "Enable/disable the 'stop at max DC voltage' criterion.",
    # --- Voltage limits ---------------------------------------------------
    "vdc_max": "DC voltage upper limit (V) - the controller will "
    "never command above this.  Hard-capped by max_vdc "
    "in config.toml.",
    "vdc_min": "DC voltage starting / lower limit (V).",
    "speciemen_voltage": "Live read-back of the specimen DC voltage (V).",
    # --- Pulse parameters -------------------------------------------------
    "pulse_mode": "Pulse source: Voltage (HV pulser) or Laser (NKT/origami).",
    "pulse_fraction": "Voltage-pulse amplitude as a percentage of the DC "
    "voltage.  Capped at pulse_fraction_max in "
    "config.toml.",
    "pulse_frequency": "Pulse repetition frequency (kHz).  Range comes "
    "from min/max_voltage_pulse_frequency or "
    "min/max_laser_pulse_frequency in config.toml.",
    "pulse_voltage": "Live read-back of the pulse-supply voltage (V).",
    "vp_min": "Pulser lower voltage limit (V).",
    "vp_max": "Pulser upper voltage limit (V).  Hard-capped by max_vp in config.toml.",
    # --- Detection / control ---------------------------------------------
    "detection_rate": "Live read-back of the detection rate (%).",
    "detection_rate_init": "Target detection rate the controller tries to hold (%).",
    "control_algorithm": "Closed-loop control algorithm used to keep "
    "detection rate at target.  Switchable LIVE during "
    "an experiment.\n"
    "  Proportional - P + deadband + asymmetric "
    "up/down gains (vdc_step_up, vdc_step_down).  "
    "Default; safest.\n"
    "  Proportional aggressive - same as above but the "
    "upward step is multiplied by "
    "control_p_aggressive_up_factor in config.toml; "
    "down-step unchanged.\n"
    "  Adaptive P - proportional core whose gain "
    "auto-scales between control_adaptive_min_factor "
    "and _max_factor based on observed loop behaviour "
    "(grows when sluggish, shrinks when overshooting).\n"
    "  PID - simple_pid with control_pid_kp/ki/kd "
    "gains and ±control_pid_max_step_v cap.  Tunings "
    "in config.toml; expect to retune on hardware.",
    "ex_freq": "Control-loop refresh rate (Hz) for the feedback algorithm.",
    "vdc_steps_up": "K_p gain for upward DC steps (controller's proportional gain when raising voltage).",
    "vdc_steps_down": "K_p gain for downward DC steps (proportional gain when lowering voltage).",
    "counter_source": "Where ion counts are read from (TDC, DRS, Counter, etc.).",
    "parameters_source": "Where setup parameters are loaded from (file vs. live GUI values).",
    # --- Live experiment statistics --------------------------------------
    "elapsed_time": "Live elapsed seconds since the experiment started.",
    "total_ions": "Live cumulative detected-ion count.",
    # --- Run controls -----------------------------------------------------
    "start_button": "Start the experiment with the parameters above.  "
    "Disabled until device checks pass (or override is "
    "active).",
    "stop_button": "Stop the running experiment cleanly (closes data files, ramps voltages down).",
    "superuser": "Override Access - bypasses device-availability "
    "checks and other safety interlocks.  Click for a "
    "warning dialog; the button turns green while active.",
    # --- Status -----------------------------------------------------------
    "Error": "Status / error messages.  Red text indicates a problem; messages auto-hide after a few seconds.",
}

# ===========================================================================
# Stage Control window  (pyccapt.control.gui.gui_stage_control)
# ===========================================================================

STAGE_TOOLTIPS = {
    # --- Position display (3 LCDs per axis) -------------------------------
    "stage_x_mm": "X-axis position, millimetres digit.  Read-only - "
    "driven by the SmarAct controller.  Combined with the "
    "µm and nm columns: total X = mm.µm.nm.",
    "stage_x_um": "X-axis position, micrometres digit.",
    "stage_x_nm": "X-axis position, nanometres digit.  Updated every 500 ms.",
    "stage_y_mm": "Y-axis position, millimetres digit.",
    "stage_y_um": "Y-axis position, micrometres digit.",
    "stage_y_nm": "Y-axis position, nanometres digit.",
    "stage_z_mm": "Z-axis position, millimetres digit.",
    "stage_z_um": "Z-axis position, micrometres digit.",
    "stage_z_nm": "Z-axis position, nanometres digit.",
    # --- Speed presets ----------------------------------------------------
    "stage_speed_x": "Exact X-axis speed preset in mm/s. The default 0.004 mm/s moves 0.8 µm per 0.2-second jog interval.",
    "stage_speed_y": "Exact Y-axis speed preset in mm/s. Values come from stage_speed_table_mm_s in config.toml.",
    "stage_speed_z": "Exact Z-axis speed preset in mm/s. Values come from stage_speed_table_mm_s in config.toml.",
    "stage_speed_x_label": "X-axis distance moved during each jog interval at the selected speed.",
    "stage_speed_y_label": "Y-axis distance moved during each jog interval at the selected speed.",
    "stage_speed_z_label": "Z-axis distance moved during each jog interval at the selected speed.",
    # --- Direction buttons ------------------------------------------------
    "stage_up": "Hold to jog the Y axis in the positive direction.  Release to stop.",
    "stage_down": "Hold to jog the Y axis in the negative direction.  Release to stop.",
    "stage_left": "Hold to jog the X axis in the negative direction.  Release to stop.",
    "stage_right": "Hold to jog the X axis in the positive direction.  Release to stop.",
    "stage_forward": "Hold to jog forward along Z+; release to stop.",
    "stage_backward": "Hold to jog backward along Z−; release to stop.",
    # --- Home / Reference / Stop / Override -------------------------------
    "stage_home": "Move all three axes to the home position set in "
    "config.toml (stage_home_x_mm, stage_home_y_mm, "
    "stage_home_z_mm).  Edit those values to change what "
    "'home' means for your experiment.",
    "stage_reference": "Run the SmarAct reference search - moves every "
    "axis until it finds its physical reference mark "
    "and zeros position to absolute coordinates.  "
    "REQUIRED once after every power-on.  Disabled "
    "until Override Access is granted, because the "
    "stage moves on its own.",
    "stage_stop": "Immediately stop all axes (calls ctl.Stop on every channel).  Always clickable - your panic button.",
    "superuser": "Enable potentially dangerous controls (currently the "
    "Reference button).  Click to grant access (a "
    "confirmation dialog appears); the button turns green "
    "while active.  Click again to deactivate.",
    "Error": "Status / error messages from the stage controller.  Red "
    "text indicates a problem (no SDK, controller unreachable, "
    "etc.); the cause stays visible until the next button "
    "click resolves it.",
}

# ===========================================================================
# Laser Control window  (pyccapt.control.gui.gui_laser_control)
# ===========================================================================

LASER_TOOLTIPS = {
    # --- Top-row controls -------------------------------------------------
    "laser_wavelegnth": "Output wavelength.  IR is the fundamental, Green "
    "is frequency-doubled, DUV is frequency-quadrupled.  "
    "Cannot be changed while the laser is emitting.",
    "laser_power": "Average output power, milliwatts.  Capped by "
    "max_laser_power in config.toml.  Type a number "
    "or use the spin arrows; sent to the laser "
    "immediately.",
    "laser_rate": "Base pulse-repetition frequency (Hz).  Above "
    "100 kHz the per-pulse energy decreases linearly "
    "with rate.  Effective rate at the sample = "
    "rate / Division Factor.",
    "laser_divition_factor": "Pulse division factor (integer).  Effective "
    "rate = base rate / this value.  Use to "
    "drop from MHz down to a few kHz without "
    "changing the base oscillator.",
    # --- Mode buttons + LEDs ---------------------------------------------
    "laser_listen": "Put the laser into Listen mode (lowest activity, safe).  No emission, ready to receive commands.",
    "laser_standby": "Bring the laser to Standby - powered, warmed up, but not emitting.  Required before Laser On.",
    "laser_on": "Start laser emission.  Only works from Standby.  Wavelength becomes locked while On.",
    "laser_enable": "Enable / disable the AOM output gate.  "
    "Toggles the actual output at the sample "
    "without changing the laser's emission "
    "state.",
    "led_laser_listen": "Listen-mode indicator.  Green = active.",
    "led_laser_laser_standby": "Standby-mode indicator.  Green = active, orange = transitioning.",
    "led_laser_on": "Emission indicator.  Green = laser is emitting, orange = transitioning.",
    "led_laser_enable": "Output-enable indicator.  Green = AOM open.",
    # --- Live readouts ---------------------------------------------------
    "laser_power_disp": "Live measured average power (mW) read back from the laser.",
    "laser_pulse_energy_disp": "Live per-pulse energy (nJ) read back from the laser.",
    "laser_repetion_rate_disp": "Effective pulse rate at the sample (kHz), accounting for the division factor.",
    # --- Scan / Focus mode -----------------------------------------------
    "laser_scan_mode5": "Scanning pattern selector.  Currently only 'Standard' is implemented.",
    "laser_focus_mode": "Focus-mode selector.  Currently only 'Standard' is implemented.",
    "scanning_disp": "Visualisation of the active scanning pattern.",
    "start_scanning": "Start / stop the scanning routine using the selected scan and focus modes.",
    "nktpbus_mode_switch": "Switch the laser from CLI to NKTPBus mode - "
    "hands control over to NKT's own software.  "
    "Once switched, you must use the NKT control "
    "tool to bring it back to CLI mode.  Disabled "
    "until Override Access is granted.",
    # --- SmarAct laser focusing stage: position display ------------------
    "laser_x_mm": "Laser-stage X position, millimetres digit.  Read-only.  "
    "Combined with µm and nm columns: total X = mm.µm.nm.",
    "laser_x_um": "Laser-stage X position, micrometres digit.",
    "laser_x_nm": "Laser-stage X position, nanometres digit.  Updated every 500 ms.",
    "laser_y_mm": "Laser-stage Y position, millimetres digit.",
    "laser_y_um": "Laser-stage Y position, micrometres digit.",
    "laser_y_nm": "Laser-stage Y position, nanometres digit.",
    "laser_z_mm": "Laser-stage Z position, millimetres digit.",
    "laser_z_um": "Laser-stage Z position, micrometres digit.",
    "laser_z_nm": "Laser-stage Z position, nanometres digit.",
    # --- Laser-stage speed presets ---------------------------------------
    "laser_speed_x": "Exact laser-stage X speed preset in mm/s. The default 0.004 mm/s moves 0.8 µm per 0.2-second jog interval.",
    "laser_speed_y": "Exact laser-stage Y speed preset in mm/s. Values come from stage_speed_table_mm_s in config.toml.",
    "laser_speed_z": "Exact laser-stage Z speed preset in mm/s. Values come from stage_speed_table_mm_s in config.toml.",
    "laser_speed_x_label": "Laser-stage X distance moved during each jog interval at the selected speed.",
    "laser_speed_y_label": "Laser-stage Y distance moved during each jog interval at the selected speed.",
    "laser_speed_z_label": "Laser-stage Z distance moved during each jog interval at the selected speed.",
    # --- Laser-stage direction buttons -----------------------------------
    "laser_up": "Hold to jog the laser stage along Y+; release to stop.",
    "laser_down": "Hold to jog the laser stage along Y−; release to stop.",
    "laser_left": "Hold to jog the laser stage along X−; release to stop.",
    "leser_right": "Hold to jog the laser stage along X+; release to stop.",
    "laser_forward": "Hold to jog the laser stage forward along Z+; release to stop.",
    "laser_backward": "Hold to jog the laser stage backward along Z−; release to stop.",
    # --- Home / Reference / Stop / Override ------------------------------
    "laser_home": "Move the laser stage to the home position set in config.toml (laser_stage_home_x_mm, _y_mm, _z_mm).",
    "laser_stage_reference": "Run the SmarAct reference search on the "
    "laser stage.  REQUIRED once after every "
    "power-on.  Disabled until Override Access "
    "is granted.",
    "laser_stage_stop": "Immediately stop all laser-stage axes.  Always clickable.",
    "laser_stage_superuser": "Enable potentially disruptive controls: "
    "the stage Reference button and the Nktpbus "
    "mode switch.  Click to grant access (a "
    "confirmation dialog appears); the button "
    "turns green while active.",
    "Error": "Status / error messages from both the laser and the laser "
    "stage.  Red text indicates a problem; messages auto-hide "
    "after 8 seconds.",
}

# ===========================================================================
# Gates window  (pyccapt.control.gui.gui_gates)
# ===========================================================================

GATES_TOOLTIPS = {
    "diagram": "Live chamber state: pale blue = vented. Gate symbols follow the pipe direction; green = open flow, red = closed barrier.",
    "main_chamber_switch": "Open / close the main-chamber gate valve.  "
    "Interlocked: cannot open if vacuum levels "
    "are wrong.  Bypass via Override Access.",
    "load_lock_switch": "Open / close the load-lock gate valve.",
    "cryo_switch": "Open / close the cryo gate valve.",
    "superuser": "Override Access - bypass the gate-vacuum "
    "interlocks.  Click for a warning dialog; "
    "button turns green while active.  USE WITH "
    "CARE: opening a gate against the wrong "
    "vacuum can damage hardware.",
    "Error": "Status / error messages from the gate controller.",
}

# ===========================================================================
# Pumps & Vacuum window  (pyccapt.control.gui.gui_pumps_vacuum)
# ===========================================================================

PUMPS_TOOLTIPS = {
    # --- Live vacuum readings (mBar) -------------------------------------
    "vacuum_main": "Main chamber pressure (mBar).  Live read from the gauge; updated continuously.",
    "vacuum_buffer": "Buffer chamber pressure (mBar).",
    "vacuum_buffer_back": "Buffer chamber backing-line pressure (mBar).",
    "vacuum_load_lock": "Load-lock chamber pressure (mBar).",
    "vacuum_load_lock_back": "Load-lock backing-line pressure (mBar).",
    "vacuum_cryo_load_lock": "Cryo load-lock chamber pressure (mBar).",
    "vacuum_cryo_load_lock_back": "Cryo load-lock backing-line pressure (mBar).",
    # --- Pump switches ---------------------------------------------------
    "pump_load_lock_switch": "Vent / pump the load lock.  Click to toggle; green means venting is active.  "
                              "Interlocked behind Override Access.",
    "pump_cryo_load_lock_switch": "Fully vent / pump the cryo load lock.  Interlocked behind Override Access.  "
                                  "Green means fully vented.  Cryo head vacuum depends on the CLL backing pump - "
                                  "check everything before venting.",
    "vent_cryo_load_lock_partial_switch": "Partially vent the cryo load lock for fast sample/cryo exchange "
                                          "(drives a 3-valve sequence).  Blocked during an experiment or with a "
                                          "gate open unless Override Access is active.  Green means active.",
    # --- Temperatures ---------------------------------------------------
    "temp_cryo_head": "Cryo head temperature (K) - live reading.",
    "temp_cryo_head_inside": "Cryo head inside temperature (K).",
    "temp_stage": "Stage temperature (K).",
    "temp_ll": "Load-lock temperature (°C).",
    "set_temperature_cryo": "Set the cryo target temperature (K).  Range: min/max_temperature_cryo in config.toml.",
    "set_temperature_ll": "Set the load-lock target temperature (°C).  Range: min/max_temperature_ll in config.toml.",
    "target_tempreature_cryo": "Cryo target temperature (K).  Type a number then click Set.",
    "target_tempreature_ll": "Load-lock target temperature (°C).",
    # --- Baking ---------------------------------------------------------
    "ll_baking_time": "Load-lock bake-out duration (minutes).  Type an integer, then start the bake from the Baking window.",
    # --- Status ---------------------------------------------------------
    "Error": "Status / error messages from the pump and gauge controllers.",
}

# ===========================================================================
# Cameras & Alignment window  (pyccapt.control.gui.gui_cameras)
# ===========================================================================

CAMERAS_TOOLTIPS = {
    # --- Live image views ------------------------------------------------
    "cam_s_o": "Camera Side - overview view (wide field).",
    "cam_s_d": "Camera Side - detail view (zoomed in).",
    "cam_b_o": "Camera Top - overview view (wide field).",
    "cam_b_d": "Camera Top - detail view (zoomed in).",
    "cam_angle_o": "Camera Angle - overview view.",
    "cam_angle_d": "Camera Angle - detail view.",
    # --- Override / illumination ----------------------------------------
    "superuser": "Override Access unlocks illumination and camera exposure controls after confirmation.",
    "light": "Turn the Arduino-controlled NeoPixel light on or off. Disabled until Override Access is granted.",
    "led_light": "Illumination state: green = on, red = off.",
    "illumination_percent": "NeoPixel dimming level, 0 to 100 percent. Disabled until Override Access is granted.",
    # --- Exposure controls ----------------------------------------------
    "exposure_time_cam_1": "Exposure time for the side camera, microseconds.  Increase if the image is too dark.",
    "exposure_time_cam_2": "Exposure time for the top camera (µs).",
    "exposure_time_cam_3": "Exposure time for the angle camera (µs).",
    "auto_exposure_time": (
        "Toggle automatic exposure for all cameras. Disabled until Override Access is granted."
    ),
    "led_auto_exposure": "Auto-exposure indicator: green = on (Continuous), red = off (manual).",
    "default_exposure_time": "Enable/disable manual camera exposure entry. Disabled until Override Access is granted.",
    # --- Status ----------------------------------------------------------
    "Error": "Status / error messages from the camera controller.",
}

# ===========================================================================
# Visualization window  (pyccapt.control.gui.gui_visualization)
# ===========================================================================

VISUALIZATION_TOOLTIPS = {
    # --- Live readouts --------------------------------------------------
    "voltage": "Live DC voltage on the specimen (V).",
    "detection_rate": "Live detection rate (%).  Set the target in the main window.",
    "hitmap_count": "Number of ion hits currently rendered in the hitmap panel (capped by 'hits displayed').",
    # --- Detector view (hitmap panel, left) -----------------------------
    "diagram": "Live detector hitmap - red dots are recent ion impacts on the detector face.",
    "hit_displayed": "Maximum number of hits to keep on the hitmap at once.  Lower = lighter rendering.",
    "hitmap_plot_size": "Hitmap dot size (pixels).",
    "reset_heatmap_v": "Clear both hitmap and FDM panels and start fresh.",
    # --- Detector view (FDM panel, right) -------------------------------
    "detector_fdm": "Field-desorption map - log-scaled 2D histogram of "
    "ion impacts.  Colour intensity shows how many ions "
    "hit each detector pixel.",
    "fdm_count": "Number of ions currently contributing to the FDM.  "
    "With Last Events OFF this is the cumulative ion "
    "total; with Last Events ON it is the size of the "
    "sliding window (capped at the N field).",
    "fdm_last_events_switch": "Toggle between cumulative FDM (off, "
    "default - every ion since experiment "
    "start) and sliding-window FDM (on - "
    "only the last N ions, where N is the "
    "field to the right).",
    "fdm_max_ions": "When Last Events is ON, the FDM uses only the "
    "most recent N ions (sliding window).  When OFF, "
    "this field is ignored and every ion is "
    "accumulated forever.  Default 1000000.",
    # --- Spectrum view --------------------------------------------------
	"btn_view_mc": "Show the raw (uncalibrated) mass-to-charge spectrum (Da).",
	"btn_view_tof": "Show the raw (uncalibrated) time-of-flight spectrum (ns).",
	"btn_view_mc_cal": "Show the live-calibrated mass-to-charge spectrum (Da).",
	"btn_view_tof_cal": "Show the live-calibrated time-of-flight spectrum (ns). "
	                    "All four spectra accumulate in parallel; switching is instant and never resets data.",
    "spectrum_last_events_switch": "Show only the last N events in the spectrum (rolling window).",
    "num_last_events": "Number of most-recent events used by the 'Last Events' spectrum mode.",
    "max_mc": "Upper limit of the mass spectrum X axis (Da).",
    "max_tof": "Upper limit of the time-of-flight spectrum X axis (ns).",
    # --- Hold + range ---------------------------------------------------
    "dc_hold": "Hold the DC voltage at its current value (pause feedback control).",
    "set_dc_voltage_value": "Target DC voltage (V) to apply to the supply. Enabled only while DC is held; limited to the config min/max DC range.",
    "set_dc_voltage": "Apply the entered DC voltage to the supply now. Enabled only while DC voltage is held.",
    "detection_rate_range_switch": "Toggle the detection-rate display between Short and Long Range.",
    "Error": "Status messages from the visualization process.",
}

# ===========================================================================
# Baking window  (pyccapt.control.gui.gui_baking)
# ===========================================================================

BAKING_TOOLTIPS = {
    "save_data": "Save the current bake-out log to CSV.  The file is "
    "written next to the experiment data with a timestamp "
    "in the name.",
    "Error": "Status / error messages from the baking logger.",
}
