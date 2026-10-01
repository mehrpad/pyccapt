# Experiment plans

Use **Setup Parameters → TOML Plan** to prepare, reuse and run a queue of
experiments. The main window shows a compact queue. **TextBox** provides the
single-experiment form; **TOML Plan** provides the reusable queue.

The narrower main window groups the less frequently changed TextBox parameters
under **Advanced settings…**: detection mode, control algorithm, refresh frequency,
pulse-voltage limits and K_p up/down. Edits apply as each field is changed; the
normal experiment locks still apply. For TOML plans, use the queue's **Edit**
button to change those settings for a specific experiment. The Advanced settings
label and button sit directly below the target Detection Rate input on the left
in TextBox mode. A separator below the Run Statistics detection rate precedes
Electrode and Flat Test. A second line separates these buttons from the bordered
Auto Alignment box containing alignment start voltage, voltage increment and Automatic
Alignment. Start sits above Stop in a separate group just above the bottom
full-width separator and experiment queue.

## Load and run the example

1. Copy `pyccapt/files/experiment_plan.example.toml` to a working file such as
   `my_experiments.toml`. The example contains the two `test1` and
   `test2` experiments. Edit it for your instrument and samples.
2. Start PyCCAPT and select **TOML Plan** in **Setup Parameters**.
3. Click **Load TOML** and select your file. Loading resolves shared defaults
   and validates every experiment against the current hardware configuration.
   An invalid file leaves the existing queue intact and shows the reason.
4. Review the table: sample ID, experiment name, pulse mode, DC range in volts,
   pulse frequency in kHz, target detection percentage and stopping conditions.
   Double-click a row or click **Edit** to inspect all its settings and units.
5. Click **Start** after completing the usual instrument preparation. The queue
   runs from top to bottom. Each experiment finishes and its worker exits before
   the next starts. **Stop**, a run failure or an unsuccessful safe shutdown
   prevents automatic continuation. Load/edit/reorder controls are locked while
   the experiment or automatic sample sequence is active.

With **Automatic Alignment off**, all rows run at the current physical sample
position. `sample_id` is informational in this mode: it does not command stage
movement. To transfer to different saved positions, enable Automatic Alignment.

## Automatic sample alignment

Save coarse positions in **Cameras** for the sample IDs used in the plan, then
enable **Automatic Alignment**. Every experiment must specify `sample_id = 1`,
`2` or `3`, and that ID must have a saved position. Plan order controls the
sequence: sample 2 can precede sample 1, saved samples absent from the plan are
not visited, and repeated IDs deliberately revisit the same sample position.

All parameter sets, positions, stage calibration and alignment prerequisites
are checked before the first transfer. The normal three-step transfer retracts
Z to the configured clearance height, moves XY, then returns Z to the saved
position. See `pyccapt/control/AUTOMATIC_ALIGNMENT.md` for commissioning
and `pyccapt/control/LASER_ALIGNMENT.md` for combined laser alignment.

For the included example, set **Alignment start voltage to 2700 V**: the default
1500 V is outside both experiments' DC ranges. The alignment start voltage must
fit every queued experiment and the configured alignment voltage ceiling.
Stage and laser calibration settings remain in `pyccapt/config.toml`; the plan
does not replace instrument configuration or store stage coordinates.

## Create and edit a queue

- **Add** opens an experiment editor initialized from the main form's visible
  settings. Set a sample ID if automatic sample alignment will be used.
- **Edit** opens the selected row; changes must pass validation before acceptance.
- **Duplicate**, **Remove**, **↑** and **↓** change the selected row or its order.
- **Save As** writes a reusable `.toml` file. Common values are stored under
  `[defaults]`; each experiment contains its differences. Export regenerates
  the file and does not preserve comments from an imported file.

Loading and editing affect the in-memory queue. Editing the source file on disk
does not change it until you load that file again. At Start the resolved queue
is frozen; later experiments use the same validated snapshot.

## File format and units

The top-level keys are `schema_version = 1`, an optional `[defaults]` table and
one or more `[[experiments]]` tables. Every required setting must exist either
in defaults or in its experiment. Experiment values override defaults.
`sample_id` belongs to individual experiments and cannot appear in defaults.
The complete runnable example is `pyccapt/files/experiment_plan.example.toml`.

| Settings | Type / units |
| --- | --- |
| `ex_user`, `ex_name`, `electrode` | Nonempty quoted strings |
| `email` | Quoted string; `""` disables recipient email |
| `pulse_mode` | `"Voltage"`, `"Laser"`, `"VoltageLaser"` |
| `counter_source` | `"TDC"`, `"HSD"` |
| `control_algorithm` | `"Proportional"`, `"Proportional aggressive"`, `"Adaptive P"`, `"PID"` |
| `ex_time` | Whole seconds |
| `ex_freq` | Positive whole control-loop Hz |
| `max_ions` | Whole ion count |
| `vdc_min`, `vdc_max`, `vp_min`, `vp_max` | Whole volts |
| `vdc_steps_up`, `vdc_steps_down` | Nonnegative volts per control update; decimals allowed |
| `pulse_fraction` | Whole percent |
| `pulse_frequency` | Positive whole kHz, e.g. `200` means 200 kHz |
| `detection_rate_init` | Detection percentage; decimals allowed |
| `hit_displayed` | Positive whole displayed-hit count |
| `criteria_time`, `criteria_ions`, `criteria_vdc` | TOML booleans `true` / `false`; enable at least one |
| `criteria_email` | Optional boolean; default `false`; progress-email toggle, not a stopping condition |
| `email_interval_events` | Optional positive whole ion count; default `1000000` |
| `sample_id` | Optional whole number 1–3; required for automatic sample alignment |

Numbers are unquoted and booleans are lowercase, e.g. `vdc_steps_up = 0.5` and
`criteria_time = true`. Unknown keys, duplicate TOML keys, wrong types, missing
settings, nonfinite numbers, unsupported modes and hardware-limit violations
are rejected. Plans reject values that the single-run form would clamp, so a
file cannot silently run with different settings. Multiple stopping conditions
use the existing experiment logic: reaching any enabled condition stops the run.
The maximum-DC criterion includes the existing approximately ten-second dwell
at the voltage ceiling.

## Recorded settings

Each TOML-plan experiment writes:

- `meta_data/experiment_plan.toml`: the complete resolved starting settings for
  that experiment, including its sample ID when supplied. It can be loaded again
  as a one-row plan.
- `meta_data/experiment_plan_source.json`: the source filename (empty for an
  unsaved queue) and one-based queue index.

These accompany the existing experiment metadata, logs and alignment records.
The snapshot captures starting parameters; later operator adjustments remain
subject to the normal runtime controls and logging.
