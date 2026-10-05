# Control state mechanisms

All active control owners publish a common state contract through
`core/control_state.py` and `core/share_variables.py`. Existing experiment,
alignment and laser state machines keep their device-specific transitions and
interlocks. Compatibility adapters preserve legacy flags and existing command
sequences while providing consistent state records for monitoring and diagnosis.

## Shared contract

Each resource has one registered observation owner and an explicit list of
requesting roles in `STATE_SPECS`. Roles describe logical ownership; `main` and
`pump` can be separate threads in the same process. `Variables._OWNERSHIP`
registers every shared record. Updates use a dedicated multiprocessing lock and
an atomic read/reduce/write transaction, including on Windows with spawned
processes. State-lock acquisition is bounded: best-effort publications wait at
most 10 ms, snapshots at most 50 ms. An exited publisher holding that lock
cannot trap subsequent physical stop commands. State publication failures are
logged while existing hardware operations continue. Records and returned
metadata are detached from their publisher.

| Field | Meaning |
| --- | --- |
| `connection` | Unknown, configured disabled, connecting, connected or disconnected |
| `requested`, `command_id` | Latest requested action/target, independent of the observed state |
| `observed`, `evidence` | Owner's named state and its basis: none, commanded, hardware readback or software lifecycle |
| `valid`, `observed_at`, `fresh` | Whether the observation is usable, its monotonic timestamp and its age check |
| `command_status` | None, requested, sent, confirmed, failed, cancelled or timed out |
| `confirmation`, `deadline`, `command_overdue` | Expected completion observation, optional deadline and diagnostic overdue check |
| `fault`, `command_error` | Current observation fault and retained last command failure/cancellation reason |
| `details`, `request_details`, `revision` | Separate observation/request metadata and monotonically increasing update revision |

Requests and command writes do not overwrite existing readback/software observations or refresh their age. A successful command write remains
`sent` with `commanded` evidence. Confirmation requires a matching, valid, fresh
readback or software lifecycle observation obtained after the request. A reply
for a superseded command ID is ignored. Older observations cannot clear a newer
fault or request. Current readback faults may clear on a successful new read;
failed command outcomes and their errors remain available until a new command.

Freshness limits are specified per resource. Position/motion and the physical
interlock use three seconds; pumps, vacuum/cryo readbacks, optical telemetry and
camera frames use six seconds. The baking temperature DAQ uses three seconds.
Persisted software phases and write-only outputs retain their last published
state; existing worker heartbeat monitoring remains authoritative for process
liveness. Freshness is computed when diagnostics are read, without changing
stored outcomes or issuing any hardware command.

### Resource coverage

| Owner / resources | Publication source and evidence |
| --- | --- |
| Experiment, output shutdown | Existing validated lifecycle and safe-off flag; software lifecycle/report. Individual output writes retain commanded evidence |
| Detector (all acquisition backends) | Existing start/stop/join/health interface; worker lifecycle evidence, separate from detector electronics |
| Sample and laser alignment | Existing native phases, outcomes and requests; queued GUI requests remain separate from worker progress |
| Both motion services | Original command IDs, moving/error/settled results and calibrated settling checks; completion uses measured positions |
| Both SmarAct stages | Connect/read errors, manual jog/home/reference/stop requests and XYZ snapshots; position readback confirms coordinates, not a new motion policy |
| Laser and optical monitors | Actual CLI status, state requests and original optical telemetry; cached UI refreshes never refresh readback age |
| Main/LL/CLL gates | Existing NI pulse commands and pulse errors; commanded positions, without position-switch confirmation |
| LL/CLL pumps | Original Edwards command sequence and speed polling; `at_speed` / `below_speed` expose the controller's existing 90 threshold |
| Seven pressure channels, cryo | Existing sensor publication/error sentinels; measured pressures and stage temperature |
| Cryo/LL heater controllers | Existing setpoint writes; commanded regulation/off, without independent heater-current measurement |
| CLL vent/backing/turbo valves | Existing NI writes and delayed sequences; commanded line-high/line-low preserves the actual wiring convention |
| Three camera slots | Attach/detach, successful frame grabs and failures; camera backend/readback evidence |
| Illumination | Original Arduino connection and on/off/brightness commands; commanded light state |
| DC/pulse supplies, signal generator | Original initialization/output-on/output-off/frequency commands; commanded output state |
| Safety interlock | Existing NI-DAQ readback, or explicitly `no_hardware_backend` with software evidence |
| Baking / LL baking / visualization | Existing monitoring/logging/display lifecycle; software evidence |
| Baking temperature DAQ | Eight existing MCC temperature inputs, including unavailable-backend/read errors |

Legacy field adapters are restricted to low-rate status/request fields. Per-ion
arrays and counters do not enter the registry. Existing button locks, override
rules, motion limits, voltage regulation, acquisition and cleanup sequences keep
their current behavior. The registry observes these decisions and validates its
own command lifecycle; callers continue to send actions through the existing
device owner and its guards. Publishing a state request performs no actuation.

### Diagnosis and extension

Live health messages include a `control_states` dictionary. State/connection,
request, error and recovery transitions are logged under `pyccapt.state` in the
existing application log. Repeated numeric polling updates timestamps/details
without producing another transition log line.

Each run that creates its dataset directory saves an atomic final snapshot to
`meta_data/control_states.json` after cleanup and final lifecycle publication.
Invalid numeric values are represented as JSON null. This snapshot complements
the stage/laser alignment event journals and daily transition log. Its freshness
flags describe the snapshot time; stored monotonic times belong to the same
computer runtime clock and should not be compared against a later reboot's
clock. Snapshot failures are logged and leave normal completion reporting intact.

Developers can inspect the common API without touching hardware:

```python
states = variables.control_states()
laser = states["laser"]
print(laser.requested, laser.observed, laser.command_status.value)
print(laser.to_dict())  # includes current fresh/command_overdue calculations
```

For a new resource, register its owner, permitted requesting roles,
configuration key and readback age in `STATE_SPECS`. Publish `request`,
`connection`, `observe`, `result` and `fault` events via
`Variables.update_control_state()` or the best-effort owner helpers. Direct
assignment to shared `control_state_*` fields is rejected. Use real readbacks
only at their acquisition sites. Add explicit resource-specific transitions and
command guards in the owner when its hardware contract requires them. Keep
unsupported physical feedback represented as commanded/unavailable.

## Experiment and acquisition

`apt/experiment_state.py` defines and validates the normal sequence:

`idle → initializing → running → stopping → safe_off → finalizing → complete`

Failures may enter `failed`; a subsequent run starts through `initializing`.
Shared state is published through the existing `Variables` wrapper. The worker
reports typed status/health messages and a completion acknowledgement. The main
GUI monitors worker heartbeats and completion. Detector backends expose
`start`, `stop`, bounded `join`, and `health`; unresponsive detector processes
are terminated after the cooperative shutdown wait.

The `hardware_safe` flag includes initialization assumptions and is exposed as
`safe_off_reported` with software evidence. The experiment milestone
`safe_off` means the configured supply/pulser/signal-generator shutdown commands
completed without reported exceptions. It is not independent measurement that
every physical output is de-energized, and does not confirm the laser shutter.
The NI-DAQ E-stop/watchdog backend is available but requires configuration and
physical wiring. This installation currently selects
`safety_interlock_backend = "none"`, which uses the no-hardware interlock.

## Stage and alignment

Stage and laser alignment use explicit phases, command identities, measured
positions, bounded travel, settling checks, cancellation and deadlines. The
motion services own hardware commands and publish results for the experiment.
They reject inappropriate concurrent moves and stale requests. Alignment
journals record commands, results and phase changes. These contracts are more
specific than a universal device lifecycle.

## Laser

`nkt_photonics/state.py` maps **observed** status codes to named states and
defines permitted manual actions. The GUI uses the same permissions when
enabling buttons and executing requests. It displays the observed state and
logs changes; a clicked button is not treated as evidence of completion.

| Observed status | State | Available state requests |
| --- | --- | --- |
| 9 | Listen | Standby |
| 17 | Setup / warming | Listen |
| 33 | Standby / ready | Listen, Laser On (emits) |
| 65 | Laser on / output closed | Listen, Standby, Output Enable |
| 129 | Laser on / output open | Listen, Standby, Close Output |
| 1, 3, 5 or unknown | Booting, error, warning or status unavailable | Listen recovery if the CLI port is still open |
| Port closed | Disconnected | Reconnect via the existing CLI controls |

Standby thermalization may take up to 15 minutes according to the local NKT
manual, page 116. Setup is shown explicitly, with an orange Standby indicator.
Emission stays unavailable until ready Standby is actually read. Listen can be
requested during warming and is sent even if a status query is failing.
Clicking Standby enables Listen immediately, including the interval where the
device still reports its previous Listen state. Clicking Listen cancels any
unsent Standby/emission request. The pending transition clears on confirmed
ready status, a Listen cancellation or a communication error.
Laser On requests keep Listen and Standby available before the new status
arrives. Output Enable requests immediately expose Close Output while the last
readback still says output closed. A Close Output click queues an explicit
disable command, rather than re-evaluating a toggle against old readback and
accidentally enabling output again. Lower-state requests cancel queued upward
requests; they are not permission to automatically retry emission.
Availability of Listen means the software can attempt the command; a broken
serial connection can still prevent physical completion.

This firmware may reply to a setter with only the command echo. Setter
transactions may return after the echo and the quiet interval, then the GUI
reads back the actual state/settings. The echo is never proof of success.
Read queries still require a non-echo reply. Empty or incomplete replies time
out with a byte count and bounded raw excerpt.

Pending state requests are consumed once. Failed or disallowed emission
requests are cancelled rather than replayed after recovery or warmup. Settings
permissions use status read **after** a requested transition; wavelength and
frequency edits cannot slip through using the preceding Standby status after
Laser On has been sent. Settings are blocked while state is unstable.
The base rate (`e_freq`) requires Listen or Standby. The divider (`e_div`) can
be edited in stable on states outside an experiment. During acquisition both
remain locked. Base and output rates are displayed in kHz; device command
indexes and stored Hz telemetry retain their existing meaning.

Serial I/O still runs synchronously on the laser GUI thread with a re-entry
guard and bounded individual transactions. A device owner running I/O outside
the GUI thread, with queued commands and correlated results, remains a useful
future reliability improvement.

## Physical feedback and owner scheduling

The common contract provides consistent publication and diagnosis. Physical
feedback depends on each installed device: gates and several outputs remain
write-only, detector health describes a worker process, and the no-hardware
interlock has software evidence. Existing owner-specific guards retain their
current behavior. As hardware feedback becomes available, publish it through the
same observation contract and validate completion against that readback.

Laser serial I/O still runs on the GUI thread with its existing re-entry guard
and bounded transactions. Moving that I/O into a dedicated owner thread is a
separate scheduling change and is not part of this compatibility migration.
