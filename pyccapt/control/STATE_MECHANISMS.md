# Control state mechanisms

PyCCAPT has a validated experiment lifecycle and several device-specific state
mechanisms. It does **not** yet have one uniform, hardware-confirmed state
contract for every device. The following describes the current implementation,
including the laser corrections made on 2026-10-05.

## Experiment and acquisition

`apt/experiment_state.py` defines and validates the normal sequence:

`idle → initializing → running → stopping → safe_off → finalizing → complete`

Failures may enter `failed`; a subsequent run starts through `initializing`.
Shared state is published through the existing `Variables` wrapper. The worker
reports typed status/health messages and a completion acknowledgement. The main
GUI monitors worker heartbeats and completion. Detector backends expose
`start`, `stop`, bounded `join`, and `health`; unresponsive detector processes
are terminated after the cooperative shutdown wait.

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

Serial I/O still runs synchronously on the laser GUI thread with a re-entry
guard and bounded individual transactions. A device owner running I/O outside
the GUI thread, with queued commands and correlated results, remains a useful
future reliability improvement.

## Gaps in device-wide consistency

Gates largely track commanded position using booleans; the diagram is not
independent confirmation from gate position switches. Gate opening uses
experiment, pump and vacuum checks. Pump control has command flags plus
controller status polling, rather than the experiment's typed transition table.
These mechanisms cannot all be described as one complete device state machine.

A consistent future contract should separate connection, requested operation,
observed operation, validity/freshness and fault information. Each device owner
should validate permitted commands, attach a command ID and deadline, confirm
completion from available hardware readback, publish recovery options and log
transitions. Hardware without position/output feedback must expose its state as
commanded or unconfirmed. This is a proposed next step, not a claim that the
contract has already been implemented for every device.
