# Origami laser control audit

Checked against the local vendor documents in `D:/Softwares/Oxcart_laser_manual`
on 2026-09-24. This is a source/manual and simulated-protocol verification;
no serial port, laser emission or stage motion was operated during the audit.

## Findings and changes

| Area | Manual evidence | Implemented correction |
| --- | --- | --- |
| Power setting | 800-621-01 p119: `ly_oxp2_power` is legacy pre-5.0 pulse energy in nJ; p120: `e_power` is relative AOM amplitude 0–4000 | Modern GUI uses **IR AOM setting (%)**, converted to 0–4000. Never presents this as W or mW. `laser_aom_max_percent` limits commands; old `max_laser_power` is not reused. |
| IR monitor | p119 `e_mlp?` description mentions average power and a factory energy option; both examples explicitly return nJ | Parse the response unit (W/mW/µW or J/mJ/µJ/nJ/pJ). Unitless or ambiguous readings are unavailable. |
| Harmonic monitor | pp139–140 `ls_output_power?` returns selected harmonic average power in W | Green/UV/DUV use this monitor, never substitute the IR monitor. Unsupported FHG replies remain unavailable. FHG firmware compatibility still needs a real readback. |
| Energy arithmetic | E=P/f, with f=base/divider | Monitor power produces energy in nJ; energy replies produce estimated power. GUI shows W and µJ. These are internal-monitor estimates, not measured energy at the specimen. |
| Reply parsing | pp119–125 examples include descriptive text, identifiers containing digits and multiline responses | Complete bounded serial transactions; parse values outside identifiers. Query commands use the documented question marks. Missing response raises a timeout. |
| Frequency | pp121–122 indices/frequencies are factory configured per laser | Read `e_freq_available?`; do not assume index 6/7 mappings from another laser/manual example. Read accepted index and divider back after edits. |
| Divider | p123 range 1–10,000,000; QSG p6 explicitly uses 400 kHz / 10 = 40 kHz | Remove the erroneous 50 kHz output floor. Preserve the firmware-accepted divider. |
| Wavelength | pp138–139 indices 0 IR, 1 Green, 2 UV; p143 DWLS table additionally identifies index 3 FHG | Read actual selected position and show nominal wavelength, not just the requested dropdown. DUV uses index 3. |
| Emission | pp116–118 and QSG p8: `ly_oxp2_enabled` enables emission and opens the output | Button explicitly says **Laser On (emits)**; state/LEDs follow actual status. Removed automatic full-power AOM writes after settings changes. Output toggle acts from states 65/129. |
| Editing while operating | p122 requires Listen/Standby for base frequency; wavelength changes can reopen AOM (p139) | Base frequency/wavelength restricted to Listen/Standby outside experiments; divider frozen during experiments. AOM edits require internal mode (2). |
| Recording | Existing HDF5 schema declares pJ while GUI stored nJ | Convert nJ ×1000 at detector acquisition boundary. Add monitor source, raw replies, nominal wavelength, rate, divider, AOM and validity to metadata/time series. |

Pure Laser experiments use actual readback output frequency for rate calculations.
VoltageLaser requires the configured voltage pulse frequency to match laser output
frequency before startup (within 1 Hz). This is a numerical consistency check;
it does not establish physical timing synchronization. Fresh valid telemetry is
required when starting an enabled laser experiment. Pure Laser runs stop if the
monitor data becomes invalid or older than 10 seconds. GUI reads do not modify
settings; requested commands are confirmed by subsequent readback.

## CLI connection diagnosis (2026-10-05)

Read-only queries on the configured COM9 at 38400 baud confirmed that the laser
was already in CLI mode. The raw status reply was
`ly_oxp2_dev_status?\nly_oxp2_dev_status 9\n`: command echo followed by a
newline-terminated Listen status, **without a trailing `>` prompt**. The AOM
mode reply was `e_mode of AOM: 2`. The repetition-rate table also omitted the
prompt and arrived in several fragments over approximately 62 ms.

The old transaction reader required a prompt and therefore rejected these
valid replies as timeouts. Transactions now accept either an explicit prompt
or an LF-terminated non-echo response after 200 ms of serial silence, within
the existing two-second deadline. This quiet interval preserves multiline
replies instead of returning immediately after the first newline. An echo
alone, an empty response or an unterminated line still times out. Timeout
messages include the byte count and a bounded raw-reply excerpt for diagnosis.
Firmware with gaps longer than 200 ms between complete lines may need a longer
quiet interval; no such gaps were observed in this instrument's replies.

The CLI mode probe uses the same status parser and documented question-mark
syntax. It requires a valid numeric status rather than accepting an echo as
proof of communication. The GUI releases its current serial handle before
probing to avoid mistaking its own port ownership for a failed CLI connection.
Failure messages no longer infer NKTPBus mode from a timeout alone.

After the fix, the driver read status 9, internal AOM mode, all seven available
repetition rates, IR wavelength and `0 mW`, and the CLI probe returned true.
This diagnostic sent read queries only; it did not change interface mode,
emission, wavelength, power or stage position. Restart the laser GUI/application
to load the corrected driver. If a port-open error occurs, close other serial
applications using that port; NKT CONTROL was running during this check, but
COM9 opened successfully, so exclusive ownership was not the observed failure.

The subsequent GUI startup at 12:44:12 recorded an echo-only setter reply,
`ly_oxp2_listen\r\n`. Setters now permit that framing, then their callers read
back actual status/settings; an echo remains insufficient for a query or for
claiming a completed transition. Setup status 17 no longer disables the Listen
recovery request. See [STATE_MECHANISMS.md](STATE_MECHANISMS.md) for the action
table, warmup handling and the limits of the broader device-state architecture.

## Wavelength values

The GUI shows **nominal harmonic wavelengths**: IR 1030 nm, Green 515 nm,
UV 343 nm if present, DUV 257.5 nm. It does not contain a spectrometer and cannot
show an instantaneous measured wavelength. The specific factory report for laser
SN4906 and FHG SN00002 lists:

| Output | Factory measured centre | Report page |
| --- | --- | --- |
| IR | 1029.04 nm | 13 |
| Green | 514.17 nm | 8 |
| DUV | 257.16 nm | 2 |

Those 2023 measurements are reference values, not live readings. At the report's
400 kHz nominal setting, 0.53 W DUV corresponds to 1.325 µJ, consistent with its
rounded 1.33 µJ entry. The report's “IR power set to 4.65 W” is an operating
condition; it does not establish the units of a CLI command.

## Base frequency, divider and displayed units

The main manual, **800-621-01 page 122**, requires Listen or Standby for changing
the base frequency using `e_freq`. **Page 123** explicitly permits changing the
divider `e_div` during laser operation. The GUI therefore keeps the base-rate
dropdown locked while on and permits divider edits in stable on states outside
an experiment. Both controls remain locked during acquisition to preserve run
timing; that acquisition restriction is PyCCAPT policy, not a vendor prohibition
on changing the divider while emitting.

The specific SN4906 factory report, **page 13**, lists 400 kHz as the nominal IR
rate and adjustable rates of approximately 400, 500, 580, 721, 800, 899 and
1000 kHz. The read-only `e_freq_available?` reply recorded on 2026-10-05 provides
their exact factory values: 400000, 500000, 579710, 720720, 800000, 898876 and
1000000 Hz. Thus 400 kHz is not this laser's maximum supported base rate. No
frequency was changed to obtain that evidence.

The dropdown is labelled **Base rate (kHz)** and displays the device's rates
divided by 1000. Its stored item data remains the factory command index, and
telemetry/calculations remain in Hz. The optical LCD is labelled **Output rate
(kHz)** and shows the actual readback base rate divided by the actual divider.
For example, base 400 kHz and divider 4 produce output 100 kHz. External gating
can further reduce the rate delivered to the specimen. Hardcoded GUI frequency
options have been removed; the list comes only from device readback.

An error containing only `e_div=4\r\n` indicates an echo-only setter response,
not a reported laser fault or proof that the divider change was refused. The
corrected driver returns from that setter and the GUI reads `e_div?` to obtain
the accepted value. Restart the entire Python/application process to replace
an older driver still loaded in memory; reopening a GUI from the same Python
console may retain imported modules.

## Remaining instrument verification

The supplied general manual predates some FHG-specific details. Its `e_mlp`
description is internally inconsistent, so the application trusts only explicit
units in a complete returned response. Monitor location relative to the pulse
picker and downstream optics must be verified on the actual setup before using
P/f as calibrated specimen pulse energy. Division and external gating can affect
the relationship; software makes no claim to measure external-gate duty cycle.

Capture actual read-only replies for `e_freq_available?`, `e_freq?`, `e_div?`,
`e_power?`, `e_mode?`, `e_mlp?`, `ls_wavelength?`, and `ls_output_power?` to check
the installed firmware dialect and FHG monitor availability. No connection or
emission changes were made to obtain those replies in this audit.

For DUV, a working frequency readout alongside dashes for optical power/energy
means the selected harmonic monitor has no usable reading; dashes are **not
zero watts**. The general manual's `ls_output_power?` description (pp139–140)
names Green/UV and does not establish support for this newer FHG/DUV firmware.
The real DUV reply is needed before deciding whether to extend the parser.
An attempted read-only diagnostic on 2026-10-05 could not open COM9 because it
was already owned by another process; no live DUV reply was captured.

The optical LCD tooltips now show the reason and a bounded raw monitor reply.
`laser_telemetry.error` preserves the diagnostic alongside the existing raw
replies, and the GUI log records changed failures once and subsequent recovery.
Pulse energy requires a valid selected-output power/energy reading; knowing
frequency alone is insufficient. A valid `0.00W` reply displays numeric zero;
unsupported, ambiguous or missing monitor readings display dashes. IR monitor
values retain their IR meaning and are not presented as measured DUV output.

The laser GUI displays **Measured IR power (W)** separately from **Selected
output power (W)**, **Pulse energy (µJ)** and **Output rate (kHz)**. The AOM
control remains a relative percentage. The IR readout uses the existing
`e_mlp?` query and stays available when the selected harmonic monitor is
unsupported or times out. Conversely, an IR-monitor failure does not hide a
valid selected-harmonic reading. Missing, disconnected or stale readings show
dashes; an explicit zero remains zero. When `e_mlp?` returns energy units, the
IR label changes to **IR power estimate (W)** and its tooltip explains the
energy/rate conversion. These values describe internal monitors, not specimen
power or pulse energy.

The local `Oxcart Laser Safty Turn On.docx` describes the external curtain,
cover/key/arm interlocks and physical shutter. Software status is not a readback
of those external interlocks; this audit does not claim they are instrumented.

## Switching between PyCCAPT and NKT CONTROL

In the Laser Control window, enable **Override Access** and click **Nktpbus
mode** to hand the laser to NKT Photonics CONTROL. PyCCAPT suspends CLI polling,
clears pending laser actions, closes its current session, then sends the same
`ly_oxp2_nktpbus=1\n` command as `python -m
pyccapt.control.nkt_photonics.nktpbus_activate`. The command is flushed and the
port closed without waiting for a CLI reply: this instrument can switch
protocols without acknowledging it. The banner reports that the command was
sent and the port released; it does not claim a verified NKTPBus readback.

PyCCAPT can remain open while CONTROL operates the laser. Disconnect the laser
in CONTROL (or close CONTROL) before clicking **Switch to CLI**. Windows permits
only one serial-port owner. If another application holds the port, PyCCAPT
reports that and allows another attempt without restarting. Switching is
blocked during an experiment or laser alignment.

The CLI button first releases any previous PyCCAPT session and probes CLI. If
CLI does not respond, it uses the NKT SDK to write OXPS register `0x39 = 1`
(documented in the installed SDK's `Register Files/95.txt`). The SDK port is
closed before the GUI reconnects at 38400 baud. Reconnection requests Listen
and refreshes device readbacks; failed handshakes close their port before a
retry. Polling resumes after the transition, and repeated mode clicks cannot
overlap. Failed SDK imports are removed from the cache so a later click can
retry loading the SDK.

Restart PyCCAPT once after installing this code to load the updated handlers.
Subsequent mode switches do not require a software restart. Hardware round-trip
verification remains necessary; regression tests use simulated serial ports.

The 2026-10-20 GUI log found OXPS at `0x0F` on COM9 but returned SDK result
`6:RegResultTimeout` from the `0x39 = 1` write. The protocol change resets the
mainboard (manual p166), so an absent bus acknowledgement is ambiguous. Both
the CLI button and IR-watt restore now close the SDK port first, wait for the
reset, and retry read-only CLI status queries. SDK result 6 is accepted only
when a valid numeric CLI status is subsequently received; NACK/CRC/type errors
still fail. Even an acknowledged write is not reported as a completed switch
without CLI verification. Probes close their serial handles after each attempt,
and a shutdown cancels further retries.

A separate live diagnostic process encountered a native SDK access violation
in `deviceGetType` during address probing. Discovery now reads the SDK's
auto-scan table through `deviceGetAllTypes`, whose byte indexes are the module
addresses. It does not issue type queries to absent addresses. Read-only COM9
and SDK checks initially did not establish communication; the operator then
confirmed the controller had been powered on. At 38400 baud a read-only status
query returned status 11 after a standalone newline cleared partial input.
New CLI sessions now terminate partial ASCII input and drain boot/help output
until a quiet interval before their first query. Clearing only the host RX
buffer cannot clear a partial command buffered by the laser itself.

With the controller powered, a live CLI -> NKTPBus -> CLI cycle found module
`0x0F`, reproduced SDK result `6:RegResultTimeout`, then verified CLI status on
the first probe after closing the SDK port. The final status was 17 (setup).
No emission-enable, interlock-reset or stage-motion commands were issued.
Two further complete cycles succeeded in one process using the same cached
SDK module. Each reproduced result 6 and verified CLI on the first status
probe; the laser was left in CLI mode with all diagnostic port handles closed.
Regression tests cover the reported reset timeout, boot delays, partial input,
help output, absent CLI replies and exclusive port ownership.

## IR watt control alongside relative AOM

### Persistent native IR control

For watt adjustment while emitting, select IR, Green or DUV through CLI and
use Listen/Standby, grant Override Access, then choose **Native W control**
before enabling emission. Wait for any queued wavelength edit to be read back.
This performs one CLI -> NKTPBus handover and retains the SDK connection.
The state label identifies native watt control. **Apply W** then writes only
the native `0x20` power setpoint and verifies its readback; repeated watt
changes do not write the FSM, shutter, modulation mode or protocol registers.
Relative AOM remains editable using `0x05`. Telemetry refreshes the firmware
watt setpoint as well as the relative AOM setting without issuing new writes.
The requested watts always refer to the fundamental IR power feeding the
converter, including while Green or DUV is selected. They are not a calibrated
Green/DUV output-power target and do not use a fixed conversion efficiency.

The persistent adapter maps the documented main-board FSM, interlock, fault,
frequency, divider, IR monitor and power registers into the existing GUI
state contract. Connection and polling never reset an interlock or enable
emission. Explicit Laser On/Output Enable buttons retain their emission
semantics. Manual watt requests remain blocked during experiments/alignment,
warmup, external modulation and faults. A relative AOM ceiling below 100%
blocks nonzero native watt requests because its amplitude cannot be predicted
before the firmware applies its calibration; a watt ceiling is enforced
independently.

The native adapter retains the wavelength index actually read back through CLI
before handover; it never defaults a Green/DUV selection to IR. This is cached
selection provenance, not a new native harmonic-position measurement. The raw
reply is marked `CLI-verified`, telemetry records this basis, and the wavelength
tooltips identify it. Wavelength editing is locked until returning to CLI.
The installed harmonic register map does not establish FHG/DUV scaling, so no
harmonic watt values are guessed: selected harmonic power and pulse energy
remain unavailable in native mode while the measured IR monitor remains live.
Existing CLI harmonic controls remain available. **Switch to CLI**
and **Nktpbus mode** release the SDK session; these explicit ownership handovers
request Listen, unlike watt-only changes. Native shutdown uses the existing
safe-off sequence before closing the handle. No live emission was enabled to
test this feature; repeated updates in emitting FSM states with IR, Green and
DUV selected are simulated. Physical wavelength retention across the initial
protocol reset and harmonic conversion efficiency still need instrument
verification; individual watt updates perform no protocol reset.

### Legacy CLI watt requests

**IR power control (W)** and **Apply W (Listen)** provide an explicit watt request
without replacing the AOM percentage. The instrument-specific ceiling is
`laser_ir_max_power_w` (5.2 W for this setup); requests use 0.001 W resolution.
The NKT SDK must be installed. This is a requested IR setpoint, distinct from
the measured IR display and the selected harmonic's power/energy.

The CLI's `e_power` command is relative; it is not converted to watts using
the configured maximum. The vendor manual 800-621-01 p165 documents OXPS
NKTPBus register `0x20` (U16, 0.001 W) as a firmware conversion into the
linearized AOM amplitude in `0x05`. PyCCAPT uses that conversion, confirms
the accepted watt setpoint and reads the calibrated AOM value. Both protocol
changes reset the laser to Listen (manual pp133,166), so the GUI restores
CLI, verifies the same base-rate index, divider and wavelength, and reapplies
the calibrated AOM amplitude only after confirming internal mode and Listen.
Its AOM readback must match before the GUI reports success.

Apply is available only in Listen or ready Standby, outside experiments and
alignment; it leaves the laser in Listen and sends no emission-enable command.
Polling is paused and each serial/SDK handle is released before the next open.
Rejected requests attempt CLI recovery and never reapply an unverified AOM
value. Firmware refusals, changed timing/wavelength, the AOM limit, external
modulation, shutdown and reconnect failures prevent successful application.

The watt request is not a continuous power feedback loop. Subsequent manual
AOM edits and firmware resets/settings changes can override it. Readbacks
remain authoritative; the GUI never automatically restores a previous power
request after a state or frequency change. Live watt accuracy and round-trip
behavior still require instrument verification; automated tests use simulated
SDK and serial responses.

## Verification

`tests/control/test_laser_manual_contract.py` exercises manual-format multiline
serial replies, explicit power/energy units, command-name digit rejection,
timeouts, frequency tables, rejected divider settings, unknown harmonic data,
button locks and Laser On behavior using simulated devices only.
