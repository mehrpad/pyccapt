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

The local `Oxcart Laser Safty Turn On.docx` describes the external curtain,
cover/key/arm interlocks and physical shutter. Software status is not a readback
of those external interlocks; this audit does not claim they are instrumented.

## Verification

`tests/control/test_laser_manual_contract.py` exercises manual-format multiline
serial replies, explicit power/energy units, command-name digit rejection,
timeouts, frequency tables, rejected divider settings, unknown harmonic data,
button locks and Laser On behavior using simulated devices only.
