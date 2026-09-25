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
