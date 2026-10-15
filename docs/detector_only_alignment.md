# Detector-only alignment methods

These methods use ion detector hits and count/detection-rate evidence. They do
not use cameras, spectra, MRP, charge-state ratio or laser-power optimization.
The user sets the laser power. Its enabled readback, frequency and wavelength
remain fixed throughout sample/laser comparisons. Changes end alignment.

All speed/coverage options are opt-in in `pyccapt/config.toml`. The bounds,
motion owner, interlocks, acknowledgements, settling, fresh detector epochs,
10-second no-event gate and existing fixed-DC candidate verification remain.
Only the experiment owner makes the already-supported stationary DC retries.
No instrument commissioning or real-device timing benchmark is implied.

## Laser options

| Key | Default | Effect |
| --- | --- | --- |
| `laser_alignment_search_strategy` | `"raster"` | `"hybrid"` tries a bounded local response proposal, with raster fallback |
| `laser_alignment_adaptive_dwell` | `false` | Use count evidence to shorten/extend observations within limits |
| `laser_alignment_min_dwell_s` / `laser_alignment_max_dwell_s` | 0.5 / 3 | Minimum/maximum measurement duration; settling is unchanged |
| `laser_alignment_target_relative_error` | 0.15 | Target approximate count error `1/sqrt(N)`, not alignment confidence |
| `laser_alignment_early_reject_sigma` | 4 | Reject clearly inferior positions after minimum dwell; never accept a candidate early from this test |
| `laser_alignment_reuse_previous` | `false` | Verify a compatible previous successful position before repeating coarse search |
| `laser_alignment_reuse_context` | empty | Explicit specimen/tip identity, required for reuse; change when the specimen changes |
| `laser_alignment_coordinate_calibration_id` | empty | Stage-coordinate/optics calibration identity, required for reuse |
| `laser_alignment_reuse_max_age_s` | 3600 | Maximum age of the cached solution |
| `laser_alignment_reuse_voltage_tolerance_v` | 5 | Maximum DC difference for proposing reuse; all new comparisons hold current DC fixed |

The hybrid method measures centre and axial probes, then informative diagonals.
Information gain and travel time order equivalent probes. In normalized stage
coordinates it fits a local surface:

```text
log(rate) = b0 + bx*x + by*y + hxx*x²/2 + hxy*x*y + hyy*y²/2
```

Count errors weight the regression. A proposal requires distinct positions,
positive rate evidence, full-rank coverage, acceptable condition number, small
leave-one-out residuals and negative curvature separated from its approximate
uncertainty. The stationary point must lie inside both the probed range and the
original absolute/session limits. Its response is **measured**, the centre is
remeasured for drift, and the selected candidate is separately verified before
the original statistical/fractional improvement rules may accept it.

Flat, low-count, non-concave, inconsistent or out-of-range fits use unvisited
raster points. This assumes one smooth local lobe; it cannot rule out a narrow
unvisited competing lobe or guarantee the global optimum. Focus Z retains its
bounded measured scan and optimizes detector response only. Legacy spectral ROI
settings no longer influence motor decisions.

Adaptive dwell cannot accept a low-count peak: count/signal rules still apply.
It ends after the minimum only when count precision is adequate or an inferior
position's upper rate bound is below a measured reference's lower bound. The
early-rejection shortcut is disabled for candidate verification. At the maximum
dwell an inconclusive point stays inconclusive. Missing detector publications
retain the bounded watchdog behavior and do not reset the no-event gate.

Reuse stores successful position, available local curvature, explicit specimen
and coordinate identities, laser signature, sample location, DC, bounds and age
in shared application state. It is lost on app exit. Changed conditions or Z
reject reuse. Compatible curvature may shrink the following fine radius within
its configured half/full range. Fresh verification failure returns to ordinary
coarse search at unchanged DC. Operator-supplied identities must be updated after
specimen, coordinate-reference or optical changes; the software cannot infer them.

## Sample coverage and geometric evidence

`alignment_coverage_max_probes` defaults to zero. A positive budget, up to 256,
adds space-filling probes after the sparse search fails. With the default
`alignment_coverage_subdivisions = 8`, the ±49.8 µm inset envelope has 12.45 µm
candidate spacing. Largest remaining gaps are sampled first, with travel-cost
tie breaking. A sufficiently large budget can find an island at a grid point
missed by the sparse cross/diagonals; narrower unresolved islands can still be
missed. Added coverage costs observations on an absent sample and is included
in the conservative startup timeout estimate.

Dense-region centring uses region geometry; circular fits use boundary geometry.
Bright-pole/intensity centroids are separate diagnostics. Uniform background,
insufficient contrast, tiny hot pixels and clipping cannot become accepted
density geometry. The measured optional XY Jacobian is used as supplied; no
Jacobian is inferred or replaced by an identity matrix.

Area alone cannot determine Z: voltage, magnification and specimen geometry can
change it. `alignment_z_response_calibrated` defaults to false, and a nonempty
`alignment_z_response_calibration_id` is required when enabling it. This asserts
an independently measured response/clearance relation in addition to existing
approach limits. Both controller and stage service enforce it. Otherwise Z
approach is withheld; XY and stationary bounded DC correction remain available.

## Replay and progress

Live status shows model/fallback/reuse phase, measured windows/ions and coverage
probe budget. Journals record proposals, diagnostics, dwell decisions and
independent acceptance. Analyze one recorded session without controlling hardware:

```python
from pyccapt.control.apt.alignment_replay import read_sessions, summarize_replay
for session, records in read_sessions("meta_data/laser_alignment.jsonl").items():
    print(session, summarize_replay(records))
```

Reports include completed alignment, elapsed time, measured-window ions, moves,
commanded travel and fallback counts. `false_aligned_count` is unavailable unless
independent replay truth is supplied as `expected_position_m` and a position
tolerance. Laser counts exclude movement/transition packets; sample sequence
counts are a lower bound. Alignment selection does not filter saved acquisition
events. If Surface Concept still has copied packets queued when outputs are
switched off, their voltage cannot be assigned reliably: these packets are
retained under `temp_data/unattributed`, and finalization/recovery explicitly
report incomplete acquisition instead of silently discarding the tail or
inventing shutdown voltage metadata.

Tests cover known smooth and noisy peaks, flat/mismatched responses, bounds,
changed settings, stale reuse, missing Z calibration, bright poles, uniform
background and a narrow sample island missed by sparse probes. A smooth coarse
replay compares hybrid against raster with identical dwell/motion parameters;
complete noisy searches also score alignment against explicit known truth.
These are software regression results, not real-instrument success probabilities.
