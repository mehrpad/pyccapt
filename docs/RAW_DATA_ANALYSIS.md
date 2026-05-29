# Raw-Data Analysis and Partial-Hit Recovery (Surface Concept)

This page documents how PyCCAPT analyses the **raw delay-line timestamps**
written by a Surface Concept (SC) 2-D delay-line detector, and how it
recovers atoms that the detector firmware discarded. It is a companion to
the shorter summary in [CALIBRATION.md](CALIBRATION.md#partial-hit-recovery-surface-concept)
and the on-disk schema in
[Calibration_DATA_STRUCTURE.md](Calibration_DATA_STRUCTURE.md).

The intent is practical: after reading this you should be able to predict
what happens to an SC pulse of any length, and read the statistics printed
at load time.

---

## 1. Two parallel data streams

A PyCCAPT acquisition file stores two groups that describe the same
experiment from two different points in the pipeline:

| Group  | Content | Produced by |
|--------|---------|-------------|
| `/dld` | Reconstructed events: `x_det`, `y_det`, `t (ns)`, `high_voltage`, … — one row per atom the detector firmware accepted. | SC firmware "quadrupel finder" (real time). |
| `/tdc` | Every raw delay-line **stop** the TDC recorded: `channel`, `time_data` (TDC bins), `start_counter`, … — several rows per pulse. | TDC hardware (raw). |

The two streams are captured separately. The firmware keeps **one hit per
pulse** (configuration `MultiHitDepth = 1` in `tdc_gpx3.ini`); the raw
`/tdc` stream keeps **all** stops. So `/tdc` generally contains more
information than `/dld`, and the two are not guaranteed to be stop-for-stop
identical.

When a file is loaded with `load_tdc_raw=True`, both groups are read and
linked by a shared **`event_group_id`** column. That link is what makes the
recovery and the cross-checks below possible.

---

## 2. The delay-line detector and the four channels

A 2-D crossed delay-line detector measures a hit with **four stops**:

- **x axis**: channels `0` and `1` (the two ends of the x delay line),
- **y axis**: channels `2` and `3` (the two ends of the y delay line).

When an ion strikes the detector, the MCP signal launches a pulse down each
delay line. Each line reports the arrival time at its two ends. The
detector **coordinate** along an axis depends only on the **difference** of
the two end times (where along the line the pulse started); the **arrival
time** (and therefore time-of-flight) depends on their **sum**.

### Reconstruction formulas

PyCCAPT reconstructs a hit with the same arithmetic the firmware uses
(`_raw_workflow_surface_concept._surface_concept_position_from_pair` and
`_surface_concept_hit_from_time_data`). With the SC defaults
(`TOF_FACTOR_NS = 27.432 / 4000`, `TOF_FACTOR_NS_1D = 27.432 / 2000`,
`XY_FACTOR = 80 / 4900 × 2` mm/bin):

```
det_x (cm) = -0.5 × (t_ch1 - t_ch0) × XY_FACTOR × 0.1
det_y (cm) = -0.5 × (t_ch3 - t_ch2) × XY_FACTOR × 0.1
tof  (ns)  = (t_ch0 + t_ch1 + t_ch2 + t_ch3) × TOF_FACTOR_NS
```

### The time-sum coincidence (the key constraint)

Both delay lines start from the same MCP signal and have the same total
propagation length, so for one ion:

```
t_ch0 + t_ch1  =  t_ch2 + t_ch3   (≈ 2 × t_MCP)
```

The SC quadrupel finder accepts a 4-stop set as one hit only when this
time-sum agrees within a tolerance (`SumDifMin/SumDifMax`, ±200 bins by
default). This identity is also what lets PyCCAPT reconstruct a missing
channel end offline (Section 4, "3 stops").

A useful corollary: because `t_ch0 + t_ch1 = t_ch2 + t_ch3 = S`, the full
time-of-flight equals the single-axis time-of-flight:

```
tof = (2S) × TOF_FACTOR_NS = S × TOF_FACTOR_NS_1D
```

so a complete single axis already fixes the ToF.

---

## 3. Grouping stops into pulses

Stops are grouped into pulses by **contiguous runs of equal
`start_counter`** in acquisition (row) order, not by the counter value.
`start_counter` is a hardware counter that **wraps** roughly every ~71
minutes at typical rates, so two physically distinct pulses can share a
counter value. Run-length grouping on a *change* in `start_counter` is
wrap-safe (`partial_hit_diagnostics.tdc_pulse_completeness`,
`partial_recovery.merge_partial_tdc_into_dld`).

A channel counts as **fired** when its `time_data` is non-zero.

---

## 4. Worked examples by pulse length

The examples below use small, illustrative TDC-bin values so the arithmetic
is easy to follow. Real stop times are larger (tens of thousands of bins),
but only the **differences** and **sums** matter. Throughout, a stop is
written `chN@T` meaning "channel `N` fired at time `T` bins".

### 0 stops — empty pulse

A trigger with no stops. Counted as `empty` in the channel histogram;
nothing to reconstruct.

### 1 stop — e.g. `ch0@100`

A single end of one axis. Neither a coordinate (needs both ends of an axis)
nor the time-sum (needs the other axis) can be formed. **Not
recoverable.** Counted as a 1-channel partial.

### 2 stops

**Same axis — e.g. `ch0@100`, `ch1@140`** (a complete x axis):

```
det_x = -0.5 × (140 - 100) × 0.0326531 × 0.1 = -0.06531 cm
det_y = NaN                       (y axis never fired)
tof   = (100 + 140) × TOF_FACTOR_NS_1D = 240 × 0.013716 = 3.292 ns
```

This is a **single-axis (1-D) partial**: `dlts = 2`,
`dlts_quality = recovered_x`, `y_det = NaN`. It is **not merged into the
DLD frame by default** (a `NaN` coordinate cannot be placed in the 3-D
reconstruction); set `include_one_d_partials=True` to keep it for the mass
spectrum, where it appears via a centred-axis `mc` estimate.

**Cross axis — e.g. `ch0@100`, `ch2@130`** (one end of x, one end of y):
no complete axis and no time-sum partner. **Not recoverable.**

### 3 stops — 3-of-4 (time-sum) recovery

A pulse that fired **one complete axis plus a single end of the other
axis** is reconstructable to a full `(x, y)` hit
(`_recover_three_channel_hits`). The complete axis gives the time-sum `S`,
which supplies the missing end of the other axis.

Example — `ch2@130`, `ch3@110` (complete y), plus `ch0@100` (one end of x):

```
S        = t_ch2 + t_ch3 = 240
t_ch1    = S - t_ch0 = 240 - 100 = 140   (reconstructed missing end)
det_x    = -0.5 × (140 - 100) × 0.0326531 × 0.1 = -0.06531 cm
det_y    = -0.5 × (110 - 130) × 0.0326531 × 0.1 =  0.03265 cm
tof      = S × TOF_FACTOR_NS_1D = 3.292 ns
```

Result: a full hit, `dlts = 4`, `dlts_quality = recovered_xy_3of4`. The
reconstruction is rejected if the recovered end is negative, the position
is off-detector, or the ToF is outside the window — so a wrong pairing is
not promoted. This is on by default in the merge path
(`recover_three_channel=True`). The SC firmware requires all four stops, so
these hits exist only after this offline step.

Patterns that **do not** recover: `ch0, ch1, ch2` with the *x* axis already
complete recovers the missing `ch3` (works); but `ch0, ch1` + `ch0` (a
duplicate end, no second axis) does not — there must be one complete axis
and exactly one orphan end of the *other* axis.

### 4 stops

**One stop per channel — e.g. `ch0@100, ch1@140, ch2@130, ch3@110`**: the
normal case. Both axes complete and the time-sum agrees, so this is a
**native full hit**:

```
det_x = -0.06531 cm,  det_y = 0.03265 cm,  tof = 480 × TOF_FACTOR_NS = 3.292 ns
```

In `/dld` this is `dlts = 4`, `dlts_quality = native`. No recovery needed.

**Two stops on the same axis — e.g. `ch0, ch0, ch1, ch1`** (two x pairs, no
y): two 1-D x partials (or none, if the pairs cannot be assigned within the
detector gate). Handled by the per-axis enumeration + bipartite matching in
`_recover_surface_concept_partial_hits`.

### 5–8+ stops — multi-hit pulses

When several ions strike within one pulse window the TDC records all their
stops together. PyCCAPT enumerates **every** `(ch0, ch1)` and `(ch2, ch3)`
pair, gates each by the detector radius and ToF window, runs a
maximum-cardinality 1-to-1 assignment per axis (each stop used at most
once), then cross-matches x-pairs to y-pairs by **ToF agreement**
(`axis_consistency_ns`). Matched pairs become full `(x, y)` hits; leftovers
fall back to 3-of-4 recovery and then to single-axis partials.

Example — 8 stops, two clean ions far apart in ToF:

```
ion A:  ch0@100  ch1@140  ch2@130  ch3@110   (tof ≈ 3.29 ns)
ion B:  ch0@2000 ch1@2060 ch2@2050 ch3@2010  (tof ≈ 55.7 ns)
```

The two `(ch0, ch1)` pairs and two `(ch2, ch3)` pairs are enumerated; the
ToF cross-match pairs A's x with A's y (ToFs agree) and B's x with B's y,
yielding **two full hits**. A's x is *not* stitched to B's y even though a
position match alone might allow it — the ToFs disagree, so the cross-match
rejects it.

A 20-stop pulse can yield, for instance, 3 full `xy` hits + 2 x-only
partials + 1 y-only partial — each emitted as its own row.

---

## 5. Outcome summary

| Stops | Pattern | Outcome | `dlts` | `dlts_quality` | Merged by default? |
|------:|---------|---------|:-----:|----------------|:------------------:|
| 0 | — | empty | — | — | no |
| 1 | any | not recoverable | — | — | no |
| 2 | one complete axis | 1-D partial | 2 | `recovered_x` / `recovered_y` | no¹ |
| 2 | one end of each axis | not recoverable | — | — | no |
| 3 | complete axis + 1 orphan end | full hit (time-sum) | 4 | `recovered_xy_3of4` | yes |
| 4 | one stop per channel | native full hit | 4 | `native` | (already in `/dld`) |
| 4 | two same-axis pairs | 1-D partial(s) | 2 | `recovered_x` / `recovered_y` | no¹ |
| 5–8+ | multi-hit | per-ion mix of the above | 4 / 2 | `recovered_xy`, `recovered_xy_3of4`, `recovered_x/y` | full hits yes¹ |

¹ Single-axis (`dlts == 2`) partials are excluded unless
`include_one_d_partials=True`; full `(x, y)` hits are always merged.

Recovered atoms are inserted at their **acquisition position** (right after
the native rows of the most recent preceding pulse), never by sorting on
`start_counter`, so the reconstructed z/depth sequence is preserved.

---

## 6. Residual recovery from *matched* multi-hit pulses

The cases above recover ions from **orphan** pulses (`has_dld_match ==
False`). A separate, opt-in step
(`recover_from_matched_multihit=True`) mines pulses the firmware **did**
match: because the firmware keeps only one hit per pulse, a *matched* pulse
that fired more stops than its DLD event used can hold a second ion.

The procedure (`partial_recovery._subtract_firmware_stops`):

1. Take the firmware's DLD event for the pulse — its recorded
   `(det_x, det_y, t)`.
2. Find a **coherent quadruplet** in the raw stops: a `(ch0, ch1)` pair
   within `multihit_match_tol_cm` of `det_x` **and** a `(ch2, ch3)` pair
   within `multihit_match_tol_cm` of `det_y`, whose two axis-ToFs agree
   (`axis_consistency_ns`) **and** whose combined ToF matches the firmware
   event's recorded `t (ns)` within `tof_match_tol_ns`.
3. Remove those four stops; run the residual through the normal recovery.

Why the ToF match matters: a detector coordinate depends only on the
per-axis time **difference**, so a second ion at a *similar position* but a
*different flight time* would otherwise be matched and removed, fabricating
a garbage residual or double-counting the firmware's own hit. Keying on the
firmware's recorded flight time (which differs between ions) prevents that.

It is opt-in and conservative: if **no** coherent quadruplet matches the
firmware event within tolerance, the pulse is **skipped**. In particular, a
firmware hit that was itself a 3-channel reconstruction has no full 4-stop
set in the raw stream, so it skips safely rather than stitching one ion's x
to another ion's y.

Worked example — a matched pulse with 8 stops whose DLD event is ion A:

```
DLD event (firmware):  det_x_A, det_y_A, tof_A ≈ 3.29 ns
raw stops:  ch0@100 ch1@140 ch2@130 ch3@110   (ion A)
            ch0@2000 ch1@2060 ch2@2050 ch3@2010 (ion B, tof ≈ 55.7 ns)
```

The quadruplet matching A's `(det_x, det_y, tof)` is removed; the residual
(ion B's four stops) reconstructs to a second full hit, tagged
`recovered_xy`. If the firmware event cannot be matched (e.g. its position
does not correspond to any stop pair, or it was a 3-channel hit), the pulse
is left untouched.

---

## 7. Match-quality cross-check (TDC vs DLD)

Because `/dld` and `/tdc` are captured separately and linked only by
`event_group_id`, it is worth verifying that the link is correct. The
diagnostic `partial_hit_diagnostics.matched_dld_tdc_residuals` does this by
**re-deriving** `det_x`, `det_y`, and `tof` from the raw stops and comparing
them to the firmware-recorded DLD values.

To keep the comparison an unambiguous ground truth, it is restricted to the
**clean case**: a pulse that fired exactly the four delay-line channels once
each (`ch0, ch1, ch2, ch3`), is flagged `has_dld_match`, and links to
exactly one DLD row. Multi-hit, unmatched, recovered, and ambiguous
(multi-row `event_group_id`) pulses are excluded.

Interpretation:

- **Near-zero residuals** (within one detector bin ≈ 0.0016 cm and one TDC
  bin ≈ 0.0069 ns) confirm both the `event_group_id` link and that the
  reconstruction formula matches the firmware's. On correctly matched data
  the residuals are numerically zero.
- **Large residuals** flag a mis-link: the stops associated with a DLD event
  are not the stops that produced it.

The check reports the mean / median / 99th-percentile / max of `|Δx_det|`,
`|Δy_det|`, `|Δtof|` and the fraction within tolerance. It is printed
automatically by `summarize_loaded_events` after the matched-pulse counts.

---

## 8. Running it

From the data-processing notebook, or directly:

```python
from pyccapt.calibration.tutorials.tutorials_helpers import helper_data_loader

helper_data_loader.load_data(
    dataset_path, max_mc, flightPathLength, pulse_mode, tdc='pyccapt',
    variables=variables,
    load_tdc_raw=True,                      # read and link the /tdc group
    # merge_partial_tdc defaults to True when load_tdc_raw is True (pyccapt h5)
    recover_from_matched_multihit=True,     # opt-in residual recovery (Section 6)
    multihit_match_tol_cm=0.1,
)

helper_data_loader.summarize_loaded_events(variables)
```

`summarize_loaded_events` prints the DLD/TDC counts, the per-channel-count
histogram, and the match-quality block from Section 7. Representative output
(the raw-TDC histogram and DLD/matched counts are from a real Al test
dataset; the recovered and cross-checked counts depend on the options you
enable):

```
Event statistics
================
  Complete events in DLD (atoms)          : 12,305,247
  Raw TDC pulses (triggers)               : 16,947,143
      fired 4 channel(s) (complete) : 12,081,535
      fired 3 channel(s)            : 2,187,190
      fired 2 channel(s)            : 1,658,884
      fired 1 channel(s)            : 1,019,527
  Complete pulses in raw TDC (4 ch)       : 12,081,535
  Partial pulses in raw TDC (<4 ch)       : 4,865,601
  Raw TDC pulses linked to a DLD event    : 12,197,571
  Match check (TDC-derived vs DLD recorded, clean 4-ch single-hit pulses):
      pulses cross-checked                : 11,9xx,xxx
      |dx_det| median / max (cm)          : 0.00e+00 / 0.00e+00
      |dy_det| median / max (cm)          : 0.00e+00 / 0.00e+00
      |dtof|   median / max (ns)          : 0.00e+00 / 0.00e+00
      within tol (x/y=1.63e-03 cm, tof=6.86e-03 ns)  : 100.0% / 100.0% / 100.0%
```

The zero residuals and 100 % within-tolerance shown here are the verified
behaviour for correctly matched pulses (the re-derived position and ToF are
identical to the firmware's because they use the same stops and formula);
non-zero values would indicate a problem with the link.

(The "Partial pulses in raw TDC" count being larger than the recovered-atom
count is expected: not every partial pulse is physically reconstructable,
and single-axis partials are excluded from the merge by default.)

---

## 9. Columns written into `/dld`

Recovery adds two columns to the DLD frame (full schema in
[Calibration_DATA_STRUCTURE.md](Calibration_DATA_STRUCTURE.md)):

- `dlts` (`int8`): `4` for a native or fully-recovered two-axis hit, `2`
  for a single-axis partial.
- `dlts_quality` (`str`): `native`, `recovered_xy`, `recovered_xy_3of4`,
  `recovered_x`, or `recovered_y`.

Downstream code that needs both detector axes filters `dlts == 4` (or drops
rows with `NaN` in `x_det (cm)` / `y_det (cm)`). ToF and mass-spectrum plots
can include the partials directly.

---

## 10. Function reference

| Purpose | Function |
|---------|----------|
| Merge recovered atoms into `/dld` | `data_tools.partial_recovery.merge_partial_tdc_into_dld` |
| Per-pulse multi-hit recovery | `_raw_workflow_surface_concept._recover_surface_concept_partial_hits` |
| 3-of-4 time-sum recovery | `_raw_workflow_surface_concept._recover_three_channel_hits` |
| Firmware-stop subtraction (matched multi-hit) | `partial_recovery._subtract_firmware_stops` |
| Position / hit from stops | `_surface_concept_position_from_pair`, `_surface_concept_hit_from_time_data` |
| Pulse completeness histogram | `partial_hit_diagnostics.tdc_pulse_completeness` |
| TDC↔DLD match cross-check | `partial_hit_diagnostics.matched_dld_tdc_residuals` |
| Load + summary | `helper_data_loader.load_data`, `helper_data_loader.summarize_loaded_events` |
