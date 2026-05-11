"""Notebook helper for Cameca RHIT/STR/HITS raw import workflows."""

from __future__ import annotations

import json
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.data_tools import file_dialog, raw_data_workflow
from pyccapt.calibration.leap_tools.cameca_raw import rhit_tools, str_tools

label_layout = widgets.Layout(width="220px")
field_layout = widgets.Layout(width="420px")


def _browse_file(text_widget: widgets.Text, out: Output, variables=None, file_kind: str = "cameca_raw") -> None:
    try:
        selected_path = file_dialog.choose_file_path(
            file_dialog.resolve_initial_directory(
                text_widget.value,
                getattr(variables, "last_directory", None) if variables is not None else None,
            ),
            file_kind=file_kind,
        )
        if selected_path:
            text_widget.value = selected_path
            if variables is not None:
                variables.last_directory = str(Path(selected_path).parent)
    except Exception as exc:
        with out:
            print(f"File chooser failed: {exc}")


def _path_row(label: str, text_widget: widgets.Text, button: widgets.Button) -> widgets.HBox:
    return widgets.HBox([widgets.Label(value=label, layout=label_layout), text_widget, button])


def _suggest_save_paths(input_path: str, processed_widget, raw_widget) -> None:
    """Populate processed/raw save fields with ``<basename>_pyccapt(.h5|-raw.h5)``.

    Only writes into a widget that is currently empty so a user-edited path is
    never overwritten. Called whenever the input RHIT/STR path changes.
    """
    if not input_path:
        return
    path = Path(input_path)
    if not path.name:
        return
    base = path.stem  # strips the file extension
    processed_default = str(path.with_name(f"{base}_pyccapt.h5"))
    raw_default = str(path.with_name(f"{base}_pyccapt-raw.h5"))
    if not processed_widget.value:
        processed_widget.value = processed_default
    if not raw_widget.value:
        raw_widget.value = raw_default


def _display_head_tail(dataframe: pd.DataFrame, edge_rows: int = 10) -> None:
    """Show a dataframe with both its first and last rows.

    Pandas truncates a dataframe larger than ``display.max_rows`` and shows
    ``max_rows // 2`` from the head and tail with the total ``N rows x M
    columns`` footer, so the user can immediately see the actual row count.
    """
    with pd.option_context("display.max_rows", 2 * edge_rows):
        display(dataframe)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _summarize_histograms(histograms):
    if not histograms:
        print("No stored histograms were found in the file.")
        return
    print("Stored histograms:")
    for name, histogram in histograms.items():
        if "values" in histogram:
            print(f"  {name}: {len(histogram['values']):,} bins")
        else:
            print(f"  {name}")


def _load_rhit_reference(path, epos_path, calibration_method: str = "metadata"):
    """Load a RHIT (and optionally apply a calibration).

    ``calibration_method`` is one of:

    * ``'metadata'`` (default, recommended): use the RHIT file's own
      ``flight_path_mm`` / ``t0_ns`` / ``kf`` to compute mc. No EPOS needed.
    * ``'pmass_match'``: spectrum-match against the Cameca pMass histogram
      stored inside the RHIT — IVAS's own ground-truth spectrum, no EPOS needed.
    * ``'spectrum_match'``: spectrum-match against the EPOS mc histogram.
      Robust to events not being 1:1 aligned.
    * ``'event_match'``: MATLAB-style per-event VDC + position matcher (strict).
    """
    hits, histograms, metadata = rhit_tools.rhit_load(path)
    calibration = None
    method = (calibration_method or "metadata").lower()
    if method == "metadata":
        if epos_path:
            print("  EPOS path provided but calibration method='metadata' -- ignoring EPOS.")
    elif method == "pmass_match":
        hits, calibration = rhit_tools.rhit_calibrate_from_epos(
            hits, epos=None, method="pmass_match",
            metadata=metadata, rhit_histograms=histograms,
        )
    elif method in {"spectrum_match", "event_match"}:
        if not epos_path:
            raise ValueError(
                f"calibration mode '{method}' needs an EPOS file path. "
                "Either provide one or switch to 'metadata' / 'pmass_match'."
            )
        hits, calibration = rhit_tools.rhit_calibrate_from_epos(
            hits, epos_path, method=method, metadata=metadata, rhit_histograms=histograms,
        )
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")
    return hits, histograms, metadata, calibration


def call_cameca_raw_import_workflow(variables=None):
    """Display a widget workflow for RHIT and STR/HITS imports.

    Each tab exposes three clear export paths for the loaded events:

      * Inspect the raw pandas DataFrame in-place (head + tail with row count).
      * "Save PyCCAPT processed HDF5" -> writes a fully processed dataset
        with PyCCAPT standard columns (mc, t, high_voltage, x_det, y_det, ...).
      * "Save raw analysis HDF5" -> writes a ``dld/``-group HDF5 directly
        consumable by the PyCCAPT raw-data analysis workflow.
    """
    out = Output()
    state = {
        "rhit_hits": None,
        "rhit_histograms": None,
        "rhit_metadata": None,
        "rhit_calibration": None,
        "str_hits": None,
        "str_metadata": None,
        "str_calibration": None,
    }

    # -------------------------- RHIT tab widgets --------------------------
    rhit_path = widgets.Text(value="", description="", layout=field_layout)
    rhit_epos_path = widgets.Text(value="", description="", layout=field_layout)
    rhit_save_path = widgets.Text(value="", description="", layout=field_layout)
    rhit_raw_hdf5_path = widgets.Text(value="", description="", layout=field_layout)
    rhit_calibration_path = widgets.Text(value="", description="", layout=field_layout)
    rhit_path_browse = widgets.Button(description="Browse")
    rhit_epos_path_browse = widgets.Button(description="Browse")
    rhit_save_path_browse = widgets.Button(description="Browse")
    rhit_raw_hdf5_path_browse = widgets.Button(description="Browse")
    rhit_calibration_path_browse = widgets.Button(description="Browse")
    rhit_load_button = widgets.Button(description="Load RHIT")
    rhit_export_button = widgets.Button(description="Save PyCCAPT processed")
    rhit_export_raw_button = widgets.Button(description="Save raw analysis HDF5")
    rhit_save_calibration_button = widgets.Button(description="Save RHIT calibration JSON")
    rhit_load_into_variables = widgets.Dropdown(
        options=[("True", True), ("False", False)],
        value=variables is not None,
        description="Load data:",
    )
    rhit_pulse_mode = widgets.Dropdown(
        options=[("voltage", "voltage"), ("laser", "laser")],
        value="voltage",
        description="Pulse mode:",
    )
    rhit_calibration_method = widgets.Dropdown(
        options=[
            ("Metadata only (recommended; no EPOS needed)", "metadata"),
            ("RHIT pMass match (no EPOS needed; uses stored Cameca spectrum)", "pmass_match"),
            ("EPOS spectrum match (robust; needs EPOS)", "spectrum_match"),
            ("EPOS event match (MATLAB-style; strict)", "event_match"),
        ],
        value="metadata",
        description="Calib mode:",
        layout=widgets.Layout(width="500px"),
    )

    # -------------------------- STR tab widgets --------------------------
    str_path = widgets.Text(value="", description="", layout=field_layout)
    str_rhit_path = widgets.Text(value="", description="", layout=field_layout)
    str_epos_path = widgets.Text(value="", description="", layout=field_layout)
    str_save_path = widgets.Text(value="", description="", layout=field_layout)
    str_raw_hdf5_path = widgets.Text(value="", description="", layout=field_layout)
    str_path_browse = widgets.Button(description="Browse")
    str_rhit_path_browse = widgets.Button(description="Browse")
    str_epos_path_browse = widgets.Button(description="Browse")
    str_save_path_browse = widgets.Button(description="Browse")
    str_raw_hdf5_path_browse = widgets.Button(description="Browse")
    str_load_button = widgets.Button(description="Load STR/HITS")
    str_calibrate_button = widgets.Button(description="Calibrate from RHIT (slow)")
    str_export_button = widgets.Button(description="Save PyCCAPT processed")
    str_export_raw_button = widgets.Button(description="Save raw analysis HDF5")
    str_load_into_variables = widgets.Dropdown(
        options=[("True", True), ("False", False)],
        value=variables is not None,
        description="Load data:",
    )
    str_pulse_mode = widgets.Dropdown(
        options=[("voltage", "voltage"), ("laser", "laser")],
        value="voltage",
        description="Pulse mode:",
    )
    str_calibration_method = widgets.Dropdown(
        options=[
            ("Metadata only (recommended; no EPOS needed)", "metadata"),
            ("RHIT pMass match (no EPOS needed; uses stored Cameca spectrum)", "pmass_match"),
            ("EPOS spectrum match (robust; needs EPOS)", "spectrum_match"),
            ("EPOS event match (MATLAB-style; strict)", "event_match"),
        ],
        value="metadata",
        description="Calib mode:",
        layout=widgets.Layout(width="500px"),
    )

    # ------------------------- RHIT button handlers -----------------------
    def on_load_rhit(_):
        rhit_load_button.disabled = True
        with out:
            out.clear_output()
            try:
                hits, histograms, metadata, calibration = _load_rhit_reference(
                    rhit_path.value, rhit_epos_path.value, rhit_calibration_method.value
                )
                state["rhit_hits"] = hits
                state["rhit_histograms"] = histograms
                state["rhit_metadata"] = metadata
                state["rhit_calibration"] = calibration
                print(f"Loaded {len(hits):,} RHIT hits.")
                tree_entries = metadata.get("numHits")
                if isinstance(tree_entries, int) and tree_entries != len(hits):
                    print(
                        f"  RHIT 'nth' tree reports {tree_entries:,} pulse entries; "
                        f"{len(hits):,} of those produced an ion hit."
                    )
                if calibration is not None:
                    print("Applied RHIT calibration from the matching EPOS file.")
                    print(
                        f"  t_offset={calibration['t_offset']:.4f} ns, "
                        f"matched_events={calibration['matched_events']:,}, "
                        f"residual_std={calibration['residual_std']:.4e}, "
                        f"ICF={calibration['ICF']:.4f}"
                    )
                    if calibration["matched_events"] < 1000:
                        print(
                            "  [WARN] very few matched events -- the calibration is unreliable. "
                            "The default RHIT mass-charge formula will give a better answer than this fit."
                        )
                params = metadata.get("instrumentParams", {})
                if params:
                    print("Instrument parameters:")
                    for key in ("flight_path_mm", "t0_ns", "detector_halfsize_mm", "ivas_version"):
                        if key in params:
                            print(f"  {key}: {params[key]}")
                _summarize_histograms(histograms)
                print("\nRaw RHIT events (pandas DataFrame, head + tail):")
                _display_head_tail(hits)
            except Exception as exc:
                print(f"RHIT load failed: {exc}")
        rhit_load_button.disabled = False

    def on_export_rhit_processed(_):
        rhit_export_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["rhit_hits"] is None:
                    raise ValueError("Load a RHIT file first.")
                dataset = rhit_tools.rhit_to_ccapt(state["rhit_hits"])
                print(f"Prepared {len(dataset):,} RHIT events as a PyCCAPT processed dataset.")
                if rhit_save_path.value:
                    raw_data_workflow.save_processed_raw_dataset(dataset, rhit_save_path.value)
                    print(f"Saved processed dataset to: {rhit_save_path.value}")
                else:
                    print("(No 'Save processed HDF5' path provided — dataset prepared in memory only.)")
                if variables is not None and rhit_load_into_variables.value:
                    variables.sync_from_data(dataset, update_backups=True)
                    print("Loaded the RHIT dataset into the active workflow variables.")
                print("\nPyCCAPT processed (head + tail):")
                _display_head_tail(dataset)
            except Exception as exc:
                print(f"RHIT processed export failed: {exc}")
        rhit_export_button.disabled = False

    def on_export_rhit_raw(_):
        rhit_export_raw_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["rhit_hits"] is None:
                    raise ValueError("Load a RHIT file first.")
                if not rhit_raw_hdf5_path.value:
                    raise ValueError(
                        "Provide an output path in 'Save raw analysis HDF5' before exporting."
                    )
                output = rhit_tools.rhit_to_raw_hdf5(
                    state["rhit_hits"], rhit_raw_hdf5_path.value, pulse_mode=rhit_pulse_mode.value
                )
                print(
                    f"Saved raw-analysis HDF5 to:\n  {output}"
                )
                print(
                    "  Contains both 'dld/' (extract_mode='dld') and 'tdc/' (extract_mode='tdc_sc')\n"
                    "  groups so it loads directly into the PyCCAPT raw-data analysis workflow.\n"
                    "  Note: the tdc/ group is a single-channel mirror because RHIT files do not\n"
                    "  carry per-DL-end TDC counts."
                )
            except Exception as exc:
                print(f"RHIT raw-HDF5 export failed: {exc}")
        rhit_export_raw_button.disabled = False

    def on_save_rhit_calibration(_):
        rhit_save_calibration_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["rhit_calibration"] is None:
                    raise ValueError("No RHIT calibration is available yet.")
                if not rhit_calibration_path.value:
                    raise ValueError("Provide a JSON output path for the RHIT calibration.")
                with open(rhit_calibration_path.value, "w", encoding="utf-8") as handle:
                    json.dump(_json_ready(state["rhit_calibration"]), handle, indent=2)
                print(f"Saved RHIT calibration to: {rhit_calibration_path.value}")
            except Exception as exc:
                print(f"Saving RHIT calibration failed: {exc}")
        rhit_save_calibration_button.disabled = False

    # ------------------------- STR button handlers ------------------------
    def on_load_str(_):
        str_load_button.disabled = True
        with out:
            out.clear_output()
            try:
                hits, metadata = str_tools.str_load(str_path.value)
                hits = str_tools.str_calculate_positions(hits)
                state["str_hits"] = hits
                state["str_metadata"] = metadata
                state["str_calibration"] = None
                print(f"\nLoaded {len(hits):,} STR/HITS events (raw delay-line + computed TDC positions).")
                if "nFull6Channels" in metadata:
                    print(f"Events with all 6 channels: {metadata['nFull6Channels']:,}")
                print(
                    "STR data is in raw TDC counts at this point. To get mm/ns/Da columns, "
                    "click 'Calibrate from RHIT' below."
                )
                if metadata.get("note"):
                    print(metadata["note"])
                print("\nRaw STR events (pandas DataFrame, head + tail):")
                _display_head_tail(hits)
            except Exception as exc:
                print(f"STR/HITS load failed: {exc}")
        str_load_button.disabled = False

    def on_calibrate_str_from_rhit(_):
        str_calibrate_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["str_hits"] is None:
                    raise ValueError("Load a STR/HITS file first.")
                rhit_hits = state["rhit_hits"]
                rhit_histograms = state["rhit_histograms"]
                rhit_metadata = state["rhit_metadata"]
                if str_rhit_path.value:
                    print(f"Loading the matching RHIT from {str_rhit_path.value} ...")
                    rhit_hits, rhit_histograms, rhit_metadata, _ = _load_rhit_reference(
                        str_rhit_path.value, str_epos_path.value, str_calibration_method.value
                    )
                if rhit_hits is None:
                    raise ValueError(
                        "STR calibration requires a matching RHIT — either load one in the RHIT "
                        "tab first or fill in 'Matching RHIT path' here."
                    )
                print(
                    "Running multi-start clock/t0 optimization on the STR mass spectrum vs RHIT — "
                    "this can take a minute or two on multi-million-row files."
                )
                hits, calibration = str_tools.str_calibrate_from_rhit(
                    state["str_hits"], rhit_hits, rhit_histograms, rhit_metadata
                )
                state["str_hits"] = hits
                state["str_calibration"] = calibration
                print(
                    f"Calibrated STR from RHIT: clock={calibration['clock_ns'] * 1000:.2f} ps, "
                    f"t0={calibration['t0_tdc']:.1f} TDC, corr={calibration['spectrum_correlation']:.4f}"
                )
                detector_scale = calibration["detector_scale"]
                print(
                    f"Estimated detector scale: x={detector_scale['x_scale']:.5f} mm/TDC, "
                    f"y={detector_scale['y_scale']:.5f} mm/TDC"
                )
                print("\nCalibrated STR events (head + tail):")
                _display_head_tail(hits)
            except Exception as exc:
                print(f"STR/HITS calibration failed: {exc}")
        str_calibrate_button.disabled = False

    def on_export_str_processed(_):
        str_export_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["str_hits"] is None:
                    raise ValueError("Load a STR/HITS file first.")
                dataset = str_tools.str_to_ccapt(state["str_hits"])
                print(f"Prepared {len(dataset):,} STR/HITS events as a PyCCAPT processed dataset.")
                if str_save_path.value:
                    raw_data_workflow.save_processed_raw_dataset(dataset, str_save_path.value)
                    print(f"Saved processed dataset to: {str_save_path.value}")
                else:
                    print("(No 'Save processed HDF5' path provided — dataset prepared in memory only.)")
                if variables is not None and str_load_into_variables.value:
                    variables.sync_from_data(dataset, update_backups=True)
                    print("Loaded the STR dataset into the active workflow variables.")
                print("\nPyCCAPT processed (head + tail):")
                _display_head_tail(dataset)
            except Exception as exc:
                print(f"STR/HITS processed export failed: {exc}")
        str_export_button.disabled = False

    def on_export_str_raw(_):
        str_export_raw_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["str_hits"] is None:
                    raise ValueError("Load a STR/HITS file first.")
                if not str_raw_hdf5_path.value:
                    raise ValueError(
                        "Provide an output path in 'Save raw analysis HDF5' before exporting."
                    )
                output = str_tools.str_to_raw_hdf5(
                    state["str_hits"], str_raw_hdf5_path.value, pulse_mode=str_pulse_mode.value
                )
                print(
                    f"Saved raw-analysis HDF5 to:\n  {output}"
                )
                print(
                    "  Contains:\n"
                    "    - 'dld/' (loads with extract_mode='dld') -- per-event calibrated x/y/t,\n"
                    "      only present when the STR has been calibrated against a RHIT.\n"
                    "    - 'tdc/' (loads with extract_mode='tdc_ro' or 'tdc_sc') -- 6-channel\n"
                    "      delay-line TDC counts: 0=detxt1, 1=detxt2, 2=detyt1, 3=detyt2,\n"
                    "      4=detwt1, 5=detwt2."
                )
            except Exception as exc:
                print(f"STR/HITS raw-HDF5 export failed: {exc}")
        str_export_raw_button.disabled = False

    # --------------------------- click bindings ---------------------------
    rhit_load_button.on_click(on_load_rhit)
    rhit_export_button.on_click(on_export_rhit_processed)
    rhit_export_raw_button.on_click(on_export_rhit_raw)
    rhit_save_calibration_button.on_click(on_save_rhit_calibration)

    str_load_button.on_click(on_load_str)
    str_calibrate_button.on_click(on_calibrate_str_from_rhit)
    str_export_button.on_click(on_export_str_processed)
    str_export_raw_button.on_click(on_export_str_raw)

    # ----- Auto-fill the save paths from the input file path ----------------
    def _on_rhit_path_change(change):
        if change["name"] == "value":
            _suggest_save_paths(change["new"], rhit_save_path, rhit_raw_hdf5_path)

    def _on_str_path_change(change):
        if change["name"] == "value":
            _suggest_save_paths(change["new"], str_save_path, str_raw_hdf5_path)

    rhit_path.observe(_on_rhit_path_change, names="value")
    str_path.observe(_on_str_path_change, names="value")

    rhit_path_browse.on_click(lambda _: _browse_file(rhit_path, out, variables))
    rhit_epos_path_browse.on_click(lambda _: _browse_file(rhit_epos_path, out, variables))
    rhit_save_path_browse.on_click(lambda _: _browse_file(rhit_save_path, out, variables))
    rhit_raw_hdf5_path_browse.on_click(lambda _: _browse_file(rhit_raw_hdf5_path, out, variables))
    rhit_calibration_path_browse.on_click(lambda _: _browse_file(rhit_calibration_path, out, variables))
    str_path_browse.on_click(lambda _: _browse_file(str_path, out, variables))
    str_rhit_path_browse.on_click(lambda _: _browse_file(str_rhit_path, out, variables))
    str_epos_path_browse.on_click(lambda _: _browse_file(str_epos_path, out, variables))
    str_save_path_browse.on_click(lambda _: _browse_file(str_save_path, out, variables))
    str_raw_hdf5_path_browse.on_click(lambda _: _browse_file(str_raw_hdf5_path, out, variables))

    tabs = widgets.Tab(
        [
            widgets.VBox(
                [
                    _path_row("RHIT file path:", rhit_path, rhit_path_browse),
                    _path_row("Matching EPOS path:", rhit_epos_path, rhit_epos_path_browse),
                    _path_row("Save processed HDF5:", rhit_save_path, rhit_save_path_browse),
                    _path_row("Save raw analysis HDF5:", rhit_raw_hdf5_path, rhit_raw_hdf5_path_browse),
                    _path_row("Save calibration JSON:", rhit_calibration_path, rhit_calibration_path_browse),
                    widgets.HBox([widgets.Label(value="Load into variables:", layout=label_layout), rhit_load_into_variables]),
                    widgets.HBox([widgets.Label(value="Pulse mode:", layout=label_layout), rhit_pulse_mode]),
                    widgets.HBox([widgets.Label(value="EPOS calibration mode:", layout=label_layout), rhit_calibration_method]),
                    widgets.HBox(
                        [rhit_load_button, rhit_export_button, rhit_export_raw_button, rhit_save_calibration_button]
                    ),
                ]
            ),
            widgets.VBox(
                [
                    _path_row("STR/HITS file path:", str_path, str_path_browse),
                    _path_row("Matching RHIT path:", str_rhit_path, str_rhit_path_browse),
                    _path_row("RHIT matching EPOS:", str_epos_path, str_epos_path_browse),
                    _path_row("Save processed HDF5:", str_save_path, str_save_path_browse),
                    _path_row("Save raw analysis HDF5:", str_raw_hdf5_path, str_raw_hdf5_path_browse),
                    widgets.HBox([widgets.Label(value="Load into variables:", layout=label_layout), str_load_into_variables]),
                    widgets.HBox([widgets.Label(value="Pulse mode:", layout=label_layout), str_pulse_mode]),
                    widgets.HBox([widgets.Label(value="EPOS calibration mode:", layout=label_layout), str_calibration_method]),
                    widgets.HBox(
                        [str_load_button, str_calibrate_button, str_export_button, str_export_raw_button]
                    ),
                ]
            ),
        ]
    )
    tabs.set_title(0, "RHIT")
    tabs.set_title(1, "STR / HITS")

    display(tabs)
    display(out)
