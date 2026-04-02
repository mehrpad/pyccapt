"""Notebook helper for Cameca RHIT/STR/HITS raw import workflows."""

from __future__ import annotations

import json

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.data_tools import raw_data_workflow
from pyccapt.calibration.leap_tools.cameca_raw import rhit_tools, str_tools

label_layout = widgets.Layout(width="220px")


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


def _load_rhit_reference(path, epos_path):
    hits, histograms, metadata = rhit_tools.rhit_load(path)
    calibration = None
    if epos_path:
        hits, calibration = rhit_tools.rhit_calibrate_from_epos(hits, epos_path)
    return hits, histograms, metadata, calibration


def call_cameca_raw_import_workflow(variables=None):
    """Display a widget workflow for RHIT and STR/HITS imports."""
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

    rhit_path = widgets.Text(value="", description="RHIT path:")
    rhit_epos_path = widgets.Text(value="", description="match EPOS:")
    rhit_save_path = widgets.Text(value="", description="save dataset:")
    rhit_calibration_path = widgets.Text(value="", description="save calib:")
    rhit_load_button = widgets.Button(description="Load RHIT")
    rhit_export_button = widgets.Button(description="Export RHIT dataset")
    rhit_save_calibration_button = widgets.Button(description="Save RHIT calibration")
    rhit_load_into_variables = widgets.Dropdown(
        options=[("True", True), ("False", False)],
        value=variables is not None,
        description="load data:",
    )

    str_path = widgets.Text(value="", description="STR/HITS path:")
    str_rhit_path = widgets.Text(value="", description="match RHIT:")
    str_epos_path = widgets.Text(value="", description="RHIT EPOS:")
    str_save_path = widgets.Text(value="", description="save dataset:")
    str_load_button = widgets.Button(description="Load/process STR")
    str_export_button = widgets.Button(description="Export STR dataset")
    str_load_into_variables = widgets.Dropdown(
        options=[("True", True), ("False", False)],
        value=variables is not None,
        description="load data:",
    )

    def on_load_rhit(_):
        rhit_load_button.disabled = True
        with out:
            out.clear_output()
            try:
                hits, histograms, metadata, calibration = _load_rhit_reference(rhit_path.value, rhit_epos_path.value)
                state["rhit_hits"] = hits
                state["rhit_histograms"] = histograms
                state["rhit_metadata"] = metadata
                state["rhit_calibration"] = calibration
                print(f"Loaded {len(hits):,} RHIT hits.")
                if calibration is not None:
                    print("Applied RHIT calibration from the matching EPOS file.")
                    print(
                        f"t_offset={calibration['t_offset']:.4f} ns, "
                        f"matched_events={calibration['matched_events']:,}, "
                        f"residual_std={calibration['residual_std']:.4e}"
                    )
                params = metadata.get("instrumentParams", {})
                if params:
                    print("Instrument parameters:")
                    for key in ("flight_path_mm", "t0_ns", "detector_halfsize_mm", "ivas_version"):
                        if key in params:
                            print(f"  {key}: {params[key]}")
                _summarize_histograms(histograms)
                display(hits.head(20))
            except Exception as exc:
                print(f"RHIT load failed: {exc}")
        rhit_load_button.disabled = False

    def on_export_rhit(_):
        rhit_export_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["rhit_hits"] is None:
                    raise ValueError("Load a RHIT file first.")
                dataset = rhit_tools.rhit_to_ccapt(state["rhit_hits"])
                print(f"Prepared {len(dataset):,} RHIT events as a PyCCAPT-style dataset.")
                if rhit_save_path.value:
                    raw_data_workflow.save_processed_raw_dataset(dataset, rhit_save_path.value)
                    print(f"Saved dataset to: {rhit_save_path.value}")
                if variables is not None and rhit_load_into_variables.value:
                    variables.sync_from_data(dataset, update_backups=True)
                    print("Loaded the RHIT dataset into the active workflow variables.")
                display(dataset.head(20))
            except Exception as exc:
                print(f"RHIT export failed: {exc}")
        rhit_export_button.disabled = False

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

    def on_load_str(_):
        str_load_button.disabled = True
        with out:
            out.clear_output()
            try:
                hits, metadata = str_tools.str_load(str_path.value)
                hits = str_tools.str_calculate_positions(hits)
                calibration = None
                rhit_hits = state["rhit_hits"]
                rhit_histograms = state["rhit_histograms"]
                rhit_metadata = state["rhit_metadata"]

                if str_rhit_path.value:
                    rhit_hits, rhit_histograms, rhit_metadata, _ = _load_rhit_reference(str_rhit_path.value, str_epos_path.value)

                if rhit_hits is not None:
                    hits, calibration = str_tools.str_calibrate_from_rhit(hits, rhit_hits, rhit_histograms, rhit_metadata)

                state["str_hits"] = hits
                state["str_metadata"] = metadata
                state["str_calibration"] = calibration

                print(f"Loaded {len(hits):,} STR/HITS events.")
                print(f"Decoded format version: {metadata.get('version', 'unknown')}")
                if "nFull6Channels" in metadata:
                    print(f"Events with all 6 channels: {metadata['nFull6Channels']:,}")
                if calibration is not None:
                    detector_scale = calibration["detector_scale"]
                    print(
                        f"Calibrated STR from RHIT: clock={calibration['clock_ns'] * 1000:.2f} ps, "
                        f"t0={calibration['t0_tdc']:.1f} TDC, "
                        f"corr={calibration['spectrum_correlation']:.4f}"
                    )
                    print(
                        f"Estimated detector scale: "
                        f"x={detector_scale['x_scale']:.5f} mm/TDC, "
                        f"y={detector_scale['y_scale']:.5f} mm/TDC"
                    )
                elif metadata.get("note"):
                    print(metadata["note"])
                display(hits.head(20))
            except Exception as exc:
                print(f"STR/HITS processing failed: {exc}")
        str_load_button.disabled = False

    def on_export_str(_):
        str_export_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["str_hits"] is None:
                    raise ValueError("Load a STR/HITS file first.")
                dataset = str_tools.str_to_ccapt(state["str_hits"])
                print(f"Prepared {len(dataset):,} STR/HITS events as a PyCCAPT-style dataset.")
                if str_save_path.value:
                    raw_data_workflow.save_processed_raw_dataset(dataset, str_save_path.value)
                    print(f"Saved dataset to: {str_save_path.value}")
                if variables is not None and str_load_into_variables.value:
                    variables.sync_from_data(dataset, update_backups=True)
                    print("Loaded the STR/HITS dataset into the active workflow variables.")
                display(dataset.head(20))
            except Exception as exc:
                print(f"STR/HITS export failed: {exc}")
        str_export_button.disabled = False

    rhit_load_button.on_click(on_load_rhit)
    rhit_export_button.on_click(on_export_rhit)
    rhit_save_calibration_button.on_click(on_save_rhit_calibration)
    str_load_button.on_click(on_load_str)
    str_export_button.on_click(on_export_str)

    tabs = widgets.Tab(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="RHIT file path:", layout=label_layout), rhit_path]),
                    widgets.HBox([widgets.Label(value="Matching EPOS path:", layout=label_layout), rhit_epos_path]),
                    widgets.HBox([widgets.Label(value="Save processed HDF5:", layout=label_layout), rhit_save_path]),
                    widgets.HBox([widgets.Label(value="Save calibration JSON:", layout=label_layout), rhit_calibration_path]),
                    widgets.HBox([widgets.Label(value="Load into variables:", layout=label_layout), rhit_load_into_variables]),
                    widgets.HBox([rhit_load_button, rhit_export_button, rhit_save_calibration_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="STR/HITS file path:", layout=label_layout), str_path]),
                    widgets.HBox([widgets.Label(value="Matching RHIT path:", layout=label_layout), str_rhit_path]),
                    widgets.HBox([widgets.Label(value="RHIT matching EPOS:", layout=label_layout), str_epos_path]),
                    widgets.HBox([widgets.Label(value="Save processed HDF5:", layout=label_layout), str_save_path]),
                    widgets.HBox([widgets.Label(value="Load into variables:", layout=label_layout), str_load_into_variables]),
                    widgets.HBox([str_load_button, str_export_button]),
                ]
            ),
        ]
    )
    tabs.set_title(0, "RHIT")
    tabs.set_title(1, "STR / HITS")

    display(tabs)
    display(out)
