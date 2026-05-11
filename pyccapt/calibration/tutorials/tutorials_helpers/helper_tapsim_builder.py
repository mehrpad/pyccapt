"""Widget workflow for creating TAPSim-ready synthetic specimens."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.reconstructions import tapsim_builder


label_layout = widgets.Layout(width="240px")


def _pretty(value: object) -> str:
    return json.dumps(value, indent=2)


def _parse_mapping(text: str, field_name: str) -> dict[str, object]:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary-like value")
    return dict(value)


def call_tapsim_node_workflow():
    """Display the TAPSim specimen/node-file workflow."""
    out = Output()
    state: dict[str, object] = {"result": None}

    preset = widgets.Dropdown(options=tapsim_builder.available_presets(), value="Nimonic 90")
    load_preset_button = widgets.Button(description="Load preset")
    mode = widgets.Dropdown(
        options=[("gamma-gamma-prime", "gamma-gamma-prime"), ("single-phase", "single-phase")],
        value="gamma-gamma-prime",
    )
    geometry_mode = widgets.Dropdown(
        options=[("tapered tip", "tapered-tip"), ("cylinder + hemisphere", "cylinder-hemisphere"), ("bulk", "bulk")],
        value="tapered-tip",
    )
    a_gamma = widgets.FloatText(value=3.54)
    a_gprime = widgets.FloatText(value=3.57)
    supercell_x = widgets.BoundedIntText(value=24, min=1, max=500)
    supercell_y = widgets.BoundedIntText(value=24, min=1, max=500)
    supercell_z = widgets.BoundedIntText(value=36, min=1, max=1000)
    single_phase_composition = widgets.Textarea(value='{"Ni": 1.0}', layout=widgets.Layout(width="500px", height="80px"))
    wt_spec = widgets.Textarea(value="{}", layout=widgets.Layout(width="500px", height="150px"))
    gamma_partition = widgets.Textarea(value="{}", layout=widgets.Layout(width="500px", height="120px"))
    gprime_a_site_ratio = widgets.Textarea(value='{"Al": 0.6, "Ti": 0.4}', layout=widgets.Layout(width="500px", height="80px"))
    n_precip = widgets.BoundedIntText(value=10, min=0, max=500)
    r_min = widgets.FloatText(value=6.0)
    r_max = widgets.FloatText(value=18.0)
    target_gprime_fraction = widgets.FloatText(value=0.18)
    clearance = widgets.FloatText(value=1.0)
    margin = widgets.FloatText(value=2.0)
    periodic_precipitates = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    radius_dist = widgets.Dropdown(options=[("uniform", "uniform"), ("lognormal", "lognormal")], value="uniform")
    lognorm_sigma = widgets.FloatText(value=0.35)
    remove_cut = widgets.FloatText(value=2.3)
    cone_diameter = widgets.FloatText(value=140.0)
    cone_length = widgets.FloatText(value=600.0)
    hemisphere_base = widgets.FloatText(value=220.0)
    surf_tol = widgets.FloatText(value=2.0)
    bottom_tol = widgets.FloatText(value=2.0)
    elem_start_id = widgets.BoundedIntText(value=10, min=4, max=31)
    max_id = widgets.BoundedIntText(value=31, min=10, max=64)
    separate_phase_ids = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    seed = widgets.IntText(value=42)
    output_dir = widgets.Text(value=str(Path.cwd() / "tapsim_outputs"), layout=widgets.Layout(width="500px"))
    dataset_name = widgets.Text(value="nimonic90_tapsim")
    include_potential = widgets.Dropdown(options=[("False", False), ("True", True)], value=False)
    potential_start = widgets.FloatText(value=1000.0)
    potential_end = widgets.FloatText(value=2000.0)
    write_csv = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    write_hdf = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    write_range_hdf = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    preview_button = widgets.Button(description="Preview specimen")
    export_button = widgets.Button(description="Export TAPSim files")

    def apply_preset(_=None):
        config = tapsim_builder.get_preset_config(preset.value)
        mode.value = config.get("mode", mode.value)
        geometry_mode.value = config.get("geometry_mode", geometry_mode.value)
        a_gamma.value = float(config.get("a_gamma", a_gamma.value))
        a_gprime.value = float(config.get("a_gprime", a_gprime.value))
        supercell = config.get("supercell", (supercell_x.value, supercell_y.value, supercell_z.value))
        supercell_x.value, supercell_y.value, supercell_z.value = map(int, supercell)
        single_phase_composition.value = _pretty(config.get("single_phase_composition", {"Ni": 1.0}))
        wt_spec.value = _pretty(config.get("wt_spec", {}))
        gamma_partition.value = _pretty(config.get("gamma_partition", {}))
        gprime_a_site_ratio.value = _pretty(config.get("gprime_a_site_ratio", {"Al": 0.6, "Ti": 0.4}))
        n_precip.value = int(config.get("n_precip", 0))
        r_min.value = float(config.get("precipitate_r_min_angstrom", r_min.value))
        r_max.value = float(config.get("precipitate_r_max_angstrom", r_max.value))
        target_gprime_fraction.value = float(config.get("target_gprime_fraction", 0.0))
        cone_diameter.value = float(config.get("cone_diameter_angstrom", cone_diameter.value))
        cone_length.value = float(config.get("cone_length_angstrom", cone_length.value))
        hemisphere_base.value = float(config.get("hemisphere_base_angstrom", hemisphere_base.value))
        surf_tol.value = float(config.get("surf_tol_angstrom", surf_tol.value))
        bottom_tol.value = float(config.get("bottom_tol_angstrom", bottom_tol.value))
        dataset_name.value = preset.value.lower().replace(" ", "_") + "_tapsim"

    def build_result():
        return tapsim_builder.build_tapsim_specimen(
            preset=preset.value,
            mode=mode.value,
            geometry_mode=geometry_mode.value,
            single_phase_composition=_parse_mapping(single_phase_composition.value, "single-phase composition"),
            a_gamma=a_gamma.value,
            a_gprime=a_gprime.value,
            supercell_size=(supercell_x.value, supercell_y.value, supercell_z.value),
            wt_spec=_parse_mapping(wt_spec.value, "weight-percent specification") if wt_spec.value.strip() else {},
            gamma_partition=_parse_mapping(gamma_partition.value, "gamma partition") if gamma_partition.value.strip() else {},
            gprime_a_site_ratio=_parse_mapping(gprime_a_site_ratio.value, "gamma-prime A-site ratio"),
            n_precip=n_precip.value,
            precipitate_r_min_angstrom=r_min.value,
            precipitate_r_max_angstrom=r_max.value,
            target_gprime_fraction=target_gprime_fraction.value if target_gprime_fraction.value > 0 else None,
            clearance_angstrom=clearance.value,
            margin_angstrom=margin.value,
            periodic_precipitates=periodic_precipitates.value,
            radius_dist=radius_dist.value,
            lognorm_sigma=lognorm_sigma.value,
            remove_cut_angstrom=remove_cut.value,
            cone_diameter_angstrom=cone_diameter.value,
            cone_length_angstrom=cone_length.value,
            hemisphere_base_angstrom=hemisphere_base.value,
            surf_tol_angstrom=surf_tol.value,
            bottom_tol_angstrom=bottom_tol.value,
            elem_start_id=elem_start_id.value,
            max_id=max_id.value,
            separate_phase_ids=separate_phase_ids.value,
            seed=seed.value,
        )

    def render_result(result):
        print("=" * 60)
        print("TAPSIM SPECIMEN SUMMARY")
        print("=" * 60)
        print(f"Preset: {result.preset}")
        print(f"Mode: {result.mode}")
        print(f"Geometry: {result.geometry_mode}")
        print(f"Atoms before shaping: {result.summary['total_atoms_phase_structure']:,}")
        print(f"Atoms after shaping: {result.summary['total_atoms_shaped']:,}")
        print(f"Precipitates: {result.summary['precipitate_count']}")
        print(f"Achieved gprime fraction: {result.summary['achieved_gprime_fraction']:.4f}")
        print(f"Used TAPSim IDs: {result.summary['used_ids']}")
        print()
        print("Phase counts:")
        for key, value in result.summary["phase_counts"].items():
            print(f"  {key}: {value:,}")
        print()
        print("Shaped output counts:")
        for key, value in result.summary["shaped_phase_counts"].items():
            print(f"  {key}: {value:,}")
        print()
        print("Species map:")
        display(result.species_map_df)
        print("Atom table preview:")
        display(result.atoms_df.head(25))

        fig1 = tapsim_builder.plot_precipitate_layout(result)
        display(fig1)
        plt.close(fig1)
        fig2 = tapsim_builder.plot_structure_preview(result)
        display(fig2)
        plt.close(fig2)
        fig3 = tapsim_builder.plot_species_summary(result)
        display(fig3)
        plt.close(fig3)

    def on_preview(_):
        preview_button.disabled = True
        with out:
            out.clear_output()
            try:
                result = build_result()
                state["result"] = result
                render_result(result)
            except Exception as exc:
                print(f"Failed to build TAPSim specimen: {exc}")
        preview_button.disabled = False

    def on_export(_):
        export_button.disabled = True
        with out:
            try:
                result = build_result()
                state["result"] = result
                export_paths = tapsim_builder.export_tapsim_bundle(
                    result,
                    output_dir=output_dir.value,
                    dataset_name=dataset_name.value,
                    include_potential=include_potential.value,
                    potential_start=potential_start.value,
                    potential_end=potential_end.value,
                    write_csv=write_csv.value,
                    write_hdf=write_hdf.value,
                    write_range_hdf=write_range_hdf.value,
                )
                print("Exported TAPSim bundle:")
                for key, value in export_paths.items():
                    print(f"  {key}: {value}")
            except Exception as exc:
                print(f"Failed to export TAPSim bundle: {exc}")
        export_button.disabled = False

    load_preset_button.on_click(apply_preset)
    preview_button.on_click(on_preview)
    export_button.on_click(on_export)
    apply_preset()

    controls = widgets.VBox(
        [
            widgets.HBox([widgets.Label(value="Preset:", layout=label_layout), preset, load_preset_button]),
            widgets.HBox([widgets.Label(value="Mode:", layout=label_layout), mode]),
            widgets.HBox([widgets.Label(value="Geometry:", layout=label_layout), geometry_mode]),
            widgets.HBox([widgets.Label(value="A_gamma (A):", layout=label_layout), a_gamma]),
            widgets.HBox([widgets.Label(value="A_gprime (A):", layout=label_layout), a_gprime]),
            widgets.HBox(
                [
                    widgets.Label(value="Supercell (x, y, z):", layout=label_layout),
                    widgets.HBox([supercell_x, supercell_y, supercell_z]),
                ]
            ),
            widgets.HBox([widgets.Label(value="Single-phase composition:", layout=label_layout), single_phase_composition]),
            widgets.HBox([widgets.Label(value="Weight-% spec:", layout=label_layout), wt_spec]),
            widgets.HBox([widgets.Label(value="Gamma partition:", layout=label_layout), gamma_partition]),
            widgets.HBox([widgets.Label(value="Gprime A-site ratio:", layout=label_layout), gprime_a_site_ratio]),
            widgets.HBox([widgets.Label(value="Number of precipitates:", layout=label_layout), n_precip]),
            widgets.HBox([widgets.Label(value="Precipitate r_min (A):", layout=label_layout), r_min]),
            widgets.HBox([widgets.Label(value="Precipitate r_max (A):", layout=label_layout), r_max]),
            widgets.HBox([widgets.Label(value="Target gprime fraction:", layout=label_layout), target_gprime_fraction]),
            widgets.HBox([widgets.Label(value="Precipitate clearance (A):", layout=label_layout), clearance]),
            widgets.HBox([widgets.Label(value="Precipitate margin (A):", layout=label_layout), margin]),
            widgets.HBox([widgets.Label(value="Periodic precipitates:", layout=label_layout), periodic_precipitates]),
            widgets.HBox([widgets.Label(value="Radius distribution:", layout=label_layout), radius_dist]),
            widgets.HBox([widgets.Label(value="Lognormal sigma:", layout=label_layout), lognorm_sigma]),
            widgets.HBox([widgets.Label(value="Remove cut (A):", layout=label_layout), remove_cut]),
            widgets.HBox([widgets.Label(value="Cone diameter (A):", layout=label_layout), cone_diameter]),
            widgets.HBox([widgets.Label(value="Cone length (A):", layout=label_layout), cone_length]),
            widgets.HBox([widgets.Label(value="Hemisphere base (A):", layout=label_layout), hemisphere_base]),
            widgets.HBox([widgets.Label(value="Surface tolerance (A):", layout=label_layout), surf_tol]),
            widgets.HBox([widgets.Label(value="Bottom tolerance (A):", layout=label_layout), bottom_tol]),
            widgets.HBox([widgets.Label(value="Element ID start:", layout=label_layout), elem_start_id]),
            widgets.HBox([widgets.Label(value="Element ID max:", layout=label_layout), max_id]),
            widgets.HBox([widgets.Label(value="Separate phase IDs:", layout=label_layout), separate_phase_ids]),
            widgets.HBox([widgets.Label(value="Random seed:", layout=label_layout), seed]),
            widgets.HBox([widgets.Label(value="Output directory:", layout=label_layout), output_dir]),
            widgets.HBox([widgets.Label(value="Dataset name:", layout=label_layout), dataset_name]),
            widgets.HBox([widgets.Label(value="Include potential:", layout=label_layout), include_potential]),
            widgets.HBox([widgets.Label(value="Potential start:", layout=label_layout), potential_start]),
            widgets.HBox([widgets.Label(value="Potential end:", layout=label_layout), potential_end]),
            widgets.HBox([widgets.Label(value="Write CSV:", layout=label_layout), write_csv]),
            widgets.HBox([widgets.Label(value="Write HDF:", layout=label_layout), write_hdf]),
            widgets.HBox([widgets.Label(value="Write range HDF:", layout=label_layout), write_range_hdf]),
            widgets.HBox([preview_button, export_button]),
        ]
    )

    display(controls)
    display(out)
