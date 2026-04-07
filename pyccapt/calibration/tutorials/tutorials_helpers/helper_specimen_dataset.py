"""Notebook helper for generating specimen-like example datasets."""

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.reconstructions import specimen_builder

label_layout = widgets.Layout(width='220px')


def call_specimen_dataset_workflow(variables=None):
    out = Output()

    preset = widgets.Dropdown(options=specimen_builder.available_presets(), value='Al', description='Preset:')
    supercell_x = widgets.IntText(value=20)
    supercell_y = widgets.IntText(value=20)
    supercell_z = widgets.IntText(value=20)
    cone_diameter = widgets.FloatText(value=120.0)
    cone_length = widgets.FloatText(value=600.0)
    hemisphere_base = widgets.FloatText(value=200.0)
    noise_x = widgets.FloatText(value=0.0)
    noise_y = widgets.FloatText(value=0.0)
    noise_z = widgets.FloatText(value=0.0)
    noise_type = widgets.Dropdown(options=[('correlative', 'correlative'), ('noncorrelative', 'noncorrelative')])
    seed = widgets.IntText(value=42)
    save_path = widgets.Text(value='')
    load_into_variables = widgets.Dropdown(
        options=[('True', True), ('False', False)],
        value=variables is not None,
        description='Load data:',
    )
    build_button = widgets.Button(description='Build specimen dataset')

    def on_build(_):
        build_button.disabled = True
        with out:
            out.clear_output()
            try:
                dataset = specimen_builder.build_specimen_dataset(
                    preset=preset.value,
                    supercell=(supercell_x.value, supercell_y.value, supercell_z.value),
                    cone_diameter_angstrom=cone_diameter.value,
                    cone_length_angstrom=cone_length.value,
                    hemisphere_base_angstrom=hemisphere_base.value,
                    noise_levels_angstrom=(noise_x.value, noise_y.value, noise_z.value),
                    noise_type=noise_type.value,
                    seed=seed.value,
                    save_path=save_path.value or None,
                )
                print(f'Generated {len(dataset):,} ions for preset {preset.value}.')
                print('Element counts:')
                print(dataset['element'].value_counts().to_string())
                if save_path.value:
                    print(f'Saved dataset to: {save_path.value}')
                if variables is not None and load_into_variables.value:
                    variables.sync_from_data(dataset, update_backups=True)
                    print('Loaded the generated dataset into the active workflow variables.')
                display(dataset.head(20))
            except Exception as exc:
                print(f'Failed to generate specimen dataset: {exc}')
        build_button.disabled = False

    build_button.on_click(on_build)

    controls = widgets.VBox([
        widgets.HBox([widgets.Label(value='Preset:', layout=label_layout), preset]),
        widgets.HBox([widgets.Label(value='Supercell (x, y, z):', layout=label_layout),
                      widgets.HBox([supercell_x, supercell_y, supercell_z])]),
        widgets.HBox([widgets.Label(value='Cone diameter (A):', layout=label_layout), cone_diameter]),
        widgets.HBox([widgets.Label(value='Cone length (A):', layout=label_layout), cone_length]),
        widgets.HBox([widgets.Label(value='Hemisphere base (A):', layout=label_layout), hemisphere_base]),
        widgets.HBox([widgets.Label(value='Noise (x, y, z) A:', layout=label_layout),
                      widgets.HBox([noise_x, noise_y, noise_z])]),
        widgets.HBox([widgets.Label(value='Noise type:', layout=label_layout), noise_type]),
        widgets.HBox([widgets.Label(value='Random seed:', layout=label_layout), seed]),
        widgets.HBox([widgets.Label(value='Save path (.h5):', layout=label_layout), save_path]),
        widgets.HBox([widgets.Label(value='Load into variables:', layout=label_layout), load_into_variables]),
        widgets.HBox([build_button]),
    ])

    display(controls)
    display(out)
