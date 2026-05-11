"""Notebook helper for unified raw-data analysis workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.data_tools import data_loadcrop, file_dialog, raw_data_workflow

label_layout = widgets.Layout(width='220px')
field_layout = widgets.Layout(width='420px')
small_field_layout = widgets.Layout(width='130px')


def _build_window_rows(prefix: str) -> list[tuple[widgets.Text, widgets.FloatText, widgets.FloatText]]:
    rows = []
    defaults = [
        ('Peak 1', 0.0, 0.0),
        ('Peak 2', 0.0, 0.0),
        ('Peak 3', 0.0, 0.0),
    ]
    for index, (label, left, right) in enumerate(defaults, start=1):
        label_widget = widgets.Text(value=label, description=f'{prefix}{index}:', layout=field_layout)
        left_widget = widgets.FloatText(value=left, description='Min:', layout=small_field_layout)
        right_widget = widgets.FloatText(value=right, description='Max:', layout=small_field_layout)
        rows.append((label_widget, left_widget, right_widget))
    return rows


def _collect_windows(window_rows: list[tuple[widgets.Text, widgets.FloatText, widgets.FloatText]]) -> list[dict]:
    windows = []
    for index, (label_widget, left_widget, right_widget) in enumerate(window_rows, start=1):
        label = label_widget.value.strip() or f'Peak {index}'
        left = float(left_widget.value)
        right = float(right_widget.value)
        if left == 0 and right == 0:
            continue
        if right <= left:
            raise ValueError(f'Peak window {label!r} is invalid: max must be greater than min')
        windows.append({'label': label, 'min': left, 'max': right})
    return windows


def _parse_column_positions(text: str, label: str) -> tuple[int, ...]:
    cleaned = [item.strip() for item in text.split(',') if item.strip()]
    if not cleaned:
        raise ValueError(f'{label} cannot be empty')
    try:
        return tuple(int(item) for item in cleaned)
    except ValueError as exc:
        raise ValueError(f'{label} must be a comma-separated list of integers') from exc


def _display_figure(figure):
    if figure is None:
        return
    display(figure)
    plt.close(figure)


def _save_dataframe(dataframe: pd.DataFrame, output_path: str) -> None:
    if not output_path:
        raise ValueError('Please provide an output path before saving')
    lower_path = output_path.lower()
    if lower_path.endswith('.csv'):
        dataframe.to_csv(output_path, index=False)
    elif lower_path.endswith(('.h5', '.hdf', '.hdf5')):
        dataframe.to_hdf(output_path, key='df', mode='w')
    else:
        raise ValueError('Output path must end with .csv, .h5, .hdf, or .hdf5')


def _browse_file(text_widget: widgets.Text, out: Output, variables=None) -> None:
    try:
        selected_path = file_dialog.choose_file_path(
            file_dialog.resolve_initial_directory(
                text_widget.value,
                getattr(variables, 'last_directory', None) if variables is not None else None,
            )
        )
        if selected_path:
            text_widget.value = selected_path
            if variables is not None:
                variables.last_directory = str(Path(selected_path).parent)
    except Exception as exc:
        with out:
            print(f'File chooser failed: {exc}')


def _path_row(label: str, text_widget: widgets.Text, button: widgets.Button) -> widgets.HBox:
    return widgets.HBox([widgets.Label(value=label, layout=label_layout), text_widget, button])


def _print_processed_summary(dataframe: pd.DataFrame, title: str) -> None:
    summary = raw_data_workflow.summarize_processed_dataset(dataframe)
    print(title)
    print(f"Rows: {summary.get('num_rows', 0):,}")
    if 'mc (Da)_median' in summary:
        print(
            f"mc range: {summary['mc (Da)_min']:.4f} to {summary['mc (Da)_max']:.4f} | "
            f"median {summary['mc (Da)_median']:.4f}"
        )
    if 't (ns)_median' in summary:
        print(
            f"tof range: {summary['t (ns)_min']:.4f} to {summary['t (ns)_max']:.4f} | "
            f"median {summary['t (ns)_median']:.4f}"
        )
    if 'high_voltage (V)_median' in summary:
        print(
            f"high voltage range: {summary['high_voltage (V)_min']:.2f} to "
            f"{summary['high_voltage (V)_max']:.2f}"
        )


_PYCCAPT_RAW_EXTENSIONS = {'.h5', '.hdf', '.hdf5'}


def _validate_pyccapt_raw_path(path: str) -> None:
    """Reject non-pyccapt-style inputs early with a helpful message.

    The raw-data analysis workflow only accepts pyccapt-style raw HDF5 files
    (with a ``/dld`` and/or ``/tdc`` group). LEAP RHIT/STR/HITS files and old
    RoentDek text exports must first be converted to a pyccapt-raw HDF5 via
    the cameca raw import workflow.
    """
    if not path:
        raise ValueError('Provide a raw HDF5 file path before analyzing.')
    suffix = Path(path).suffix.lower()
    if suffix in {'.rhit', '.str', '.hits', '.epos', '.pos', '.apt', '.ato'}:
        raise ValueError(
            f"{path!r} is a LEAP raw file ({suffix}). The raw-data-analysis "
            "workflow accepts only pyccapt-raw HDF5 files. Convert your LEAP "
            "raw data with the 'cameca raw import' workflow first (it writes a "
            "<basename>_pyccapt-raw.h5 with the dld/ and tdc/ groups expected "
            "here), then load that .h5 file."
        )
    if suffix not in _PYCCAPT_RAW_EXTENSIONS:
        raise ValueError(
            f"{path!r} is not a recognized HDF5 file. The raw-data-analysis "
            "workflow accepts only pyccapt-raw HDF5 files (.h5 / .hdf / .hdf5)."
        )


def call_raw_data_workflow(variables=None):
    out = Output()
    state = {'surface': None}

    surface_path = widgets.Text(value='', description='', layout=field_layout)
    surface_browse = widgets.Button(description='Browse')
    surface_detector_type = widgets.Dropdown(
        options=[
            ('2 DL (4 channels)', '2dl'),
            ('3 DL (6 channels)', '3dl'),
        ],
        value='2dl',
        description='Detector:',
        layout=field_layout,
    )
    surface_signal_kind = widgets.Dropdown(
        options=[('TOF plots', 'tof'), ('Mass/charge plots', 'mc')],
        value='tof',
        description='Signal:',
        layout=field_layout,
    )
    surface_t0 = widgets.FloatText(value=0.0, description='T0:', layout=small_field_layout)
    surface_flight_path = widgets.FloatText(value=110.0, description='Flight mm:', layout=small_field_layout)
    surface_detector_limit = widgets.FloatText(value=4.0, description='Det lim:', layout=small_field_layout)
    surface_pulse_mode = widgets.Dropdown(
        options=[('voltage', 'voltage'), ('laser', 'laser')],
        value='voltage',
        description='Pulse:',
        layout=field_layout,
    )
    surface_bin_size = widgets.FloatText(value=0.1, description='Bin size:', layout=small_field_layout)
    surface_max_value = widgets.FloatText(value=1000.0, description='Max x:', layout=small_field_layout)
    surface_max_bins = widgets.IntText(value=20, description='Stats bins:', layout=small_field_layout)
    surface_drift_segments = widgets.IntText(value=20, description='Segments:', layout=small_field_layout)
    surface_save_processed_path = widgets.Text(value='', description='Save processed:', layout=field_layout)
    surface_analyze_button = widgets.Button(description='Analyze raw HDF5')
    surface_save_button = widgets.Button(description='Save processed')
    surface_load_button = widgets.Button(description='Load into workflow')
    surface_window_rows = _build_window_rows('peak ')

    def _print_2dl_summary(result: dict, processed: pd.DataFrame):
        recovery = result['recovery_stats']
        raw_summary = result['raw_summary']
        print(f"Parsed {len(result['sequence_records']):,} 2DL start-counter groups.")
        print(
            f"Raw sequence groups: valid 4ch={raw_summary['valid_four_channel_groups']:,}, "
            f"invalid 4ch={raw_summary['invalid_four_channel_groups']:,}, "
            f"3ch={raw_summary['length_three_groups']:,}, 2ch={raw_summary['length_two_groups']:,}, "
            f"1ch={raw_summary['length_one_groups']:,}"
        )
        print(
            f"Multi-hit groups: four-channel blocks={raw_summary['multi_hit_groups_of_four']:,} "
            f"({raw_summary['multi_hit_groups_of_four_timestamps']:,} timestamps), "
            f"irregular={raw_summary['multi_hit_irregular']:,} "
            f"({raw_summary['multi_hit_irregular_timestamps']:,} timestamps)"
        )
        channel_totals = ', '.join(
            f"ch{channel}={count:,}" for channel, count in raw_summary['channel_timestamp_totals'].items()
        )
        print(f"Channel timestamp totals: {channel_totals}")
        print(
            f"Recovered hits: {recovery['recovered_hits']:,} total, "
            f"{recovery['two_d_hits']:,} 4-DLTS, {recovery['one_d_hits']:,} 2-DLTS"
        )
        print(
            f"In detector: {recovery['two_d_in_detector']:,} 4-DLTS and "
            f"{recovery['one_d_in_detector']:,} 2-DLTS"
        )
        print(
            f"Outside detector: {recovery['outside_detector_hits']:,} | "
            f"Unrecoverable chunks: {recovery['unrecoverable_chunks']:,}"
        )
        print(f"Processed dataset rows available for the main workflow: {len(processed):,}")

    def on_analyze_surface(_):
        surface_analyze_button.disabled = True
        with out:
            out.clear_output()
            try:
                _validate_pyccapt_raw_path(surface_path.value)
                detector_type = surface_detector_type.value
                if detector_type == '2dl':
                    _analyze_2dl()
                elif detector_type == '3dl':
                    _analyze_3dl()
                else:
                    raise ValueError(f"Unknown detector type: {detector_type!r}")
            except Exception as exc:
                print(f'Raw HDF5 analysis failed: {exc}')
        surface_analyze_button.disabled = False

    def _analyze_2dl():
        # 2 DL detectors expose 4 channels (2 delay lines x 2 ends each).
        windows = _collect_windows(surface_window_rows)
        result = raw_data_workflow.analyze_surface_concept_dataset(
            surface_path.value,
            detector_limit_cm=surface_detector_limit.value,
            t0=surface_t0.value,
            flight_path_length_mm=surface_flight_path.value,
            pulse_mode=surface_pulse_mode.value,
        )
        processed = raw_data_workflow.surface_concept_hits_to_processed_dataframe(
            result['hit_table'],
            pulse_mode=surface_pulse_mode.value,
        )
        state['surface'] = {'analysis': result, 'processed': processed, 'detector_type': '2dl'}
        _print_2dl_summary(result, processed)

        _display_figure(
            raw_data_workflow.plot_surface_concept_sequence_statistics(
                result['sequence_stats'],
                max_bins=surface_max_bins.value,
            )
        )
        _display_figure(raw_data_workflow.plot_surface_concept_recovery_summary(result['recovery_stats']))
        _display_figure(
            raw_data_workflow.plot_signal_overlay_by_dlts(
                result['hit_table'],
                signal_kind=surface_signal_kind.value,
                max_value=surface_max_value.value if surface_max_value.value > 0 else None,
                bin_size=surface_bin_size.value,
                title='2DL signal overlay by DLTS',
            )
        )
        _display_figure(
            raw_data_workflow.plot_detector_overview(
                result['hit_table'],
                detector_limit_cm=surface_detector_limit.value,
                only_in_detector=False,
                title_prefix='2DL detector',
            )
        )
        _display_figure(
            raw_data_workflow.plot_surface_concept_recovery_yield(
                result['recovery_diagnostics'],
                num_bins=surface_max_bins.value,
            )
        )
        _display_figure(raw_data_workflow.plot_partial_hit_efficiency_maps(result['recovery_diagnostics']))
        _display_figure(
            raw_data_workflow.plot_tof_segment_drift(
                result['hit_table'],
                windows=windows,
                num_segments=surface_drift_segments.value,
                max_value=surface_max_value.value,
            )
        )
        _display_figure(raw_data_workflow.plot_detector_dead_zone_and_neighbors(result['hit_table']))
        window_figure = raw_data_workflow.plot_signal_window_breakdown(
            result['hit_table'],
            windows,
            signal_kind=surface_signal_kind.value,
            title='2DL peak-window counts',
        )
        _display_figure(window_figure)
        display(result['hit_table'].head(20))
        display(processed.head(20))

    def _analyze_3dl():
        # 3 DL detectors expose 6 channels (3 delay lines x 2 ends each).
        df_tdc = data_loadcrop.fetch_dataset_from_dld_grp(
            surface_path.value, extract_mode='tdc_ro'
        )
        if df_tdc is None:
            raise RuntimeError(
                "Could not read tdc/ group from the file. "
                "For 3DL detectors the HDF5 must contain a 6-channel tdc/ group "
                "(channel, start_counter, high_voltage, pulse, time_data). "
                "Re-export from the cameca raw import workflow if the source is LEAP."
            )
        unique_channels = sorted(int(c) for c in df_tdc['channel'].unique())
        print(
            f"Loaded 3DL tdc frame: {len(df_tdc):,} rows, channels={unique_channels}."
        )
        if len(unique_channels) > 6:
            print(
                f"  [WARN] {len(unique_channels)} channels seen but 3DL expects <= 6. "
                "Verify the file matches the selected detector type."
            )
        result = raw_data_workflow.analyze_roentdek_tdc_frame(df_tdc)
        state['surface'] = {'analysis': result, 'processed': None, 'detector_type': '3dl'}
        print(
            "Analyzed 3DL raw frame. A processed PyCCAPT dataset is not produced "
            "directly from the tdc/ group for 3DL; for that, use the dld/ group of "
            "the same HDF5 (load via fetch_dataset_from_dld_grp(..., extract_mode='dld'))."
        )
        if 'events' in result:
            print(f"  3DL classified events: {len(result['events']):,}")
            display(result['events'].head(20))
        if 'counters' in result and isinstance(result['counters'], dict):
            print("  Class counts:")
            for label, count in result['counters'].items():
                print(f"    {label}: {count:,}")

    def on_save_surface(_):
        surface_save_button.disabled = True
        with out:
            if state['surface'] is None:
                print('Run the raw HDF5 analysis before saving the processed dataset.')
            elif state['surface'].get('processed') is None:
                print(
                    'No processed dataset is available — the 3DL pipeline does not '
                    'currently produce one. Use the dld/ group of the same HDF5 directly.'
                )
            else:
                try:
                    raw_data_workflow.save_processed_raw_dataset(
                        state['surface']['processed'],
                        surface_save_processed_path.value.strip(),
                    )
                    print(f"Saved processed dataset to: {surface_save_processed_path.value.strip()}")
                except Exception as exc:
                    print(f'Failed to save the processed dataset: {exc}')
        surface_save_button.disabled = False

    def on_load_surface(_):
        surface_load_button.disabled = True
        with out:
            if state['surface'] is None:
                print('Run the raw HDF5 analysis before loading a processed dataset into the workflow.')
            elif state['surface'].get('processed') is None:
                print(
                    'No processed dataset is available for the selected detector type. '
                    '3DL analysis does not currently emit a processed dataset.'
                )
            elif variables is None:
                print('This workflow was opened without a shared variables object.')
            else:
                variables.sync_from_data(state['surface']['processed'], update_backups=True)
                if surface_path.value.strip():
                    variables.last_directory = str(Path(surface_path.value.strip()).parent)
                print('Loaded the processed dataset into the active workflow variables.')
        surface_load_button.disabled = False

    surface_browse.on_click(lambda _: _browse_file(surface_path, out, variables))

    surface_analyze_button.on_click(on_analyze_surface)
    surface_save_button.on_click(on_save_surface)
    surface_load_button.on_click(on_load_surface)

    surface_window_box = widgets.VBox(
        [
            widgets.HBox([label_widget, min_widget, max_widget])
            for label_widget, min_widget, max_widget in surface_window_rows
        ]
    )

    panel = widgets.VBox(
        [
            widgets.HTML(
                '<b>PyCCAPT raw HDF5 analysis</b><br>'
                'Loads a pyccapt-raw HDF5 file (the format produced by the '
                'PyCCAPT control software or by the <i>cameca raw import</i> '
                "workflow's <i>Save raw analysis HDF5</i> button). After "
                'loading, choose whether the detector is <b>2 DL (4 channels)</b> '
                'or <b>3 DL (6 channels)</b> — the analysis pipeline routes '
                'accordingly.'
                '<br><br>'
                '<i>For LEAP RHIT / STR / HITS files, first convert them with '
                "the <b>cameca raw import</b> workflow (it writes a "
                '<code>&lt;basename&gt;_pyccapt-raw.h5</code> with the '
                'dld/ and tdc/ groups expected here), then load the resulting '
                '.h5 file in this panel.</i>'
            ),
            _path_row('PyCCAPT raw HDF5 path:', surface_path, surface_browse),
            widgets.HBox([widgets.Label(value='Detector type:', layout=label_layout), surface_detector_type]),
            widgets.HBox([widgets.Label(value='Signal plots:', layout=label_layout), surface_signal_kind]),
            widgets.HBox([widgets.Label(value='Calibration inputs:', layout=label_layout), widgets.HBox([surface_t0, surface_flight_path, surface_detector_limit])]),
            widgets.HBox([widgets.Label(value='Pulse mode:', layout=label_layout), surface_pulse_mode]),
            widgets.HBox([widgets.Label(value='Plot settings:', layout=label_layout), widgets.HBox([surface_bin_size, surface_max_value, surface_max_bins, surface_drift_segments])]),
            widgets.HBox([widgets.Label(value='Peak windows:', layout=label_layout), surface_window_box]),
            widgets.HBox([widgets.Label(value='Save processed file:', layout=label_layout), surface_save_processed_path]),
            widgets.HBox([surface_analyze_button, surface_save_button, surface_load_button]),
        ]
    )

    display(panel)
    display(out)
