"""Notebook helper for unified raw-data analysis workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.data_tools import file_dialog, raw_data_workflow
from pyccapt.calibration.leap_tools.cameca_raw import rhit_tools, str_tools

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
        left_widget = widgets.FloatText(value=left, description='min:', layout=small_field_layout)
        right_widget = widgets.FloatText(value=right, description='max:', layout=small_field_layout)
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


def call_raw_data_workflow(variables=None):
    out = Output()
    state = {'roentdek': None, 'surface': None, 'cameca': None}

    roentdek_events_path = widgets.Text(value='', description='', layout=field_layout)
    roentdek_values_path = widgets.Text(value='', description='', layout=field_layout)
    roentdek_events_browse = widgets.Button(description='browse')
    roentdek_values_browse = widgets.Button(description='browse')
    roentdek_signal_kind = widgets.Dropdown(
        options=[('TOF values', 'tof'), ('Mass/charge values', 'mc')],
        value='tof',
        description='signal:',
        layout=field_layout,
    )
    roentdek_detx_columns = widgets.Text(value='6,10,14,18', description='x cols:', layout=field_layout)
    roentdek_dety_columns = widgets.Text(value='7,11,15,19', description='y cols:', layout=field_layout)
    roentdek_signal_columns = widgets.Text(value='8,12,16,20', description='signal cols:', layout=field_layout)
    roentdek_drop_zero = widgets.Checkbox(value=True, description='skip zero signal rows')
    roentdek_bin_size = widgets.FloatText(value=0.1, description='bin size:', layout=small_field_layout)
    roentdek_max_value = widgets.FloatText(value=1000.0, description='max x:', layout=small_field_layout)
    roentdek_max_bins = widgets.IntText(value=20, description='stats bins:', layout=small_field_layout)
    roentdek_drift_segments = widgets.IntText(value=20, description='segments:', layout=small_field_layout)
    roentdek_save_hits_path = widgets.Text(value='', description='save hits:', layout=field_layout)
    roentdek_analyze_button = widgets.Button(description='analyze RoentDek')
    roentdek_save_button = widgets.Button(description='save hits')
    roentdek_window_rows = _build_window_rows('peak ')

    surface_path = widgets.Text(value='', description='', layout=field_layout)
    surface_browse = widgets.Button(description='browse')
    surface_signal_kind = widgets.Dropdown(
        options=[('TOF plots', 'tof'), ('Mass/charge plots', 'mc')],
        value='tof',
        description='signal:',
        layout=field_layout,
    )
    surface_t0 = widgets.FloatText(value=0.0, description='t0:', layout=small_field_layout)
    surface_flight_path = widgets.FloatText(value=110.0, description='flight mm:', layout=small_field_layout)
    surface_detector_limit = widgets.FloatText(value=4.0, description='det lim:', layout=small_field_layout)
    surface_pulse_mode = widgets.Dropdown(
        options=[('voltage', 'voltage'), ('laser', 'laser')],
        value='voltage',
        description='pulse:',
        layout=field_layout,
    )
    surface_bin_size = widgets.FloatText(value=0.1, description='bin size:', layout=small_field_layout)
    surface_max_value = widgets.FloatText(value=1000.0, description='max x:', layout=small_field_layout)
    surface_max_bins = widgets.IntText(value=20, description='stats bins:', layout=small_field_layout)
    surface_drift_segments = widgets.IntText(value=20, description='segments:', layout=small_field_layout)
    surface_save_processed_path = widgets.Text(value='', description='save processed:', layout=field_layout)
    surface_analyze_button = widgets.Button(description='analyze Surface Concept')
    surface_save_button = widgets.Button(description='save processed')
    surface_load_button = widgets.Button(description='load into workflow')
    surface_window_rows = _build_window_rows('peak ')

    cameca_source = widgets.Dropdown(
        options=[('RHIT', 'rhit'), ('STR / HITS', 'str')],
        value='rhit',
        description='source:',
        layout=field_layout,
    )
    cameca_path = widgets.Text(value='', description='', layout=field_layout)
    cameca_epos_path = widgets.Text(value='', description='', layout=field_layout)
    cameca_rhit_path = widgets.Text(value='', description='', layout=field_layout)
    cameca_browse = widgets.Button(description='browse')
    cameca_epos_browse = widgets.Button(description='browse')
    cameca_rhit_browse = widgets.Button(description='browse')
    cameca_bin_size = widgets.FloatText(value=0.1, description='bin size:', layout=small_field_layout)
    cameca_tof_max = widgets.FloatText(value=2000.0, description='tof max:', layout=small_field_layout)
    cameca_mc_max = widgets.FloatText(value=80.0, description='mc max:', layout=small_field_layout)
    cameca_drift_segments = widgets.IntText(value=20, description='segments:', layout=small_field_layout)
    cameca_save_path = widgets.Text(value='', description='save processed:', layout=field_layout)
    cameca_analyze_button = widgets.Button(description='analyze LEAP raw')
    cameca_save_button = widgets.Button(description='save processed')
    cameca_load_button = widgets.Button(description='load into workflow')

    def _print_roentdek_summary(result: dict):
        counters = result['counters']
        raw_summary = result['raw_summary']
        print(f"Parsed {len(result['events']):,} RoentDek events.")
        print(
            f"Total timestamps: {raw_summary['total_timestamps']:,} | "
            f"matched events: {raw_summary['matched_pattern_events']:,} | "
            f"invalid events: {raw_summary['invalid_pattern_events']:,}"
        )
        print(
            f"multi-hit events: {raw_summary['multi_hit_events']:,} | "
            f"events with unmatched leftover timestamps: {raw_summary['unmatched_pattern_events']:,}"
        )
        print(
            'Recovered DLTS patterns: '
            f"2 DLTS={sum(counters['dld2'].values()):,}, "
            f"4 DLTS={sum(counters['dld4'].values()):,}, "
            f"6 DLTS={sum(counters['dld6'].values()):,}"
        )
        channel_totals = ', '.join(
            f"ch{channel}={count:,}" for channel, count in raw_summary['channel_timestamp_totals'].items()
        )
        print(f"Channel timestamp totals: {channel_totals}")
        missing_pairs = ', '.join(
            f"{pair}={raw_summary['pair_missing_partner_events'].get(pair, 0):,}"
            for pair in ('1-2', '3-4', '5-6')
        )
        unbalanced_pairs = ', '.join(
            f"{pair}={raw_summary['pair_unbalanced_events'].get(pair, 0):,}"
            for pair in ('1-2', '3-4', '5-6')
        )
        print(f"Events with missing pair partners: {missing_pairs}")
        print(f"Events with unbalanced pair counts: {unbalanced_pairs}")
        unmatched_channels = ', '.join(
            f"ch{channel}={count:,}" for channel, count in raw_summary['unmatched_timestamps'].items() if count
        )
        if unmatched_channels:
            print(f"Unmatched leftover timestamps by channel: {unmatched_channels}")
        if not result['hit_table'].empty:
            print(f"Flattened {len(result['hit_table']):,} detector hits from the numeric text table.")

    def _print_surface_summary(result: dict, processed: pd.DataFrame):
        recovery = result['recovery_stats']
        raw_summary = result['raw_summary']
        print(f"Parsed {len(result['sequence_records']):,} Surface Concept start-counter groups.")
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

    def on_analyze_roentdek(_):
        roentdek_analyze_button.disabled = True
        with out:
            out.clear_output()
            try:
                windows = _collect_windows(roentdek_window_rows)
                result = raw_data_workflow.analyze_roentdek_dataset(
                    roentdek_events_path.value,
                    numeric_values_path=roentdek_values_path.value.strip() or None,
                    detx_columns=_parse_column_positions(roentdek_detx_columns.value, 'RoentDek x columns'),
                    dety_columns=_parse_column_positions(roentdek_dety_columns.value, 'RoentDek y columns'),
                    signal_columns=_parse_column_positions(roentdek_signal_columns.value, 'RoentDek signal columns'),
                    signal_kind=roentdek_signal_kind.value,
                    drop_zero_signal=roentdek_drop_zero.value,
                )
                state['roentdek'] = result
                _print_roentdek_summary(result)

                _display_figure(raw_data_workflow.plot_roentdek_statistics(result['counters'], roentdek_max_bins.value))

                if not result['hit_table'].empty:
                    _display_figure(
                        raw_data_workflow.plot_signal_overlay_by_dlts(
                            result['hit_table'],
                            signal_kind=roentdek_signal_kind.value,
                            max_value=roentdek_max_value.value if roentdek_max_value.value > 0 else None,
                            bin_size=roentdek_bin_size.value,
                            only_in_detector=False,
                            title='RoentDek signal overlay by DLTS',
                        )
                    )
                    _display_figure(
                        raw_data_workflow.plot_detector_overview(
                            result['hit_table'],
                            only_in_detector=False,
                            title_prefix='RoentDek detector',
                        )
                    )
                    window_figure = raw_data_workflow.plot_signal_window_breakdown(
                        result['hit_table'],
                        windows,
                        signal_kind=roentdek_signal_kind.value,
                        only_in_detector=False,
                        title='RoentDek peak-window counts',
                    )
                    _display_figure(window_figure)
                    _display_figure(
                        raw_data_workflow.plot_tof_segment_drift(
                            result['hit_table'],
                            windows=windows,
                            num_segments=roentdek_drift_segments.value,
                            max_value=roentdek_max_value.value if roentdek_signal_kind.value == 'tof' else None,
                        )
                    )
                    _display_figure(raw_data_workflow.plot_detector_dead_zone_and_neighbors(result['hit_table']))
                    display(result['hit_table'].head(20))
                else:
                    print('No numeric hit table was loaded, so only event-level statistics are shown.')
            except Exception as exc:
                print(f'RoentDek analysis failed: {exc}')
        roentdek_analyze_button.disabled = False

    def on_save_roentdek(_):
        roentdek_save_button.disabled = True
        with out:
            if state['roentdek'] is None or state['roentdek']['hit_table'].empty:
                print('Run RoentDek analysis with a numeric value table before saving hits.')
            else:
                try:
                    _save_dataframe(state['roentdek']['hit_table'], roentdek_save_hits_path.value.strip())
                    print(f"Saved RoentDek hit table to: {roentdek_save_hits_path.value.strip()}")
                except Exception as exc:
                    print(f'Failed to save RoentDek hits: {exc}')
        roentdek_save_button.disabled = False

    def on_analyze_surface(_):
        surface_analyze_button.disabled = True
        with out:
            out.clear_output()
            try:
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
                state['surface'] = {'analysis': result, 'processed': processed}
                _print_surface_summary(result, processed)

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
                        title='Surface Concept signal overlay by DLTS',
                    )
                )
                _display_figure(
                    raw_data_workflow.plot_detector_overview(
                        result['hit_table'],
                        detector_limit_cm=surface_detector_limit.value,
                        only_in_detector=False,
                        title_prefix='Surface detector',
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
                    title='Surface Concept peak-window counts',
                )
                _display_figure(window_figure)
                display(result['hit_table'].head(20))
                display(processed.head(20))
            except Exception as exc:
                print(f'Surface Concept analysis failed: {exc}')
        surface_analyze_button.disabled = False

    def on_save_surface(_):
        surface_save_button.disabled = True
        with out:
            if state['surface'] is None:
                print('Run Surface Concept analysis before saving the processed dataset.')
            else:
                try:
                    raw_data_workflow.save_processed_raw_dataset(
                        state['surface']['processed'],
                        surface_save_processed_path.value.strip(),
                    )
                    print(f"Saved processed dataset to: {surface_save_processed_path.value.strip()}")
                except Exception as exc:
                    print(f'Failed to save the processed Surface Concept dataset: {exc}')
        surface_save_button.disabled = False

    def on_load_surface(_):
        surface_load_button.disabled = True
        with out:
            if state['surface'] is None:
                print('Run Surface Concept analysis before loading a processed dataset into the workflow.')
            elif variables is None:
                print('This workflow was opened without a shared variables object.')
            else:
                variables.sync_from_data(state['surface']['processed'], update_backups=True)
                if surface_path.value.strip():
                    variables.last_directory = str(Path(surface_path.value.strip()).parent)
                print('Loaded the processed Surface Concept dataset into the active workflow variables.')
        surface_load_button.disabled = False

    def on_analyze_cameca(_):
        cameca_analyze_button.disabled = True
        with out:
            out.clear_output()
            try:
                source = cameca_source.value
                if source == 'rhit':
                    hits, histograms, metadata = rhit_tools.rhit_load(cameca_path.value)
                    calibration = None
                    if cameca_epos_path.value.strip():
                        hits, calibration = rhit_tools.rhit_calibrate_from_epos(hits, cameca_epos_path.value.strip())
                    processed = rhit_tools.rhit_to_ccapt(hits)
                    state['cameca'] = {
                        'source': source,
                        'hits': hits,
                        'processed': processed,
                        'metadata': metadata,
                        'histograms': histograms,
                        'calibration': calibration,
                    }
                    print(f"Loaded {len(hits):,} RHIT hits.")
                    if calibration is not None:
                        print(
                            f"Applied EPOS calibration: matched_events={calibration['matched_events']:,}, "
                            f"t_offset={calibration['t_offset']:.4f} ns, "
                            f"residual_std={calibration['residual_std']:.4e}"
                        )
                    _print_processed_summary(processed, 'Processed RHIT dataset summary')
                else:
                    hits, metadata = str_tools.str_load(cameca_path.value)
                    hits = str_tools.str_calculate_positions(hits)
                    calibration = None
                    if not cameca_rhit_path.value.strip():
                        raise ValueError('STR / HITS analysis needs a matching RHIT file for calibration.')
                    rhit_hits, rhit_histograms, rhit_metadata = rhit_tools.rhit_load(cameca_rhit_path.value.strip())
                    if cameca_epos_path.value.strip():
                        rhit_hits, _ = rhit_tools.rhit_calibrate_from_epos(rhit_hits, cameca_epos_path.value.strip())
                    hits, calibration = str_tools.str_calibrate_from_rhit(hits, rhit_hits, rhit_histograms, rhit_metadata)
                    processed = str_tools.str_to_ccapt(hits)
                    state['cameca'] = {
                        'source': source,
                        'hits': hits,
                        'processed': processed,
                        'metadata': metadata,
                        'histograms': {},
                        'calibration': calibration,
                    }
                    print(f"Loaded {len(hits):,} STR / HITS events.")
                    print(
                        f"STR calibration: clock={calibration['clock_ns'] * 1000:.2f} ps, "
                        f"t0={calibration['t0_tdc']:.1f} TDC, "
                        f"corr={calibration['spectrum_correlation']:.4f}"
                    )
                    _print_processed_summary(processed, 'Processed STR / HITS dataset summary')

                _display_figure(
                    raw_data_workflow.plot_processed_dataset_overview(
                        state['cameca']['processed'],
                        mc_max=cameca_mc_max.value if cameca_mc_max.value > 0 else None,
                        tof_max=cameca_tof_max.value if cameca_tof_max.value > 0 else None,
                        bin_size=cameca_bin_size.value,
                        title_prefix='LEAP raw',
                    )
                )
                _display_figure(
                    raw_data_workflow.plot_tof_segment_drift(
                        state['cameca']['processed'],
                        num_segments=cameca_drift_segments.value,
                        max_value=cameca_tof_max.value if cameca_tof_max.value > 0 else None,
                    )
                )
                _display_figure(raw_data_workflow.plot_detector_dead_zone_and_neighbors(state['cameca']['processed']))
                display(state['cameca']['processed'].head(20))
            except Exception as exc:
                print(f'LEAP raw analysis failed: {exc}')
        cameca_analyze_button.disabled = False

    def on_save_cameca(_):
        cameca_save_button.disabled = True
        with out:
            if state['cameca'] is None:
                print('Run LEAP raw analysis before saving the processed dataset.')
            else:
                try:
                    raw_data_workflow.save_processed_raw_dataset(
                        state['cameca']['processed'],
                        cameca_save_path.value.strip(),
                    )
                    print(f"Saved processed dataset to: {cameca_save_path.value.strip()}")
                except Exception as exc:
                    print(f'Failed to save the LEAP processed dataset: {exc}')
        cameca_save_button.disabled = False

    def on_load_cameca(_):
        cameca_load_button.disabled = True
        with out:
            if state['cameca'] is None:
                print('Run LEAP raw analysis before loading a processed dataset into the workflow.')
            elif variables is None:
                print('This workflow was opened without a shared variables object.')
            else:
                variables.sync_from_data(state['cameca']['processed'], update_backups=True)
                if cameca_path.value.strip():
                    variables.last_directory = str(Path(cameca_path.value.strip()).parent)
                print('Loaded the LEAP processed dataset into the active workflow variables.')
        cameca_load_button.disabled = False

    roentdek_events_browse.on_click(lambda _: _browse_file(roentdek_events_path, out, variables))
    roentdek_values_browse.on_click(lambda _: _browse_file(roentdek_values_path, out, variables))
    surface_browse.on_click(lambda _: _browse_file(surface_path, out, variables))
    cameca_browse.on_click(lambda _: _browse_file(cameca_path, out, variables))
    cameca_epos_browse.on_click(lambda _: _browse_file(cameca_epos_path, out, variables))
    cameca_rhit_browse.on_click(lambda _: _browse_file(cameca_rhit_path, out, variables))

    roentdek_analyze_button.on_click(on_analyze_roentdek)
    roentdek_save_button.on_click(on_save_roentdek)
    surface_analyze_button.on_click(on_analyze_surface)
    surface_save_button.on_click(on_save_surface)
    surface_load_button.on_click(on_load_surface)
    cameca_analyze_button.on_click(on_analyze_cameca)
    cameca_save_button.on_click(on_save_cameca)
    cameca_load_button.on_click(on_load_cameca)

    roentdek_window_box = widgets.VBox(
        [
            widgets.HBox([label_widget, min_widget, max_widget])
            for label_widget, min_widget, max_widget in roentdek_window_rows
        ]
    )
    surface_window_box = widgets.VBox(
        [
            widgets.HBox([label_widget, min_widget, max_widget])
            for label_widget, min_widget, max_widget in surface_window_rows
        ]
    )

    roentdek_panel = widgets.VBox(
        [
            widgets.HTML('<b>RoentDek text workflow</b><br>Load the event text file and optionally the numeric value table used in the old notebook.'),
            _path_row('Event text file:', roentdek_events_path, roentdek_events_browse),
            _path_row('Numeric value text:', roentdek_values_path, roentdek_values_browse),
            widgets.HBox([widgets.Label(value='Signal interpretation:', layout=label_layout), roentdek_signal_kind]),
            widgets.HBox([widgets.Label(value='Detector x columns:', layout=label_layout), roentdek_detx_columns]),
            widgets.HBox([widgets.Label(value='Detector y columns:', layout=label_layout), roentdek_dety_columns]),
            widgets.HBox([widgets.Label(value='Signal columns:', layout=label_layout), roentdek_signal_columns]),
            widgets.HBox([widgets.Label(value='Filtering:', layout=label_layout), roentdek_drop_zero]),
            widgets.HBox([widgets.Label(value='Plot settings:', layout=label_layout), widgets.HBox([roentdek_bin_size, roentdek_max_value, roentdek_max_bins, roentdek_drift_segments])]),
            widgets.HBox([widgets.Label(value='Peak windows:', layout=label_layout), roentdek_window_box]),
            widgets.HBox([widgets.Label(value='Save hit table:', layout=label_layout), roentdek_save_hits_path]),
            widgets.HBox([roentdek_analyze_button, roentdek_save_button]),
        ]
    )

    surface_panel = widgets.VBox(
        [
            widgets.HTML('<b>Surface Concept raw workflow</b><br>Recover 4-DLTS and 2-DLTS hits, inspect TOF or mass/charge, then save or load the processed dataset.'),
            _path_row('Raw HDF5 path:', surface_path, surface_browse),
            widgets.HBox([widgets.Label(value='Signal plots:', layout=label_layout), surface_signal_kind]),
            widgets.HBox([widgets.Label(value='Calibration inputs:', layout=label_layout), widgets.HBox([surface_t0, surface_flight_path, surface_detector_limit])]),
            widgets.HBox([widgets.Label(value='Pulse mode:', layout=label_layout), surface_pulse_mode]),
            widgets.HBox([widgets.Label(value='Plot settings:', layout=label_layout), widgets.HBox([surface_bin_size, surface_max_value, surface_max_bins, surface_drift_segments])]),
            widgets.HBox([widgets.Label(value='Peak windows:', layout=label_layout), surface_window_box]),
            widgets.HBox([widgets.Label(value='Save processed file:', layout=label_layout), surface_save_processed_path]),
            widgets.HBox([surface_analyze_button, surface_save_button, surface_load_button]),
        ]
    )

    cameca_panel = widgets.VBox(
        [
            widgets.HTML('<b>LEAP / Cameca raw workflow</b><br>Analyze RHIT or STR/HITS raw files inside the same raw-data workflow and convert them to a processed dataset for the rest of PyCCAPT.'),
            widgets.HBox([widgets.Label(value='Raw source:', layout=label_layout), cameca_source]),
            _path_row('Primary raw file:', cameca_path, cameca_browse),
            _path_row('Matching EPOS:', cameca_epos_path, cameca_epos_browse),
            _path_row('Matching RHIT:', cameca_rhit_path, cameca_rhit_browse),
            widgets.HBox([widgets.Label(value='Plot settings:', layout=label_layout), widgets.HBox([cameca_bin_size, cameca_tof_max, cameca_mc_max, cameca_drift_segments])]),
            widgets.HBox([widgets.Label(value='Save processed file:', layout=label_layout), cameca_save_path]),
            widgets.HBox([cameca_analyze_button, cameca_save_button, cameca_load_button]),
        ]
    )

    tabs = widgets.Tab(children=[roentdek_panel, surface_panel, cameca_panel])
    tabs.set_title(0, 'RoentDek')
    tabs.set_title(1, 'Surface Concept')
    tabs.set_title(2, 'LEAP / Cameca')

    display(tabs)
    display(out)
