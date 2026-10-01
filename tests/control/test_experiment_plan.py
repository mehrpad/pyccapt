"""Plan parsing, hardware prevalidation, export, mapping and recorded provenance."""
import json
from pathlib import Path

import pytest

from pyccapt.control.core import experiment_plan as plan, read_files
from pyccapt.control.gui import main_parameters


EXAMPLE = Path(__file__).resolve().parents[2] / 'pyccapt/files/experiment_plan.example.toml'
CONFIG = EXAMPLE.parents[1] / 'config.toml'


def test_example_merges_defaults_and_passes_hardware_validation():
    rows = plan.load_plan(EXAMPLE)
    assert [row['sample_id'] for row in rows] == [1, 2]
    assert [row['vdc_steps_up'] for row in rows] == [1., .5]
    assert rows[0]['criteria_time'] and not rows[1]['criteria_time']
    assert main_parameters.validate_experiment_queue(rows, read_files.read_toml_file(CONFIG)) == rows


def test_save_reload_roundtrip_with_strings_and_shared_defaults(tmp_path):
    rows = plan.load_plan(EXAMPLE)
    rows[0]['ex_name'] = 'quote " and backslash \\ and newline\nµ'
    path = tmp_path/'roundtrip.toml'
    plan.save_plan(path, rows)
    assert plan.load_plan(path) == rows
    assert '[defaults]' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('key,value', [
    ('criteria_time', 'false'), ('vdc_min', True), ('ex_freq', 2.5),
    ('detection_rate_init', float('nan')), ('vdc_steps_up', float('inf')),
    ('sample_id', 4), ('sample_id', True), ('hit_displayed', 0),
    ('vdc_steps_down', -1), ('pulse_mode', 'Voltgae'), ('counter_source', 'invalid'),
    ('control_algorithm', 'invalid'), ('ex_name', ''), ('email_interval_events', 0),
])
def test_rejects_wrong_types_and_invalid_values(key, value):
    row = plan.load_plan(EXAMPLE)[0]
    row[key] = value
    with pytest.raises(plan.PlanError, match=key):
        plan.validate_item(row)


def test_rejects_unknown_missing_fields_and_no_stop():
    row = plan.load_plan(EXAMPLE)[0]
    with pytest.raises(plan.PlanError, match='unknown'):
        plan.validate_item({**row, 'typo': 1})
    del row['email']
    with pytest.raises(plan.PlanError, match='missing'):
        plan.validate_item(row)
    row['email'] = ''
    row['criteria_time'] = False
    with pytest.raises(plan.PlanError, match='stopping'):
        plan.validate_item(row)


@pytest.mark.parametrize('data', [
    {}, {'schema_version': True}, {'schema_version': 2},
    {'schema_version': 1, 'experiments': []},
    {'schema_version': 1, 'defaults': {'sample_id': 1}, 'experiments': [{}]},
    {'schema_version': 1, 'unknown': 1},
])
def test_rejects_invalid_plan_structure(data):
    with pytest.raises(plan.PlanError):
        plan.resolve_plan(data)


def test_duplicate_toml_and_textline_keys_rejected(tmp_path):
    path = tmp_path/'duplicate.toml'
    path.write_text('schema_version = 1\nschema_version = 1\n')
    with pytest.raises(plan.PlanError):
        plan.load_plan(path)
    with pytest.raises(main_parameters.ParameterError, match='Duplicate'):
        main_parameters.parse_textline_experiments('{ex_user=a;ex_user=b}')


def test_hardware_limits_in_later_row_reject_whole_queue():
    rows = plan.load_plan(EXAMPLE)
    conf = read_files.read_toml_file(CONFIG)
    rows[1]['vdc_max'] = conf['max_vdc'] + 1
    with pytest.raises(main_parameters.ParameterError, match='Experiment 2'):
        main_parameters.validate_experiment_queue(rows, conf)
    rows[1]['vdc_max'] = 3000
    rows[1]['pulse_frequency'] = conf['max_voltage_pulse_frequency'] + 1
    with pytest.raises(main_parameters.ParameterError, match='Experiment 2'):
        main_parameters.validate_experiment_queue(rows, conf)


def test_mapping_uses_plan_order_allows_repeat_and_requires_saved_positions():
    rows = plan.load_plan(EXAMPLE)
    assert plan.alignment_samples(rows[::-1], {1: (), 2: ()}) == (2, 1)
    assert plan.alignment_samples([rows[0], rows[0]], {1: ()}) == (1, 1)
    with pytest.raises(plan.PlanError, match='Experiment 2'):
        plan.alignment_samples(rows, {1: ()})
    del rows[0]['sample_id']
    with pytest.raises(plan.PlanError, match='sample_id'):
        plan.alignment_samples(rows, {1: (), 2: ()})


def test_snapshot_is_reloadable_and_contains_source_and_queue_index(tmp_path):
    row = plan.load_plan(EXAMPLE)[1]
    plan.write_plan_snapshot(tmp_path, {'experiment': row, 'source': str(EXAMPLE), 'queue_index': 2})
    assert plan.load_plan(tmp_path/'experiment_plan.toml') == [row]
    assert json.loads((tmp_path/'experiment_plan_source.json').read_text()) == {
        'source': str(EXAMPLE), 'queue_index': 2}
