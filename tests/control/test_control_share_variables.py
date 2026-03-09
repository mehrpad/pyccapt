"""Tests for control shared variable container."""

from __future__ import annotations

import multiprocessing

import pytest

from pyccapt.control.core import share_variables


@pytest.fixture
def base_conf():
    return {
        "COM_PORT_cryo": "COM1",
        "COM_PORT_V_dc": "COM2",
        "COM_PORT_V_p": "COM3",
        "COM_PORT_gauge_mc": "COM4",
        "COM_PORT_gauge_bc": "COM5",
        "COM_PORT_gauge_ll": "COM6",
        "COM_PORT_gauge_cll": "COM7",
        "COM_PORT_signal_generator": "USB::INSTR",
        "COM_PORT_thorlab_motor": "123456",
        "save_meta_interval_camera": 5,
        "save_meta_interval_visualization": 5,
        "pulse_amp_per_supply_voltage": 1.0,
        "max_laser_power": 100,
    }


@pytest.fixture
def variables(base_conf):
    manager = multiprocessing.Manager()
    namespace = manager.Namespace()
    shared = share_variables.Variables(base_conf, namespace)
    try:
        yield shared
    finally:
        manager.shutdown()


def test_missing_required_conf_key_raises(base_conf):
    missing_conf = dict(base_conf)
    missing_conf.pop("COM_PORT_cryo")
    manager = multiprocessing.Manager()
    namespace = manager.Namespace()
    try:
        with pytest.raises(KeyError):
            share_variables.Variables(missing_conf, namespace)
    finally:
        manager.shutdown()


def test_alias_assignment_updates_canonical_field(variables):
    variables.vdc_steps_up = 2.5
    variables.vp_min = 123

    assert variables.vdc_step_up == 2.5
    assert variables.v_p_min == 123


def test_extend_and_clear_list_field(variables):
    variables.extend_to("x", [1, 2, 3])
    assert variables.x == [1, 2, 3]

    variables.clear_to("x")
    assert variables.x == []


def test_extend_to_non_list_field_raises(variables):
    with pytest.raises(TypeError):
        variables.extend_to("counter", [1])


def test_unknown_assignment_is_shared_in_namespace(variables):
    variables.custom_runtime_flag = True
    assert variables.custom_runtime_flag is True
    assert getattr(variables.ns, "custom_runtime_flag") is True


def test_legacy_name_for_cryo_pump_led_is_supported(variables):
    variables.flag_cryo_pump_load_lock_led = True
    assert variables.flag_pump_cryo_load_lock_led is True


def _read_counter_in_spawned_process(shared_variables, queue):
    queue.put(shared_variables.COM_PORT_cryo)


def test_variables_can_be_unpickled_in_spawned_process_without_recursion(variables):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_read_counter_in_spawned_process, args=(variables, queue))

    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert queue.get(timeout=2) == "COM1"

