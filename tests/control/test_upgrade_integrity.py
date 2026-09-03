from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyccapt.control.apt.apt_exp_control import APT_Exp_Control
from pyccapt.control.apt.experiment_state import ExperimentState
from pyccapt.control.core.data_integrity import recover_chunks
from pyccapt.control.gui.main_parameters import ParameterError, validate_run_parameters


class _CompletionEvent:
    def __init__(self, variables):
        self.variables = variables
        self.was_set = False

    def set(self):
        assert self.variables.flag_end_experiment is True
        assert self.variables.experiment_state == ExperimentState.FAILED.value
        self.was_set = True


def test_experiment_failure_publishes_state_before_completion_event():
    variables = SimpleNamespace(ex_freq=1, flag_end_experiment=False, experiment_state="idle", experiment_error="")
    event = _CompletionEvent(variables)
    control = APT_Exp_Control(variables, {}, event, None, None, None, None)
    control._run_experiment_impl = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    control.clear_up = lambda: None

    control.run_experiment()

    assert event.was_set
    assert "boom" in variables.experiment_error


def test_safe_off_does_not_overwrite_failed_state():
    variables = SimpleNamespace(experiment_state=ExperimentState.FAILED.value, experiment_error="original", hardware_safe=False)
    control = APT_Exp_Control(variables, {}, None, None, None, None, None)
    control._outputs_safe = False
    control._vdc_active = lambda: False
    control._vp_active = lambda: False
    control._signal_generator_active = lambda: False

    control.safe_outputs_off()

    assert variables.hardware_safe is True
    assert variables.experiment_state == ExperimentState.FAILED.value
    assert variables.experiment_error == "original"


def test_run_config_rejects_zero_frequency():
    variables = SimpleNamespace(
        ex_freq=0, ex_time=1, max_ions=1, vdc_min=1, vdc_max=2,
        v_p_min=1, v_p_max=2, pulse_fraction=1, pulse_frequency=1,
        detection_rate=1, pulse_amp_per_supply_voltage=1, counter_source="TDC",
        criteria_time=True, criteria_ions=True,
    )
    conf = {"max_vdc": 10, "min_vp": 0, "max_vp": 10, "pulse_fraction_max": 20}
    try:
        validate_run_parameters(variables, conf)
    except ParameterError as exc:
        assert "frequency" in str(exc).lower()
    else:
        raise AssertionError("zero experiment frequency was accepted")


def test_laser_run_does_not_require_voltage_pulse_range():
    variables = SimpleNamespace(
        ex_freq=1, ex_time=1, max_ions=1, vdc_min=1, vdc_max=2,
        v_p_min=0, v_p_max=0, pulse_fraction=1, pulse_frequency=1,
        detection_rate=1, pulse_amp_per_supply_voltage=1, counter_source="TDC",
        pulse_mode="Laser", criteria_time=True, criteria_ions=True,
    )
    conf = {"max_vdc": 10, "min_vp": 100, "max_vp": 1000, "pulse_fraction_max": 20}

    assert validate_run_parameters(variables, conf).pulse_mode == "Laser"


def test_recover_chunks_builds_detector_hdf5(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    np.save(chunk_dir / "x_chunk_1.npy", np.array([1.0, 2.0], dtype=np.float64))
    np.save(chunk_dir / "x_chunk_2.npy", np.array([3.0], dtype=np.float64))
    output = tmp_path / "recovered.h5"

    result = recover_chunks(chunk_dir, output)

    assert result["recovered"] is True
    assert result["datasets_written"]["dld/x"] == 3
