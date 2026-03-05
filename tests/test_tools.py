import pytest
from unittest.mock import patch
import os
import pandas as pd
# Local module and scripts
from pyccapt.calibration.core import tools
from pyccapt.calibration.data_tools import data_tools

p = os.path.abspath(os.path.join("", "."))
path = p + '//data//data_tests//'
test_dataset_2 = 'AL_data_b.h5'

@pytest.fixture()
def mc():
    file_name = path + 'mc_seb_array.h5'
    mc = pd.read_hdf(file_name, mode='r').to_numpy()[0][0]
    return mc


@pytest.fixture()
def mc_voltage_corr():
    file_name = path + 'mc_seb_latest.h5'
    mc_voltage_corr = pd.read_hdf(file_name, mode='r').to_numpy()[0][0]
    return mc_voltage_corr


@pytest.fixture()
def fitpeak():
    file_name = path + 'fitPeak_latest.h5'
    fitpeak = pd.read_hdf(file_name, mode='r').to_numpy()[0][0]
    return fitpeak


@pytest.fixture()
def dld_data():
    file_name = path + 'AL_data_b_cropped.h5'
    data = pd.read_hdf(file_name, mode='r').to_numpy()[0][0]
    return data


def test_massSpecPlot_check_returnType(mc):
    bin = 0.1
    response = tools.massSpecPlot(mc, bin, mode='count', percent=50, prominence=500, peaks_find=False, plot=False,
                                  distance=None,
                                  fig_name='test')
    print("response", response)
    assert isinstance(response, tuple)


@patch.object(tools.logger, "info")
def test_massSpecPlot_mode_equal_count(mock, mc):
    bin = 0.1
    response = tools.massSpecPlot(mc, bin, mode='count', percent=50, prominence=500, peaks_find=False, plot=False,
                                  distance=None,
                                  fig_name='test')
    mock.assert_called_with("Selected Mode = count")


@patch.object(tools.logger, "info")
def test_massSpecPlot_mode_equal_uppercase_count(mock, mc):
    bin = 0.1
    response = tools.massSpecPlot(mc, bin, mode='COUNT', percent=50, prominence=500, peaks_find=False, plot=False,
                                  distance=None,
                                  fig_name='test')
    mock.assert_called_with("Mode not selected")


@patch.object(tools.logger, "info")
def test_massSpecPlot_mode_equal_normalised(mock, mc):
    bin = 0.1
    response = tools.massSpecPlot(mc, bin, mode='normalised', percent=50, prominence=500, peaks_find=False, plot=False,
                                  distance=None,
                                  fig_name='test')
    mock.assert_called_with("Selected Mode = normalised")


@patch.object(tools.logger, "info")
def test_massSpecPlot_mode_equal_none(mock, mc):
    bin = 0.1
    response = tools.massSpecPlot(mc, bin, mode='random', percent=50, peaks_find=False, prominence=500, plot=False,
                                  distance=None,
                                  fig_name='test')
    mock.assert_called_with("Mode not selected")


def test_history_ex_check_return_type(dld_data, mc):
    dld_highVoltage = dld_data['dld/high_voltage'].to_numpy()
    dld_highVoltage = dld_highVoltage[:len(mc)]
    response = tools.history_ex(mc, dld_highVoltage)
    assert isinstance(response, list)


def test_history_ex_check_return_type_of_each_element(dld_data, mc):
    import numpy
    dld_highVoltage = dld_data['dld/high_voltage'].to_numpy()
    dld_highVoltage = dld_highVoltage[:len(mc)]
    response = tools.history_ex(mc, dld_highVoltage)
    assert isinstance(response[0], numpy.ndarray)


@patch.object(tools.plt, "show")
def test_history_ex_plot_is_true(mock, dld_data, mc):
    variables.init()
    dld_highVoltage = dld_data['dld/high_voltage'].to_numpy()
    dld_highVoltage = dld_highVoltage[:len(mc)]
    response = tools.history_ex(mc, dld_highVoltage, plot=True)
    mock.assert_called()


def test_voltage_corr_check_return_type(dld_data, mc_voltage_corr, fitpeak):
    import numpy
    dld_highVoltage = dld_data['dld/high_voltage'].to_numpy()
    dld_t = dld_data['dld/t'].to_numpy()
    threshold = 60
    t0 = 51.74
    dld_t = dld_t - t0
    dld_t = dld_t[dld_t > threshold]
    ionsPerFitSegment = int(len(dld_t) / 70)
    dld_highVoltage = dld_highVoltage[:14799648]
    response = tools.voltage_corr(dld_highVoltage, mc_voltage_corr, fitpeak, ionsPerFitSegment)
    assert isinstance(response, numpy.ndarray)


@patch.object(tools.plt, "savefig")
def test_voltage_corr_fig_name_passed(mock, dld_data, mc_voltage_corr, fitpeak):
    variables.init()
    dld_highVoltage = dld_data['dld/high_voltage'].to_numpy()
    dld_t = dld_data['dld/t'].to_numpy()
    threshold = 60
    t0 = 51.74
    dld_t = dld_t - t0
    dld_t = dld_t[dld_t > threshold]
    ionsPerFitSegment = int(len(dld_t) / 70)
    dld_highVoltage = dld_highVoltage[:14799648]
    response = tools.voltage_corr(dld_highVoltage, mc_voltage_corr, fitpeak, ionsPerFitSegment, fig_name="test_plot")
    mock.assert_called()


@patch.object(tools.plt, "show")
def test_voltage_corr_plot_equal_true(mock, dld_data, mc_voltage_corr, fitpeak):

    dld_highVoltage = dld_data['dld/high_voltage'].to_numpy()
    dld_t = dld_data['dld/t'].to_numpy()
    threshold = 60
    t0 = 51.74
    dld_t = dld_t - t0
    dld_t = dld_t[dld_t > threshold]
    ionsPerFitSegment = int(len(dld_t) / 70)
    dld_highVoltage = dld_highVoltage[:14799648]
    response = tools.voltage_corr(dld_highVoltage, mc_voltage_corr, fitpeak, ionsPerFitSegment, plot=True)
    mock.assert_called()


@patch.object(tools.logger, "error")
def test_voltage_corr_incorrect_args_passed(mock, dld_data, mc_voltage_corr, fitpeak):
    dld_highVoltage = dld_data['dld/high_voltage']
    dld_t = dld_data['dld/t'].to_numpy()
    threshold = 60
    t0 = 51.74
    dld_t = dld_t - t0
    dld_t = dld_t[dld_t > threshold]
    ionsPerFitSegment = int(len(dld_t) / 70)
    dld_highVoltage = dld_highVoltage[:14799648]
    response = tools.voltage_corr(dld_highVoltage, mc_voltage_corr, fitpeak, ionsPerFitSegment, plot=True)
    mock.assert_called()


def test_bowl_corr_check_return_type(dld_data, mc_voltage_corr):
    import numpy
    dld_x = dld_data['dld/x'].to_numpy()
    dld_y = dld_data['dld/y'].to_numpy()
    mc_min = 21.42
    mc_max = 22.92
    mcIdeal = [22.02]
    dld_x = dld_x[:14799648]
    dld_y = dld_y[:14799648]
    response = tools.bowl_corr(dld_x, dld_y, mc_voltage_corr, mcIdeal=mcIdeal, mc_min=mc_min, mc_max=mc_max, plot=None,
                               fig_name=None)
    assert isinstance(response, numpy.ndarray)


@patch.object(tools.plt, "show")
def test_bowl_corr_plot_equal_true(mock, dld_data, mc_voltage_corr):
    dld_x = dld_data['dld/x'].to_numpy()
    dld_y = dld_data['dld/y'].to_numpy()
    mc_min = 21.42
    mc_max = 22.92
    mcIdeal = [22.02]
    dld_x = dld_x[:14799648]
    dld_y = dld_y[:14799648]
    response = tools.bowl_corr(dld_x, dld_y, mc_voltage_corr, mcIdeal=mcIdeal, mc_min=mc_min, mc_max=mc_max, plot=True,
                               fig_name=None)
    mock.assert_called()
