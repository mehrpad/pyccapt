from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyccapt.calibration.core.interactive_point_identification import AnnotationFinder, AnnoteFinder, distances
from pyccapt.calibration.core.share_variables import Variables


def test_annotation_selection_and_deselection_updates_shared_variables():
    variables = Variables()
    figure, axis = plt.subplots()
    finder = AnnotationFinder([10.0], [5.0], ["1"], variables, ax=axis, xtol=2.0, ytol=2.0)

    finder.draw_annotation(axis, 10.0, 5.0, "1")
    assert variables.peaks_x_selected == [10.0]
    assert variables.peaks_index_list == [0]

    finder.deselect_point(axis, 10.0, 5.0, "1")
    assert variables.peaks_x_selected == []
    assert variables.peaks_index_list == []
    plt.close(figure)


def test_legacy_callback_name_still_behaves_for_click_events():
    variables = Variables()
    figure, axis = plt.subplots()
    finder = AnnotationFinder([1.0], [1.0], ["1"], variables, ax=axis, xtol=0.5, ytol=0.5)

    left_click = SimpleNamespace(inaxes=axis, xdata=1.0, ydata=1.0, button=1)
    finder.annotates_plotter(left_click)
    assert variables.peaks_x_selected == [1.0]
    assert variables.peaks_index_list == [0]

    right_click = SimpleNamespace(inaxes=axis, xdata=1.0, ydata=1.0, button=3)
    finder.annotates_plotter(right_click)
    assert variables.peaks_x_selected == []
    assert variables.peaks_index_list == []
    plt.close(figure)


def test_canonical_module_exposes_expected_aliases():
    assert AnnotationFinder is not None
    assert AnnoteFinder is not None
    assert distances(0, 3, 0, 4) == 5.0

