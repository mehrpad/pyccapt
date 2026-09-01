from datetime import datetime, timedelta

from pyccapt.control.gui.vacuum_history import VacuumHistory


def test_vacuum_history_filters_requested_window():
    history = VacuumHistory()
    start = datetime(2026, 1, 1)
    for minute in range(5):
        history.add(start + timedelta(minutes=minute), (1, 2, 3, 4))

    samples = history.window(60)
    assert [sample.timestamp for sample in samples] == [start + timedelta(minutes=3), start + timedelta(minutes=4)]


def test_vacuum_history_archives_old_samples_and_limits_plot_points():
    history = VacuumHistory(raw_seconds=10, archive_seconds=5, retention_seconds=100)
    start = datetime(2026, 1, 1)
    for second in range(31):
        history.add(start + timedelta(seconds=second), (second + 1,) * 4)

    samples = history.window(100, max_points=4)
    assert len(samples) <= 5
    assert samples[-1].timestamp == start + timedelta(seconds=30)
