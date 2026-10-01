"""Compact experiment queue and plan editing for the main control window."""
from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from pyccapt.control.core import experiment_plan, runtime
from pyccapt.control.gui import main_parameters


class ExperimentEditor(QtWidgets.QDialog):
    """Edit one typed experiment, with units and explicit sample selection."""

    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit experiment')
        self.resize(540, 600)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)
        self.fields = {}
        for key in ('sample_id',) + experiment_plan.REQUIRED_FIELDS + tuple(experiment_plan.OPTIONAL_DEFAULTS):
            value = item.get(key, experiment_plan.OPTIONAL_DEFAULTS.get(key, ''))
            if key == 'sample_id':
                widget = QtWidgets.QComboBox()
                widget.addItems(['No stage mapping', '1', '2', '3'])
                widget.setCurrentIndex(int(value) if value else 0)
            elif key in experiment_plan.CHOICES:
                widget = QtWidgets.QComboBox()
                widget.addItems(experiment_plan.CHOICES[key])
                widget.setCurrentText(str(value))
            elif type(value) is bool:
                widget = QtWidgets.QCheckBox()
                widget.setChecked(value)
            else:
                widget = QtWidgets.QLineEdit(str(value))
            unit = experiment_plan.UNITS.get(key)
            form.addRow(f'{key} ({unit})' if unit else key, widget)
            self.fields[key] = widget
        scroll.setWidget(content)
        outer.addWidget(scroll)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def item(self):
        values = {}
        for key, widget in self.fields.items():
            if key == 'sample_id':
                if widget.currentIndex():
                    values[key] = widget.currentIndex()
            elif isinstance(widget, QtWidgets.QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                values[key] = widget.currentText()
            else:
                raw = widget.text()
                try:
                    if key in experiment_plan.INTEGER_FIELDS or key == 'email_interval_events':
                        values[key] = int(raw)
                    elif key in experiment_plan.NUMBER_FIELDS:
                        values[key] = float(raw)
                    else:
                        values[key] = raw
                except ValueError as exc:
                    raise experiment_plan.PlanError(f'{key}: enter a valid number') from exc
        return values


class ExperimentPlanGuiMixin:
    def _setup_experiment_plan(self):
        self.plan_items = []
        self._plan_path = ''
        self._batch_items = None
        self.result_list = []
        self.plan_panel = QtWidgets.QWidget(self.centralwidget)
        layout = QtWidgets.QVBoxLayout(self.plan_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plan_label = QtWidgets.QLabel('No experiment plan loaded')
        layout.addWidget(self.plan_label)
        actions = QtWidgets.QHBoxLayout()
        self.plan_buttons = {}
        for label, callback in (
            ('Load TOML', self._load_plan_dialog), ('Save As', self._save_plan_dialog),
            ('Add', self._add_plan_row), ('Edit', self._edit_plan_row),
            ('Duplicate', self._duplicate_plan_row), ('Remove', self._remove_plan_row),
            ('↑', lambda: self._move_plan_row(-1)), ('↓', lambda: self._move_plan_row(1)),
            ('Import TextLine', self._import_textline_dialog),
        ):
            button = QtWidgets.QPushButton(label)
            button.setFixedWidth(max(28, button.fontMetrics().horizontalAdvance(label)+20))
            button.clicked.connect(callback)
            actions.addWidget(button)
            self.plan_buttons[label] = button
        actions.addStretch()
        layout.addLayout(actions)
        self.plan_table = QtWidgets.QTableWidget(0, 7)
        self.plan_table.setHorizontalHeaderLabels(
            ['Sample', 'Experiment', 'Mode', 'DC range (V)', 'Pulse (kHz)', 'Target (%)', 'Stop when'])
        self.plan_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.plan_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.plan_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.plan_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.plan_table.setMinimumHeight(90)
        self.plan_table.setMaximumHeight(110)
        self.plan_table.cellDoubleClicked.connect(self._edit_plan_row)
        layout.addWidget(self.plan_table)
        # Use the freed right column for advanced form fields; queues span both
        # form columns so their buttons and summaries fit a compact window.
        self.verticalLayout.removeItem(self.gridLayout)
        self.verticalLayout_2.insertLayout(1, self.gridLayout)
        for index in reversed(range(self.verticalLayout_2.count())):
            if self.verticalLayout_2.itemAt(index).spacerItem() is not None:
                self.verticalLayout_2.takeAt(index)
        self.verticalLayout_2.removeWidget(self.text_line)
        keep = {self.parameters_source, self.label_173, self.ex_number, self.label_183}
        self._plan_form_widgets = [grid.itemAt(i).widget()
                                   for grid in (self.gridLayout_4, self.gridLayout_3,
                                                self.gridLayout_2, self.gridLayout)
                                   for i in range(grid.count())
                                   if grid.itemAt(i).widget() is not None and grid.itemAt(i).widget() not in keep]
        self._plan_form_widgets.extend(self.verticalLayout.itemAt(i).widget()
                                       for i in range(self.verticalLayout.count())
                                       if self.verticalLayout.itemAt(i).widget() is not None)
        self.gridLayout_6.removeItem(self.verticalLayout)
        self.gridLayout_6.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.verticalLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.verticalLayout_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.gridLayout_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.gridLayout_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.gridLayout_6.addWidget(self.plan_panel, 2, 0, 1, 2)
        self.gridLayout_6.addWidget(self.text_line, 2, 0, 1, 2)
        self.gridLayout_6.removeWidget(self.start_button)
        self.gridLayout_6.addWidget(self.start_button, 3, 2, 1, 1)
        self.plan_panel.hide()
        self.text_line.setMinimumHeight(100)
        self.text_line.setMaximumHeight(140)
        self.text_line.hide()
        self.parameters_source.addItem('TOML Plan')

    def _plan_locked(self):
        return bool(self.variables.start_flag or self.variables.sample_selection_locked or
                    getattr(self, '_alignment_batch', []))

    def _validate_plan_items(self, items):
        return main_parameters.validate_experiment_queue(
            items, self.conf, self.variables.pulse_amp_per_supply_voltage)

    def load_experiment_plan(self, path):
        if self._plan_locked():
            raise main_parameters.ParameterError('Wait for the current experiment or sample sequence to finish')
        items = self._validate_plan_items(experiment_plan.load_plan(path))
        self.plan_items = items
        self._plan_path = str(path)
        self._refresh_plan_table()
        self.parameters_source.setCurrentText('TOML Plan')

    def _load_plan_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.centralwidget, 'Load experiment plan', str(runtime.project_path('files')), 'TOML plans (*.toml)')
        if path:
            try:
                self.load_experiment_plan(path)
            except ValueError as exc:
                self.error_message(str(exc))

    def _save_plan_dialog(self):
        if self._plan_locked():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.centralwidget, 'Save experiment plan', self._plan_path or 'experiments.toml', 'TOML plans (*.toml)')
        if path:
            if not path.lower().endswith('.toml'):
                path += '.toml'
            try:
                experiment_plan.save_plan(path, self._validate_plan_items(self.plan_items))
                self._plan_path = path
                self._refresh_plan_table()
            except (OSError, ValueError) as exc:
                self.error_message(str(exc))

    def _refresh_plan_table(self, selected=0):
        self.plan_table.setRowCount(len(self.plan_items))
        for index, item in enumerate(self.plan_items):
            stops = []
            if item['criteria_time']:
                stops.append(f"{item['ex_time']} s")
            if item['criteria_ions']:
                stops.append(f"{item['max_ions']} ions")
            if item['criteria_vdc']:
                stops.append('max DC')
            values = [item.get('sample_id', '—'), item['ex_name'], item['pulse_mode'],
                      f"{item['vdc_min']}–{item['vdc_max']}", item['pulse_frequency'],
                      item['detection_rate_init'], ' or '.join(stops)]
            for column, value in enumerate(values):
                self.plan_table.setItem(index, column, QtWidgets.QTableWidgetItem(str(value)))
        if self.plan_items:
            self.plan_table.selectRow(min(selected, len(self.plan_items)-1))
        name = Path(self._plan_path).name if self._plan_path else 'Unsaved plan'
        self.plan_label.setText(f'{name} • {len(self.plan_items)} experiments')
        self.plan_label.setToolTip(self._plan_path)

    def _edit_plan_item(self, item):
        dialog = ExperimentEditor(item, self.centralwidget)
        # Keep the dialog open after validation errors so edits are retained.
        while dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            try:
                return self._validate_plan_items([dialog.item()])[0]
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(dialog, 'Invalid experiment', str(exc))
        return None

    def _add_plan_row(self):
        if self._plan_locked():
            return
        # Read the visible form directly; shared variables may describe a prior batch row.
        mapping = {'ex_user': 'ex_user', 'hit_displayed': None}
        item = {}
        for key in experiment_plan.REQUIRED_FIELDS + tuple(experiment_plan.OPTIONAL_DEFAULTS):
            if key == 'hit_displayed':
                item[key] = int(self.variables.hit_display)
                continue
            widget = getattr(self, mapping.get(key, 'email_interval' if key == 'email_interval_events' else key))
            if isinstance(widget, QtWidgets.QCheckBox):
                item[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                item[key] = widget.currentText()
            else:
                item[key] = widget.text()
        for key in experiment_plan.INTEGER_FIELDS + ('email_interval_events',):
            try:
                item[key] = int(item[key])
            except ValueError:
                item[key] = 0
        for key in experiment_plan.NUMBER_FIELDS:
            try:
                item[key] = float(item[key])
            except ValueError:
                item[key] = 0.0
        edited = self._edit_plan_item(item)
        if edited:
            self.plan_items.append(edited)
            self._plan_path = ''
            self._refresh_plan_table(len(self.plan_items)-1)

    def _edit_plan_row(self, *_args):
        row = self.plan_table.currentRow()
        if self._plan_locked() or row < 0:
            return
        edited = self._edit_plan_item(self.plan_items[row])
        if edited:
            self.plan_items[row] = edited
            self._plan_path = ''
            self._refresh_plan_table(row)

    def _duplicate_plan_row(self):
        row = self.plan_table.currentRow()
        if self._plan_locked() or row < 0:
            return
        self.plan_items.insert(row+1, dict(self.plan_items[row]))
        self._plan_path = ''
        self._refresh_plan_table(row+1)

    def _remove_plan_row(self):
        row = self.plan_table.currentRow()
        if self._plan_locked() or row < 0:
            return
        self.plan_items.pop(row)
        self._plan_path = ''
        self._refresh_plan_table(row)

    def _move_plan_row(self, direction):
        row = self.plan_table.currentRow()
        other = row+direction
        if self._plan_locked() or row < 0 or not 0 <= other < len(self.plan_items):
            return
        self.plan_items[row], self.plan_items[other] = self.plan_items[other], self.plan_items[row]
        self._plan_path = ''
        self._refresh_plan_table(other)

    def _import_textline_dialog(self):
        if self._plan_locked():
            return
        text, accepted = QtWidgets.QInputDialog.getMultiLineText(
            self.centralwidget, 'Import TextLine', 'Paste legacy parameter blocks:', self.text_line.toPlainText())
        if accepted:
            try:
                items = main_parameters.parse_textline_experiments(text)
                self.plan_items = self._validate_plan_items(items)
                self._plan_path = ''
                self._refresh_plan_table()
            except ValueError as exc:
                self.error_message(str(exc))

    def _freeze_experiment_queue(self):
        """Resolve every row once before starting; never reread a file mid-batch."""
        if self.parameters_source.currentText() == 'TOML Plan':
            self._batch_items = self._validate_plan_items(self.plan_items)
        elif self.parameters_source.currentText() == 'TextLine':
            self._batch_items = self._validate_plan_items(
                main_parameters.parse_textline_experiments(self.text_line.toPlainText()))
        else:
            self._batch_items = None
        if self._batch_items is not None:
            self.result_list = [dict(item) for item in self._batch_items]
        self.variables.experiment_plan_snapshot = {}

    def _publish_plan_row(self, index):
        if self.parameters_source.currentText() == 'TOML Plan':
            self.variables.experiment_plan_snapshot = {
                'source': self._plan_path, 'queue_index': index+1,
                'experiment': dict(self.result_list[index]),
            }
            self.plan_table.selectRow(index)
        else:
            self.variables.experiment_plan_snapshot = {}
