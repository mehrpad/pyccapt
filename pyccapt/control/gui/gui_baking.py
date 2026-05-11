import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer

try:
    from mcculw import ul
    from mcculw.enums import TempScale, TInOptions
except Exception as e:
    ul = None
    TempScale = None
    TInOptions = None
    print('Cannot import mcculw library')
    print(e)

# Local module and scripts
from pyccapt.control.core import runtime
from pyccapt.control.gui import gui_pumps_vacuum, tooltips
from pyccapt.control.devices import initialize_devices


class Ui_Baking(object):
    def __init__(self, variables, conf, SignalEmitter_Pumps_Vacuum, parent=None):
        """
        Initialize the UiBaking class.

        Args:
                variables: Instance of variables class.
                conf: Configuration settings.
                parent: Parent widget (default is None).
        """
        self.variables = variables
        self.conf = conf
        self.emitter = SignalEmitter_Pumps_Vacuum
        self.parent = parent
        self.now = datetime.now()
        self.running = True
        self.vacuum_main = 0
        self.vacuum_buffer = 0
        self.vacuum_load_lock = 0
        self.vacuum_cryo_load_lock = 0
        self._warned_messages = set()

        self.data = pd.DataFrame(
            columns=[
                'data',
                'Time',
                'timestamp',
                'MC_vacuum',
                'BC_vacuum',
                'LL_vacuum',
                'CLL_vacuum',
                'MC_NEG',
                'MC_Det',
                'Mc_Top',
                'MC_Gate',
                'BC_Top',
                'BC_Pump',
                'CLL_gate',
                'LL_pump',
            ]
        )
        now_time = self.now.strftime("%d-%m-%Y_%H-%M-%S")
        self.save_path = runtime.project_path("files", "logs", "baking", now_time)
        os.makedirs(self.save_path, mode=0o777, exist_ok=True)
        self.file_name = str(self.save_path / f'baking_logging_{now_time}.csv')
        self.file_name_backup = str(self.save_path / f'backup_baking_logging_{now_time}.csv')

    def setupUi(self, Baking):
        """
        setupUi function.
        Args:
                Baking: Parent widget.

        Returns:
                None
        """
        Baking.setWindowIcon(QtGui.QIcon('./files/logo.png'))
        Baking.setObjectName("Baking")
        Baking.resize(820, 757)
        self.gridLayout_2 = QtWidgets.QGridLayout(Baking)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        # self.tempretures = QtWidgets.QGraphicsView(parent=Baking)
        self.tempretures = pg.PlotWidget(parent=Baking)
        self.tempretures.setMinimumSize(QtCore.QSize(800, 500))
        self.tempretures.setObjectName("tempretures")
        self.gridLayout.addWidget(self.tempretures, 0, 0, 1, 1)
        self.save_data = QtWidgets.QPushButton(parent=Baking)
        self.save_data.setMinimumSize(QtCore.QSize(0, 25))
        self.save_data.setStyleSheet("QPushButton{\nbackground: rgb(193, 193, 193)\n}")
        self.save_data.setObjectName("save_data")
        self.gridLayout.addWidget(self.save_data, 2, 0, 1, 1)
        # self.presures = QtWidgets.QGraphicsView(parent=Baking)
        self.presures = pg.PlotWidget(parent=Baking)
        self.presures.setMinimumSize(QtCore.QSize(800, 200))
        self.presures.setObjectName("presures")
        self.gridLayout.addWidget(self.presures, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Baking)
        QtCore.QMetaObject.connectSlotsByName(Baking)
        tooltips.apply_tooltips(self, tooltips.BAKING_TOOLTIPS)
        ###
        read_thread = threading.Thread(target=self.read)
        read_thread.setDaemon(True)
        read_thread.start()

        self.save_data.clicked.connect(self.save_data_csv)

        # Create a QTimer to hide the warning message after 8 seconds
        self.timer = QTimer(self.parent)
        self.timer.timeout.connect(self.plot)
        self.timer.start(1000)

        self.tempretures.addLegend()
        styles = {"color": "#f00", "font-size": "12px"}
        self.tempretures.setLabel("left", "Temperature (C)", **styles)
        self.tempretures.setLabel("bottom", "Time (sec)", **styles)

        self.presures.addLegend()
        styles = {"color": "#f00", "font-size": "12px"}
        self.presures.setLabel("left", "Pressure (mBar)", **styles)
        self.presures.setLabel("bottom", "Time (sec)", **styles)

        self.emitter.vacuum_main.connect(self.update_vacuum_main)
        self.emitter.vacuum_buffer.connect(self.update_vacuum_buffer)
        self.emitter.vacuum_load_lock.connect(self.update_vacuum_load)
        self.emitter.vacuum_cryo_load_lock.connect(self.update_vacuum_cryo_load_lock)

        # Add grids to the plots
        self.tempretures.showGrid(x=True, y=True)
        self.presures.showGrid(x=True, y=True)  # Add grid to pressure plot

    def retranslateUi(self, Baking):
        """
        retranslateUi function.
        Args:
                Baking: Parent widget.

        Returns:
                None
        """
        _translate = QtCore.QCoreApplication.translate
        ###
        #  Baking.setWindowTitle(_translate("Baking", "Form"))
        Baking.setWindowTitle(_translate("Baking", "PyCCAPT Baking"))
        Baking.setWindowIcon(QtGui.QIcon(str(runtime.project_path("files", "logo.png"))))
        ###
        self.save_data.setText(_translate("Baking", "Save CSV"))

    def update_vacuum_main(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value != -1:
            self.vacuum_main = value

    def update_vacuum_buffer(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value != -1:
            self.vacuum_buffer = value

    def update_vacuum_load(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value != -1:
            self.vacuum_load_lock = value

    def update_vacuum_cryo_load_lock(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value != -1:
            self.vacuum_cryo_load_lock = value

    def _warn_once(self, key, message):
        if key not in self._warned_messages:
            print(message)
            self._warned_messages.add(key)

    def _read_temperatures(self):
        """Return DAQ temperatures or NaN placeholders when the DAQ is unavailable."""
        if ul is None or TempScale is None or TInOptions is None:
            self._warn_once(
                'daq_unavailable',
                'DAQ temperature input is unavailable. Baking temperatures will remain empty until mcculw and cbw64.dll are installed.',
            )
            return np.full(8, np.nan, dtype=float)

        board_num = 0
        value_temperature = []
        try:
            for i in range(8):
                options = TInOptions.NOFILTER
                val = float(ul.t_in(board_num, i, TempScale.CELSIUS, options))
                value_temperature.append(round(val, 3))
        except Exception as e:
            self._warn_once('daq_read_error', f'Error reading baking temperatures: {e}')
            return np.full(8, np.nan, dtype=float)

        return np.array(value_temperature, dtype=np.dtype(float))

    def read(self):
        """
        Read function.

        Args:
                None
        Returns:
                None
        """
        tpg = None
        E_AGC_ll = None
        E_AGC_cll = None
        if not self.variables.flag_pumps_vacuum_start:
            try:
                tpg = initialize_devices.TPG362(port=self.variables.COM_PORT_gauge_mc)
            except Exception as e:
                self._warn_once('baking_tpg_connect', f"Error connecting to TPG362:{e}")
                tpg = None
            if self.conf['COM_PORT_gauge_ll'] != "off":
                try:
                    E_AGC_ll = initialize_devices.EdwardsAGC(self.variables.COM_PORT_gauge_ll, self.variables)
                except Exception as e:
                    self._warn_once('baking_ll_connect', f"Error connecting to LL gauge:{e}")
                    E_AGC_ll = None
            if self.conf['COM_PORT_gauge_cll'] != "off":
                try:
                    E_AGC_cll = initialize_devices.EdwardsAGC(self.variables.COM_PORT_gauge_cll, self.variables)
                except Exception as e:
                    self._warn_once('baking_cll_connect', f"Error connecting to CLL gauge:{e}")
                    E_AGC_cll = None

        index = 0
        desired_period = 1.0
        if self.conf['baking'] == 'on':
            while self.running:
                start_time = time.perf_counter()
                # print('-----------', index, 'seconds', '--------------')
                if not self.variables.flag_pumps_vacuum_start:
                    if tpg is not None:
                        try:
                            gauge_bc, _ = tpg.pressure_gauge(1)
                        except Exception as e:
                            self._warn_once('baking_bc_read', f"Error reading BC:{e}")
                            gauge_bc = -1
                        try:
                            gauge_mc, _ = tpg.pressure_gauge(2)
                        except Exception as e:
                            self._warn_once('baking_mc_read', f"Error reading MC:{e}")
                            gauge_mc = -1
                    else:
                        gauge_bc = -1
                        gauge_mc = -1

                    if E_AGC_ll is not None:
                        response = initialize_devices.command_edwards(
                            self.conf,
                            self.variables,
                            'pressure',
                            E_AGC=E_AGC_ll,
                            status='load_lock',
                        )
                        try:
                            gauge_ll = float(response.replace(';', ' ').split()[2]) * 0.01
                        except Exception as e:
                            self._warn_once('baking_ll_read', f"Error reading LL:{e}")
                            gauge_ll = -1
                    else:
                        gauge_ll = -1

                    if E_AGC_cll is not None:
                        response = initialize_devices.command_edwards(
                            self.conf,
                            self.variables,
                            'pressure',
                            E_AGC=E_AGC_cll,
                            status='cryo_load_lock',
                        )
                        try:
                            gauge_cll = float(response.replace(';', ' ').split()[2]) * 0.01
                        except Exception as e:
                            self._warn_once('baking_cll_read', f"Error reading CLL:{e}")
                            gauge_cll = -1
                    else:
                        gauge_cll = -1
                else:
                    gauge_bc = self.vacuum_buffer
                    gauge_mc = self.vacuum_main
                    gauge_ll = self.vacuum_load_lock
                    gauge_cll = self.vacuum_cryo_load_lock

                value_temperature = self._read_temperatures()

                new_row = [
                    self.now.strftime("%d-%m-%Y"),
                    datetime.now().strftime('%H:%M:%S'),
                    index,
                    gauge_mc,
                    gauge_bc,
                    gauge_ll,
                    gauge_cll,
                    value_temperature[0],
                    value_temperature[1],
                    value_temperature[2],
                    value_temperature[3],
                    value_temperature[4],
                    value_temperature[5],
                    value_temperature[6],
                    value_temperature[7],
                ]

                self.data.loc[len(self.data)] = new_row

                index = index + 1
                if index % 20 == 0:
                    try:
                        self.data.to_csv(self.file_name, sep=';', index=False)
                        # Successful save - reset the dedup so the next
                        # failure (e.g. user re-opens the file) prints again.
                        self._warned_messages.discard('baking_csv_save')
                    except Exception as e:
                        self.data.to_csv(self.file_name_backup, sep=';', index=False)
                        self._warn_once(
                            'baking_csv_save',
                            f'csv File cannot be saved (close the csv file): {e}',
                        )

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                remaining_time = desired_period - elapsed_time

                if remaining_time > 0:
                    time.sleep(remaining_time)

    def plot(self):
        """
        Plot function.

        Args:
                None
        Returns:
                None
        """
        if self.data.empty:
            return

        time_range = 900  # 15 minutes
        time = self.data['timestamp'].tail(time_range).to_numpy()
        MC_NEG = self.data['MC_NEG'].tail(time_range).to_numpy()
        MC_Det = self.data['MC_Det'].tail(time_range).to_numpy()
        Mc_Top = self.data['Mc_Top'].tail(time_range).to_numpy()
        MC_Gate = self.data['MC_Gate'].tail(time_range).to_numpy()
        BC_Top = self.data['BC_Top'].tail(time_range).to_numpy()
        BC_Pump = self.data['BC_Pump'].tail(time_range).to_numpy()
        MC_vacuum = self.data['MC_vacuum'].tail(time_range).to_numpy()
        BC_vacuum = self.data['BC_vacuum'].tail(time_range).to_numpy()
        LL_vacuum = self.data['LL_vacuum'].tail(time_range).to_numpy()
        CLL_vacuum = self.data['CLL_vacuum'].tail(time_range).to_numpy()
        CLL_gate = self.data['CLL_gate'].tail(time_range).to_numpy()
        LL_pump = self.data['LL_pump'].tail(time_range).to_numpy()
        if len(time) == 0:
            return
        self.tempretures.clear()
        self.presures.clear()

        try:
            # Plot the latest values in the legend so the live view doubles as a quick status panel.
            self.tempretures.plot(time, MC_NEG, pen='b', name='MC_NEG %.2f' % MC_NEG[-1])
            self.tempretures.plot(time, MC_Det, pen='g', name='MC_Det %.2f' % MC_Det[-1])
            self.tempretures.plot(time, Mc_Top, pen='r', name='Mc_Top %.2f' % Mc_Top[-1])
            self.tempretures.plot(time, MC_Gate, pen='c', name='MC_Gate %.2f' % MC_Gate[-1])
            self.tempretures.plot(time, BC_Top, pen='m', name='BC_Top %.2f' % BC_Top[-1])
            self.tempretures.plot(time, BC_Pump, pen='y', name='BC_Pump %.2f' % BC_Pump[-1])
            self.tempretures.plot(time, CLL_gate, pen='orange', name='CLL_gate %.2f' % CLL_gate[-1])
            self.tempretures.plot(time, LL_pump, pen='w', name='LL_pump %.2f' % LL_pump[-1])

            self.presures.plot(time, MC_vacuum, pen='r', name='MC_vacuum %s' % MC_vacuum[-1])
            self.presures.plot(time, BC_vacuum, pen='g', name='BC_vacuum %s' % BC_vacuum[-1])
            self.presures.plot(time, LL_vacuum, pen='b', name='LL_vacuum %s' % LL_vacuum[-1])
            self.presures.plot(time, CLL_vacuum, pen='c', name='CLL_vacuum %s' % CLL_vacuum[-1])
        except Exception as e:
            self._warn_once('baking_plot_error', f'Error in plotting the data: {e}')

        self.tempretures.enableAutoRange(axis='x')
        self.presures.enableAutoRange(axis='x')

    def save_data_csv(self):
        """Prompt the user for a target file and save the baking log CSV.

        Pops up the OS Save-As dialog pre-filled with a sensible default
        filename (``manual_save_<timestamp>.csv``) under the existing
        baking-log folder. Cancel leaves nothing on disk.
        """
        now = datetime.now()
        now_time = now.strftime("%d-%m-%Y_%H-%M-%S")
        default_path = str(self.save_path / f'manual_save_{now_time}.csv')
        parent = self.parent if isinstance(self.parent, QtWidgets.QWidget) else None
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent,
            "Save baking log as…",
            default_path,
            "CSV files (*.csv);;All files (*)",
        )
        if not file_path:
            return  # user cancelled
        # Default to .csv if the user didn't type an extension.
        if not os.path.splitext(file_path)[1]:
            file_path += '.csv'
        try:
            self.data.to_csv(file_path, sep=';', index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                parent,
                "Save failed",
                f"Could not save baking log:\n{exc}",
            )

    def stop(self):
        """
        Stop function.

        Args:
                None
        Returns:
                None
        """
        self.running = False
        self.timer.stop()  # Stop the QTimer


class BakingWindow(QtWidgets.QWidget):
    closed = QtCore.pyqtSignal()  # Define a custom closed signal

    def __init__(self, gui_baking, *args, **kwargs):
        """
        Initialize the BakingWindow class.

        Args:
                gui_baking: An instance of the GUI baking class.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.gui_baking = gui_baking

    def closeEvent(self, event):
        """
        Override the close event to stop the background thread and perform additional cleanup if needed.

        Args:
                event: The close event.
        """
        if getattr(self, "force_close", False):
            event.accept()
            return
        event.ignore()
        self.hide()
        self.closed.emit()

    def setWindowStyleFusion(self):
        # Set the Fusion style
        QtWidgets.QApplication.setStyle("Fusion")


if __name__ == "__main__":
    try:
        conf, _ = runtime.load_project_config()
    except Exception as exc:
        print('Can not load the configuration file')
        print(exc)
        sys.exit()

    shared = runtime.create_shared_context(conf)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    Baking = QtWidgets.QWidget()
    SignalEmitter_Pumps_Vacuum = gui_pumps_vacuum.SignalEmitter()
    ui = Ui_Baking(shared.variables, conf, SignalEmitter_Pumps_Vacuum)
    ui.setupUi(Baking)
    Baking.show()
    sys.exit(app.exec())
