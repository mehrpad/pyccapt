"""Verified Origami settings and unit-aware readouts for the laser GUI."""
import datetime
import math
import time

from pyccapt.control.nkt_photonics import readback


class LaserReadoutMixin:
    def get_frequency(self, index):
        return getattr(self, '_frequency_table', {}).get(index, math.nan)

    def _invalidate_laser_readouts(self, reason):
        self.variables.laser_telemetry = {'monotonic': time.monotonic(), 'valid': False, 'error': str(reason)}
        for name in ('laser_average_power', 'laser_pulse_energy', 'laser_freq', 'laser_intensity'):
            setattr(self.variables, name, math.nan)
        for widget in (self.laser_power_disp, self.laser_pulse_energy_disp, self.laser_repetion_rate_disp):
            widget.display('-----')
        self._update_wavelength_nm_label()

    def _sync_controls_from_device(self, *, initial=False):
        if self.laser_device is None:
            self._invalidate_laser_readouts('CLI disconnected')
            return
        device = self.laser_device
        raw = {}
        try:
            if initial or not getattr(self, '_frequency_table', {}):
                raw['e_freq_available'] = device.freq_avaliable()
                self._frequency_table = readback.frequency_table(raw['e_freq_available'])
                self.laser_rate.blockSignals(True)
                self.laser_rate.clear()
                for index, hz in sorted(self._frequency_table.items()):
                    self.laser_rate.addItem(f'{hz:g}', index)
                self.laser_rate.blockSignals(False)
            for key, method in (('e_freq', device.FreqRead), ('e_div', device.DivRead),
                                ('e_power', device.AOMRead), ('ls_wavelength', device.wavelength_read)):
                raw[key] = method()
            index, div, aom = (readback.scalar(raw[k]) for k in ('e_freq', 'e_div', 'e_power'))
            wl = readback.wavelength(raw['ls_wavelength'])
            hz = self.get_frequency(index)
            if (not math.isfinite(hz) or div is None or div != int(div) or not 1 <= div <= 10000000
                    or aom is None or not 0 <= aom <= 4000 or wl is None):
                raise ValueError('Incomplete laser frequency, divider, AOM or wavelength readback')
            output_hz = hz/div
            # e_mlp is the IR monitor. Never relabel it as Green/DUV output.
            raw['e_mlp'] = device.read_average_power()
            ir_mw, ir_nj = readback.optical_values(raw['e_mlp'], output_hz)
            if wl == 0:
                mw, nj = ir_mw, ir_nj
                source = 'e_mlp (IR monitor; explicit reply unit)'
            else:
                raw['ls_output_power'] = device.power_read_dv_green()
                measured = readback.quantity(raw['ls_output_power'])
                # Manual defines harmonic monitor in W. Accept explicit power
                # units only; unsupported FHG replies remain unknown.
                if measured is None or not measured[1].endswith('W'):
                    mw = nj = math.nan
                else:
                    mw, nj = readback.optical_values(raw['ls_output_power'], output_hz)
                source = 'ls_output_power (selected harmonic monitor)'
            raw['status'] = device.StatusRead()
            code = readback.scalar(raw['status'])
            valid = code in (9, 33, 65, 129) and math.isfinite(mw) and math.isfinite(nj)
            snapshot = dict(monotonic=time.monotonic(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                valid=valid, raw=raw, source=source, wavelength=readback.WAVELENGTHS[wl][0],
                wavelength_index=wl, wavelength_nm=readback.WAVELENGTHS[wl][1],
                wavelength_basis='nominal selected harmonic, not a live spectral measurement',
                base_frequency_hz=hz, frequency_index=index, divider=int(div), output_frequency_hz=output_hz,
                aom_percent=aom/40., ir_power_mw=ir_mw, output_power_mw=mw if valid else math.nan,
                pulse_energy_nj=nj if valid else math.nan, status_code=code,
                location='laser internal monitor; not calibrated energy delivered to specimen',
                frequency_table=dict(self._frequency_table))
            self.variables.laser_telemetry = snapshot
            self.variables.laser_average_power = snapshot['output_power_mw']
            self.variables.laser_pulse_energy = snapshot['pulse_energy_nj']  # nJ; detector writer converts to pJ.
            self.variables.laser_intensity = snapshot['pulse_energy_nj']
            self.variables.laser_freq = hz
            self.variables.laser_division_factor = int(div)
            for widget, value in ((self.laser_power, aom/40.), (self.laser_divition_factor, int(div))):
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
            self.laser_rate.blockSignals(True)
            self.laser_rate.setCurrentIndex(self.laser_rate.findData(int(index)))
            self.laser_rate.blockSignals(False)
            self.laser_wavelegnth.blockSignals(True)
            name = readback.WAVELENGTHS[wl][0]
            if self.laser_wavelegnth.findText(name) < 0:
                self.laser_wavelegnth.addItem(name)
            self.laser_wavelegnth.setCurrentText(name)
            self.laser_wavelegnth.blockSignals(False)
            self._update_wavelength_nm_label()
            self._recompute_derived_readouts()
            self._apply_button_locks_for_status(raw['status'])
        except Exception as exc:
            self._invalidate_laser_readouts(exc)
            self.error_message(f'Laser readback unavailable: {exc}')

    def _recompute_derived_readouts(self):
        data = readback.fresh_snapshot(self.variables)
        for widget, key, scale in (
            (self.laser_power_disp, 'output_power_mw', .001),
            (self.laser_pulse_energy_disp, 'pulse_energy_nj', .001),
            (self.laser_repetion_rate_disp, 'output_frequency_hz', .001),
        ):
            value = data.get(key, math.nan)
            widget.display(value*scale if math.isfinite(value) else '-----')
        tooltip = ('Source: '+data.get('source', 'unavailable')+
                   '. Internal monitor estimate; not energy at the specimen. Missing readings show dashes.')
        self.laser_power_disp.setToolTip(tooltip)
        self.laser_pulse_energy_disp.setToolTip(tooltip)

    def _clamp_divider_to_min_output_rate(self):
        # Manual p123 permits 1..10,000,000. QSG p6 explicitly uses 40 kHz.
        self.laser_divition_factor.setMaximum(10_000_000)

    def _update_wavelength_nm_label(self):
        data = readback.fresh_snapshot(self.variables)
        nm = data.get('wavelength_nm')
        if hasattr(self, 'laser_wavelegnth_nm_label'):
            self.laser_wavelegnth_nm_label.setText(f'({nm:g} nm nominal)' if nm else '(not read back)')

    def _apply_laser_settings(self, code):
        """Apply only operator-requested settings; never reopen AOM after edits."""
        changed = False
        running = bool(self.variables.start_flag)
        for flag in ('change_laser_wavelegnth', 'change_laser_power', 'change_laser_rate',
                     'change_laser_divition_factor'):
            if not getattr(self, flag):
                continue
            setattr(self, flag, False)
            changed = True
            try:
                if (code not in (9, 33, 65, 129) or getattr(self, '_laser_standby_pending', False)
                        or getattr(self, '_laser_emission_pending', False)):
                    raise ValueError('Wait for a stable laser state before changing settings.')
                if getattr(self.variables, 'laser_alignment_status', {}).get('active'):
                    raise ValueError('Stop laser alignment before changing laser settings.')
                if flag in ('change_laser_wavelegnth', 'change_laser_rate') and (running or code not in (9, 33)):
                    raise ValueError('Change wavelength/base frequency only in Listen or Standby, outside an experiment.')
                if flag == 'change_laser_wavelegnth':
                    name = self.laser_wavelegnth.currentText()
                    index = next(i for i, (label, nm) in readback.WAVELENGTHS.items() if label == name)
                    self.laser_device.wavelength_change(index)
                elif flag == 'change_laser_rate':
                    index = self.laser_rate.currentData()
                    if index is None:
                        raise ValueError('Read the available frequency table from the laser first.')
                    self.laser_device.Freq(index)
                elif flag == 'change_laser_divition_factor':
                    if running:
                        raise ValueError('Stop the experiment before changing the laser divider.')
                    self.laser_device.Div(self.laser_divition_factor.value())
                else:
                    if self.laser_power.value() > float(self.conf.get('laser_aom_max_percent', 100.)):
                        raise ValueError('Requested AOM setting exceeds laser_aom_max_percent in config.toml.')
                    if readback.scalar(self.laser_device.ModeRead()) != 2:
                        raise ValueError('IR AOM adjustment requires internal power control mode (e_mode=2).')
                    self.laser_device.AOM(round(self.laser_power.value()*40))
            except Exception as exc:
                self.error_message(f'Laser setting not applied: {exc}')
        return changed
