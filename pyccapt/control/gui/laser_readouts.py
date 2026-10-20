"""Verified Origami settings and unit-aware readouts for the laser GUI."""
import datetime
import logging
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
        for widget in (self.laser_ir_power_disp, self.laser_power_disp,
                       self.laser_pulse_energy_disp, self.laser_repetion_rate_disp):
            widget.display('-----')
        self._update_wavelength_nm_label()
        self._report_optical_readback_error(str(reason))
        self._recompute_derived_readouts()

    def _report_optical_readback_error(self, reason):
        """Log a changed diagnostic once, without repeating it every poll."""
        previous = getattr(self, '_last_optical_readback_error', '')
        self._last_optical_readback_error = reason
        if reason and reason != previous:
            logging.getLogger('pyccapt.laser').warning('Laser optical readback unavailable: %s', reason)
        elif previous and not reason:
            logging.getLogger('pyccapt.laser').info('Laser optical readback restored')

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
                    self.laser_rate.addItem(f'{hz/1000:g}', index)
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
            try:
                raw['e_mlp'] = device.read_average_power()
            except Exception as exc:
                raw['e_mlp'] = f'Read failed: {exc}'
            ir_mw, ir_nj = readback.optical_values(raw['e_mlp'], output_hz)
            ir_quantity = readback.quantity(raw['e_mlp'])
            ir_basis = 'unavailable'
            if ir_quantity:
                ir_basis = 'measured' if ir_quantity[1].endswith('W') else 'estimated'
            ir_error = '' if math.isfinite(ir_mw) else (
                f'IR monitor has no usable unit-bearing reading; e_mlp? reply: {raw["e_mlp"][:256]!r}'
            )
            optical_error = ''
            if wl == 0:
                mw, nj = ir_mw, ir_nj
                source = 'e_mlp (IR monitor; explicit reply unit)'
                if not math.isfinite(mw) or not math.isfinite(nj):
                    optical_error = ir_error
            else:
                try:
                    raw['ls_output_power'] = device.power_read_dv_green()
                except Exception as exc:
                    raw['ls_output_power'] = f'Read failed: {exc}'
                measured = readback.quantity(raw['ls_output_power'])
                # Manual defines harmonic monitor in W. Accept explicit power
                # units only; unsupported FHG replies remain unknown.
                if measured is None or not measured[1].endswith('W'):
                    mw = nj = math.nan
                    optical_error = (
                        f"{readback.WAVELENGTHS[wl][0]} monitor returned an unsupported or ambiguous power reading; "
                        f"ls_output_power? reply: {raw['ls_output_power'][:256]!r}"
                    )
                else:
                    mw, nj = readback.optical_values(raw['ls_output_power'], output_hz)
                source = 'ls_output_power (selected harmonic monitor)'
            raw['status'] = device.StatusRead()
            code = readback.scalar(raw['status'])
            if code not in (9, 33, 65, 129):
                optical_error = f'Laser state {code} is not stable; optical readings unavailable. '+optical_error
                ir_error = f'Laser state {code} is not stable; IR reading unavailable.'
                ir_mw = math.nan
            valid = code in (9, 33, 65, 129) and math.isfinite(mw) and math.isfinite(nj)
            snapshot = dict(monotonic=time.monotonic(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                valid=valid, raw=raw, source=source, wavelength=readback.WAVELENGTHS[wl][0],
                wavelength_index=wl, wavelength_nm=readback.WAVELENGTHS[wl][1],
                wavelength_basis='nominal selected harmonic, not a live spectral measurement',
                base_frequency_hz=hz, frequency_index=index, divider=int(div), output_frequency_hz=output_hz,
                aom_percent=aom/40., ir_power_mw=ir_mw, ir_power_basis=ir_basis,
                ir_power_error=ir_error, output_power_mw=mw if valid else math.nan,
                pulse_energy_nj=nj if valid else math.nan, status_code=code,
                location='laser internal monitor; not calibrated energy delivered to specimen',
                frequency_table=dict(self._frequency_table))
            if optical_error:
                snapshot['error'] = optical_error
            self.variables.laser_telemetry = snapshot
            self._report_optical_readback_error(optical_error)
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
            (self.laser_ir_power_disp, 'ir_power_mw', .001),
            (self.laser_power_disp, 'output_power_mw', .001),
            (self.laser_pulse_energy_disp, 'pulse_energy_nj', .001),
            (self.laser_repetion_rate_disp, 'output_frequency_hz', .001),
        ):
            value = data.get(key, math.nan)
            widget.display(value*scale if math.isfinite(value) else '-----')
        tooltip = ('Source: '+data.get('source', 'unavailable')+
                   '. Internal monitor estimate; not energy at the specimen. Missing readings show dashes.')
        if data.get('error'):
            tooltip += '\n'+data['error']+'\nPulse energy is unavailable without a valid selected-output power/energy reading.'
        if not data:
            tooltip += '\nNo fresh optical readback is available.'
        self.laser_power_disp.setToolTip(tooltip)
        self.laser_pulse_energy_disp.setToolTip(tooltip)
        estimated = data.get('ir_power_basis') == 'estimated'
        self.laser_ir_power_label.setText('IR power estimate (W)' if estimated else 'Measured IR power (W)')
        ir_tooltip = ('IR internal monitor (e_mlp?); independent of the selected harmonic. '
                      'This is not the power delivered to the specimen.')
        if estimated:
            ir_tooltip += '\nCalculated from the returned IR pulse energy and output repetition rate; not a direct watt measurement.'
        if data.get('ir_power_error'):
            ir_tooltip += '\n'+data['ir_power_error']
        elif not math.isfinite(data.get('ir_power_mw', math.nan)) and data.get('error'):
            ir_tooltip += '\n'+data['error']
        if not data:
            ir_tooltip += '\nNo fresh IR monitor readback is available.'
        elif data.get('raw', {}).get('e_mlp'):
            ir_tooltip += '\nRaw IR monitor reply: '+data['raw']['e_mlp'][:256]
        self.laser_ir_power_disp.setToolTip(ir_tooltip)
        self.laser_ir_power_label.setToolTip(ir_tooltip)

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
