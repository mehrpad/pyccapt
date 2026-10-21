"""Readback-confirmed laser LEDs; animation performs no hardware I/O."""
from PyQt6 import QtCore

from pyccapt.control.nkt_photonics import readback


class LaserIndicatorMixin:
    def _setup_laser_indicators(self, parent):
        self._laser_indicator_requests = {}
        self._laser_led_phase = False
        self._laser_led_timer = QtCore.QTimer(parent)
        self._laser_led_timer.setInterval(500)
        self._laser_led_timer.timeout.connect(self._blink_laser_indicators)

    def _clear_laser_indicator_requests(self):
        self._laser_indicator_requests = {}
        self._render_laser_indicators()

    def _queue_laser_indicator(self, name, codes):
        if not hasattr(self, '_laser_indicator_requests'):
            self._laser_indicator_requests = {}
        self._laser_indicator_requests[name] = (tuple(codes), False)
        self._render_laser_indicators()

    def _mark_laser_indicators_sent(self):
        self._laser_indicator_requests = {
            name: (codes, True) for name, (codes, _) in
            getattr(self, '_laser_indicator_requests', {}).items()}

    def _observe_laser_indicators(self, code):
        if code in (9, 33, 65, 129):
            self._laser_last_stable_code = code
        if code not in (1, 9, 17, 33, 65, 129):
            # Warnings can persist while the controller is stopping. Keep
            # the explicit Listen request blinking until its FSM confirms it.
            self._laser_indicator_requests = {
                name: request for name, request in
                getattr(self, '_laser_indicator_requests', {}).items()
                if name == 'listen' and code in (3, 5)}
            self._laser_standby_pending = self._laser_emission_pending = False
        else:
            self._laser_indicator_requests = {
                name: (codes, sent) for name, (codes, sent) in
                getattr(self, '_laser_indicator_requests', {}).items()
                if not (sent and code in codes)}

    def _blink_laser_indicators(self):
        self._laser_led_phase = not getattr(self, '_laser_led_phase', False)
        self._render_laser_indicators()

    def _render_laser_indicators(self):
        connected = getattr(self, 'laser_device', None) is not None
        code = readback.scalar(getattr(self, '_last_laser_status', None)) if connected else None
        waiting = set(getattr(self, '_laser_indicator_requests', {}))
        if code == 17 and not waiting:
            waiting = {'on', 'output'} if getattr(self, '_laser_emission_pending', False) else {'standby'}
        if not connected or getattr(self, '_laser_stopping', False):
            waiting.clear()
        # Preserve the last confirmed stopped state while the requested
        # Standby transition blinks; the state caption reports live warm-up.
        previous_listen = ('standby' in waiting and
                           getattr(self, '_laser_last_stable_code', None) == 9)
        listen_led = getattr(self, 'led_laser_listen', None)
        if listen_led is not None:
            listen_led.setToolTip('Last confirmed Listen; preparing Standby.' if previous_listen else 'Listen state readback.')
        for name, widget_name, active in (
            ('listen', 'led_laser_listen', code == 9 or previous_listen),
            ('standby', 'led_laser_laser_standby', code == 33),
            ('on', 'led_laser_on', code in (65, 129)),
            ('output', 'led_laser_enable', code == 129),
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                if name in waiting:
                    pixmap = (getattr(self, 'led_orange', self.led_red)
                              if getattr(self, '_laser_led_phase', False) else self.led_red)
                else:
                    pixmap = self.led_green if active else self.led_red
                widget.setPixmap(pixmap)
        timer = getattr(self, '_laser_led_timer', None)
        if timer is not None:
            if waiting and not timer.isActive():
                timer.start()
            elif not waiting:
                timer.stop()
