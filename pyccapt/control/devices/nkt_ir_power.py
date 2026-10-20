"""Use the OXPS firmware's watt-to-AOM conversion (manual 800-621-01 p165).

The CLI has no documented watt setter. Register 0x20 converts a watt setpoint
to the calibrated relative AOM value in 0x05. Protocol changes reset to Listen,
so return that value for the caller to reapply only after verifying CLI timing
and wavelength. No emission, interlock or shutter-enable command is sent here.
"""
import math
import sys
import time

from pyccapt.control.nkt_photonics import nktpbus_switch


def calibrated_aom(port, watts, *, max_watts, max_aom_percent, base_hz, divider,
                   wait=time.sleep, cancelled=lambda: False):
    """Return (accepted watts, relative AOM) after restoring/releasing CLI."""
    watts, max_watts = float(watts), float(max_watts)
    max_aom_percent = float(max_aom_percent)
    if not (math.isfinite(watts) and math.isfinite(max_watts) and 0 <= watts <= max_watts <= 65.535):
        raise ValueError('IR power must be finite and within laser_ir_max_power_w (at most 65.535 W).')
    if not math.isfinite(max_aom_percent) or not 0 <= max_aom_percent <= 100:
        raise ValueError('laser_aom_max_percent must be within 0..100.')
    milliwatts = round(watts * 1000)
    nkt = nktpbus_switch._load_nktp_dll_module()  # Preflight before changing protocol.
    if cancelled():
        raise nktpbus_switch.NKTPSwitchError('IR power request cancelled.')
    nktpbus_switch.switch_to_nktpbus(port)
    address = None

    def read(register, kind='U16'):
        result, value = getattr(nkt, 'registerRead'+kind)(port, address, register, -1)
        if result != 0:
            raise nktpbus_switch.NKTPSwitchError(
                f'Could not read OXPS register 0x{register:02X}: {nkt.RegisterResultTypes(result)}'
            )
        return value

    try:
        wait(1.5)  # CLI -> NKTPBus resets the mainboard.
        if cancelled():
            raise nktpbus_switch.NKTPSwitchError('IR power request cancelled; use Switch to CLI to reconnect.')
        opened = nkt.openPorts(port, 1, 0)
        if opened != 0:
            raise nktpbus_switch.NKTPSwitchError(f'Could not open NKTPBus on {port}: {nkt.PortResultTypes(opened)}')
        address = nktpbus_switch._find_oxps_address(nkt, port, None)
        if read(0x37, 'U8') != 0:
            raise ValueError('IR watt adjustment requires internal power control mode.')
        if read(0x02, 'U32') != base_hz or read(0x36, 'U32') != divider:
            raise ValueError('Laser timing changed during the protocol switch; IR power was not applied.')
        if cancelled():
            raise nktpbus_switch.NKTPSwitchError('IR power request cancelled.')
        written = nkt.registerWriteU16(port, address, 0x20, milliwatts, -1)
        if written != 0:
            raise nktpbus_switch.NKTPSwitchError(f'IR watt setpoint failed: {nkt.RegisterResultTypes(written)}')
        accepted = read(0x20)
        aom = read(0x05)
        if accepted != milliwatts:
            raise ValueError(f'Laser did not accept {milliwatts / 1000:g} W (read back {accepted / 1000:g} W).')
        if not 0 <= aom <= min(4000, round(max_aom_percent * 40)):
            raise ValueError('Calibrated IR watt setting exceeds laser_aom_max_percent.')
        return accepted / 1000, aom
    finally:
        had_error = sys.exc_info()[0] is not None
        try:
            if address is not None:
                restored = nkt.registerWriteU8(port, address, 0x39, 1, -1)
                if restored != 0:
                    raise nktpbus_switch.NKTPSwitchError('Could not restore CLI; click Switch to CLI to recover.')
        finally:
            closed = nkt.closePorts(port)
            if closed != 0 and not had_error:
                raise nktpbus_switch.NKTPSwitchError(f'Could not release {port} after IR watt conversion.')
