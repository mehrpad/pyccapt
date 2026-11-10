"""Read-only harmonic monitors verified against installed NKT CONTROL.

PubHarmonicLib's SHGExt (type 0x86) reads U16 0x10 / 100 W;
SHGGExt (0x90) and THGiExt (0x93) read U16 0x10 / 1000 W.
THGiExt's U8 0x32 identifies IR/SHG/THG/FHG as 1/2/3/4.
THGExt (0x8D) uses U8 0x30 position and U16 0x10/0x11 / 1000 W.
Only scan-table modules with an unambiguous selected monitor are read.
"""
from pyccapt.control.nkt_photonics import readback


class NativeHarmonicReadoutMixin:
    supports_harmonic_power_readback = True

    def power_read_dv_green(self):
        with self._lock:
            self.harmonic_power_basis = ''
            wavelength = readback.wavelength(self.wavelength_read())
            if wavelength not in (1, 2, 3):
                raise ValueError('Read back a selected harmonic before requesting its power monitor')
            result, types = self.nkt.deviceGetAllTypes(self.comPort)
            if result != 0:
                raise ConnectionError(f'Harmonic module scan unavailable: {self.nkt.DeviceResultTypes(result)}')
            candidates = []
            for address, module_type in enumerate(types):
                if not 0 < address < 256:
                    continue
                if module_type == 0x93:
                    selection = self._read(0x32, 'U8', address=address, allow_unsupported=True)
                    if selection == wavelength + 1:
                        candidates.append((address, module_type, 0x10, 1000.))
                elif module_type == 0x8D:
                    selection = self._read(0x30, 'U8', address=address, allow_unsupported=True)
                    if selection == wavelength:
                        candidates.append((address, module_type, 0x10 if wavelength == 1 else 0x11, 1000.))
                elif wavelength == 1 and module_type in (0x86, 0x90):
                    candidates.append((address, module_type, 0x10, 100. if module_type == 0x86 else 1000.))
            name = readback.WAVELENGTHS[wavelength][0]
            if len(candidates) != 1:
                raise ValueError(f'{name} requires one matching native harmonic monitor; found {len(candidates)}')
            address, module_type, register, divisor = candidates[0]
            value = self._read(register, 'U16', address=address)
            self.harmonic_power_basis = (f'NKTPBus {name} internal monitor: module 0x{address:02X}, '
                f'type 0x{module_type:02X}, register 0x{register:02X}, {1/divisor:g} W/count; '
                'pulse energy calculated from measured average power and output repetition rate')
            tag = {1: 'SHG', 2: 'THG', 3: 'FHG'}[wavelength]
            return f'NKT {tag} Output power={value/divisor:g} W'
