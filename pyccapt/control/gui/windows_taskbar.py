"""Windows Shell identity for our own Qt windows, including hosted launchers."""
import ctypes
import logging
from uuid import UUID


class _GUID(ctypes.Structure):
    _fields_ = [('data', ctypes.c_ubyte * 16)]


class _PropertyKey(ctypes.Structure):
    _fields_ = [('fmtid', _GUID), ('pid', ctypes.c_ulong)]


class _VariantValue(ctypes.Union):
    _fields_ = [('text', ctypes.c_wchar_p),
                ('padding', ctypes.c_ubyte * (16 if ctypes.sizeof(ctypes.c_void_p) == 8 else 8))]


class _PropVariant(ctypes.Structure):
    _fields_ = [('vt', ctypes.c_ushort), ('reserved', ctypes.c_ushort * 3),
                ('value', _VariantValue)]


def _guid(value):
    return _GUID((ctypes.c_ubyte * 16).from_buffer_copy(UUID(value).bytes_le))


def set_window_identity(hwnd, app_id, icon_path=None):
    """Tell Explorer which identity and icon belong to this HWND.

    Process identity alone can leave a hosted Python window using the host's
    taskbar icon. Shell properties apply specifically to the PyCCAPT window.
    No shortcut, registry or desktop settings are changed.
    """
    store = ctypes.c_void_p()
    release = None
    try:
        iid = _guid('886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99')  # IPropertyStore
        fmtid = _guid('9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3')
        get_store = ctypes.windll.shell32.SHGetPropertyStoreForWindow
        get_store.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
        get_store.restype = ctypes.c_long
        if get_store(hwnd, ctypes.byref(iid), ctypes.byref(store)) != 0:
            raise OSError('SHGetPropertyStoreForWindow failed')
        vtable = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
        set_value = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
            ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant))(vtable[6])
        # Set the icon before the ID so Explorer sees the correct resource
        # immediately. These are PKEY_AppUserModel_RelaunchIconResource and ID.
        properties = []
        if app_id is None:
            properties.append((3, None))
        elif icon_path is not None and icon_path.is_file():
            properties.append((3, f'{icon_path.resolve()},0'))
        properties.append((5, app_id))
        for pid, value in properties:
            key = _PropertyKey(fmtid, pid)
            variant = _PropVariant()
            if value is not None:
                variant.vt = 31  # VT_LPWSTR; SetValue copies our Python-owned string.
                variant.value.text = value
            # VT_EMPTY removes properties before native window destruction.
            if set_value(store, ctypes.byref(key), ctypes.byref(variant)) != 0:
                raise OSError(f'Could not set window taskbar property {pid}')
        return True
    except Exception:
        logging.getLogger('pyccapt.gui').warning('Could not set Windows taskbar icon properties', exc_info=True)
        return False
    finally:
        if release is not None:
            release(store)
