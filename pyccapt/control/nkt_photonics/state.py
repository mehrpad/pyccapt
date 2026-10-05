"""Observed Origami states and permitted manual actions (vendor status LEDs)."""
from enum import Enum


class LaserState(str, Enum):
    UNKNOWN = 'Status unavailable'
    POWER = 'Power / booting'
    ERROR = 'Error'
    WARNING = 'Warning'
    LISTEN = 'Listen'
    SETUP = 'Setup / warming'
    STANDBY = 'Standby / ready'
    ON_DISABLED = 'Laser on / output closed'
    ON_ENABLED = 'Laser on / output open'

    @classmethod
    def from_code(cls, code):
        return {
            1: cls.POWER, 3: cls.ERROR, 5: cls.WARNING, 9: cls.LISTEN,
            17: cls.SETUP, 33: cls.STANDBY, 65: cls.ON_DISABLED, 129: cls.ON_ENABLED,
        }.get(code, cls.UNKNOWN)


def allowed_actions(state, *, connected):
    """Allow a Listen recovery request even when status readback has failed.

    Availability is permission to send a command, never confirmation that the
    hardware reached its destination. Upward commands require observed readiness.
    """
    if not connected:
        return frozenset()
    actions = set()
    if state != LaserState.LISTEN:
        actions.add('listen')
    if state in (LaserState.LISTEN, LaserState.ON_DISABLED, LaserState.ON_ENABLED):
        actions.add('standby')
    if state in (LaserState.STANDBY, LaserState.ON_ENABLED):
        actions.add('on')
    if state in (LaserState.ON_DISABLED, LaserState.ON_ENABLED):
        actions.add('output')
    return frozenset(actions)
