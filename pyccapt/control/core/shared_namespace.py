"""Manager Namespace with an explicit, bounded scalar snapshot RPC."""
from multiprocessing.managers import Namespace, NamespaceProxy, SyncManager


class ControlNamespace(Namespace):
    def read_fields(self, fields):
        return {field: getattr(self, field) for field in fields}


class ControlNamespaceProxy(NamespaceProxy):
    _exposed_ = NamespaceProxy._exposed_ + ('read_fields',)

    def read_fields(self, fields):
        return self._callmethod('read_fields', (tuple(fields),))


class ControlManager(SyncManager):
    pass


ControlManager.register('Namespace', ControlNamespace, ControlNamespaceProxy)
