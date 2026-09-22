import logging

from bbmuse.engine.module_handler import ModuleHandler
from bbmuse.engine.representation_handler import RepresentationHandler
from bbmuse.engine.transport import Transport, _TransportView

logger = logging.getLogger(__name__)

class Blackboard:

    def __init__(self, representation_handlers=None, transport: Transport = None):
        self._board = {}
        self._transport = transport or Transport()
        for rep in representation_handlers or []:
            self.register(rep)

        logger.info("Blackboard initialized: %s", self.list_content())

    def register(self, rep_handler: RepresentationHandler):
        rep_name = rep_handler.get_name()
        if rep_name.lower() in [name.lower() for name in self._board.keys()]:
            raise ValueError(f"Duplicate representation name (case ignored): {rep_name}")

        # add actual representation object (in readonly view) to blackboard
        self._board[rep_name] = rep_handler

    #def remove(self, representation):
    #    """ remove representation from board either by name or object """
    #    for key, value in self._board.items():
    #        if key == representation or value == representation:
    #            del self._board[key]
    #            break

    def list_content(self):
        return list(self._board.keys())

    def get(self, name):
        """ returns a representation handler by name """
        return self._board[name]

    def get_transport(self):
        return self._transport

    def create_trigger_view(self, module_handler: ModuleHandler):
        """
        A read-only view for a module's _trigger().

        A trigger is a predicate, so it sees everything the module declares --
        including what it provides, so it can ask whether its own output is
        stale -- but may not write any of it.
        """
        readable = list(dict.fromkeys(module_handler.get_requires()
            + module_handler.get_uses() + module_handler.get_provides()))
        return _BlackboardView(self, readable_keys=readable, writable_keys=[],
            delayed_keys=module_handler.get_delayed(),
            transport=self._transport, owner_name=module_handler.get_name())

    def create_view(self, module_handler: ModuleHandler):
        readable_keys = module_handler.get_requires() + module_handler.get_uses()
        writable_keys = module_handler.get_provides()
        return _BlackboardView(self, readable_keys, writable_keys,
            delayed_keys=module_handler.get_delayed(),
            transport=self._transport, owner_name=module_handler.get_name())

    def create_observer_view(self):
        """
        A read-only view of the whole blackboard.

        For code that judges the system rather than taking part in it -- a
        bblearn reward function, for instance. A reward is a critic, so
        restricting it to the representations the module under test happens to
        declare is backwards: the things worth judging are usually elsewhere.
        """
        return _BlackboardView(self, readable_keys=self.list_content(), writable_keys=[],
            transport=self._transport, owner_name="observer")

    def data_locks_for(self, module_handler: ModuleHandler):
        """
        The data locks of every representation a module touches.

        Sorted by representation name so that all groups acquire overlapping
        locks in the same order and therefore cannot deadlock each other.
        Modules whose representations do not overlap share no lock at all and
        so genuinely run in parallel.
        """
        names = set(module_handler.get_requires()) \
            | set(module_handler.get_uses()) \
            | set(module_handler.get_provides())
        # DELAYED reads go through a snapshot taken before the cycle, so they
        # deliberately do not take the live lock here
        return [self._board[name].get_data_lock() for name in sorted(names)]

class _BlackboardView:

    def __init__(self, blackboard: Blackboard, readable_keys=None, writable_keys=None,
            delayed_keys=None, transport: Transport = None, owner_name=""):
        rep_views = {}
        for readable_key in readable_keys:
            rep_views[readable_key] = blackboard.get(readable_key).create_view(read_only=True)
        for writable_key in writable_keys:
            rep_views[writable_key] = blackboard.get(writable_key).create_view(read_only=False)
        object.__setattr__(self, "_rep_views", rep_views)
        object.__setattr__(self, "_prev", _DelayedView(list(delayed_keys or [])))
        object.__setattr__(self, "_transport_view",
            _TransportView(transport or Transport(), owner_name))

    @property
    def prev(self):
        """ The declared DELAYED representations as they were last cycle. """
        return self._prev

    @property
    def transport(self):
        """ The project's musical clock, plus this module's own event queue. """
        return self._transport_view

    def _set_delayed_snapshots(self, snapshots):
        self._prev._replace(snapshots)

    def __getattr__(self, name):
        try:
            return self._rep_views[name]
        except KeyError:
            raise AttributeError(
                f"Module does not declare representation '{name}'. Add it to REQUIRES, "
                f"USES, DELAYED or PROVIDES.") from None

    def __setattr__(self, name, value):
        raise AttributeError(f"Setting attribute '{name}' on a BlackboardView is not allowed.")

    def __delattr__(self, name):
        raise AttributeError(f"Deleting attribute '{name}' from a BlackboardView is not allowed.")

class _DelayedView:
    """
    Read-only access to the declared representations as of the previous cycle.

    Reached as `bb.prev.<Name>`. The snapshot is taken by the control group
    before the cycle begins, so the value is the same for every module in the
    group and cannot change underneath a module while it runs. This is the
    well-defined way to close a feedback loop: the dependency creates no
    ordering edge, because it does not need one.
    """

    def __init__(self, delayed_keys):
        object.__setattr__(self, "_delayed_keys", delayed_keys)
        object.__setattr__(self, "_snapshots", {})

    def _replace(self, snapshots):
        object.__setattr__(self, "_snapshots", snapshots)

    def __getattr__(self, name):
        snapshots = object.__getattribute__(self, "_snapshots")
        if name in snapshots:
            return snapshots[name]
        keys = object.__getattribute__(self, "_delayed_keys")
        if name in keys:
            raise AttributeError(
                f"No previous-cycle value for '{name}' yet: the first cycle has no "
                f"predecessor. Guard the first read, or give the representation a "
                f"sensible initial value.")
        raise AttributeError(
            f"Module does not declare '{name}' in DELAYED, so there is no "
            f"previous-cycle value for it.")

    def __setattr__(self, name, value):
        raise AttributeError(
            f"'{name}' is a previous-cycle value and cannot be written.")

    def __delattr__(self, name):
        raise AttributeError(
            f"'{name}' is a previous-cycle value and cannot be deleted.")

