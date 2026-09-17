import copy
import logging
import types

logger = logging.getLogger(__name__)

def snapshot_component(component):
    """
    A detached copy of a representation's public state.

    Used for previous-cycle (DELAYED) reads and for the read-only tamper check
    in DEBUG mode. Values are deep-copied so that later writes to the live
    representation cannot be seen through the snapshot; anything that refuses
    to be copied (a thread, a socket, an open file) is carried over by
    reference, which is the best that can be done and is still stable for
    comparison.
    """
    snap = types.SimpleNamespace()
    for name, value in vars(component).items():
        if name.startswith("__") or isinstance(value, types.ModuleType) or callable(value):
            continue
        try:
            setattr(snap, name, copy.deepcopy(value))
        except Exception:
            logger.debug("Could not copy %s.%s for a snapshot; keeping the reference.",
                getattr(component, "__name__", component), name)
            setattr(snap, name, value)
    return snap

def diff_snapshots(before, after):
    """ Names whose value changed between two snapshots of one representation. """
    changed = []
    for name, old in vars(before).items():
        new = getattr(after, name, _MISSING)
        if new is _MISSING:
            changed.append(name)
            continue
        try:
            if old != new:
                changed.append(name)
        except Exception:
            # values that refuse to compare (arrays, custom __eq__) fall back
            # to identity, which still catches a rebind
            if old is not new:
                changed.append(name)
    for name in vars(after):
        if not hasattr(before, name):
            changed.append(name)
    return sorted(set(changed))

class _Missing:
    pass

_MISSING = _Missing()
