import copy
import logging
import types

logger = logging.getLogger(__name__)

def snapshot_component(component):
    """
    A detached copy of a representation, usable exactly like the original.

    Data attributes are deep-copied, so later writes to the live
    representation are not visible through the snapshot. Functions are
    rebound to the snapshot's own namespace, so calling `_pack()` or any other
    helper on a snapshot reads the snapshot's values rather than the live ones.
    Imported modules and dunders are carried over by reference.

    Anything that refuses to be copied (a thread, a socket, an open file) is
    carried over by reference too. That is the best that can be done, and it
    is still stable enough to compare against.
    """
    snapshot = types.ModuleType(getattr(component, "__name__", "snapshot"))
    namespace = snapshot.__dict__

    functions = {}
    for name, value in vars(component).items():
        if isinstance(value, types.FunctionType):
            functions[name] = value
        elif name.startswith("__") or isinstance(value, types.ModuleType) or callable(value):
            namespace[name] = value
        else:
            try:
                namespace[name] = copy.deepcopy(value)
            except Exception:
                logger.debug("Could not copy %s.%s for a snapshot; keeping the reference.",
                    snapshot.__name__, name)
                namespace[name] = value

    # rebound only once the data is in place, so the new globals are complete
    for name, function in functions.items():
        namespace[name] = types.FunctionType(function.__code__, namespace,
            function.__name__, function.__defaults__, function.__closure__)

    return snapshot

def data_attributes(component):
    """ The public, non-callable, non-imported names of a representation. """
    return {name: value for name, value in vars(component).items()
        if not name.startswith("__")
        and not isinstance(value, types.ModuleType)
        and not callable(value)}

def diff_snapshots(before, after):
    """ Data attributes whose value changed between two snapshots. """
    old_values = data_attributes(before)
    new_values = data_attributes(after)

    changed = set(old_values) ^ set(new_values)
    for name in set(old_values) & set(new_values):
        old, new = old_values[name], new_values[name]
        try:
            if old != new:
                changed.add(name)
        except Exception:
            # values that refuse to compare cleanly (arrays, custom __eq__)
            # fall back to identity, which still catches a rebind
            if old is not new:
                changed.add(name)
    return sorted(changed)
