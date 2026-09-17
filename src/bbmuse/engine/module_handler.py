import logging

from pathlib import Path
import importlib.util
import inspect

from time import perf_counter

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class ModuleHandler(BaseHandler):

    # set by the project when a trained model has been applied to this module;
    # a callable taking the handler and returning something with install()
    _implementation_override = None

    def set_implementation_override(self, factory):
        """
        Run something other than the module's own _update().

        The module file is still imported and still supplies the whole
        contract; only the lifecycle hooks are replaced. Used by 'bblearn
        apply' so that swapping in a trained model never edits source.
        """
        self._implementation_override = factory

    def build(self):
        logger.debug("Building %s..", self)
        
        module = self.dynamic_import_from_file(self.get_file_location())
        self.set_component(module)
    
        # attributes
        self._is_running = False
        self.timing_stats = None
        self.trigger_stats = None

        # check for required attributes
        for attr_name, expected_type in (("PROVIDES", list), ("REQUIRES", list),
                ("USES", list), ("DELAYED", list), ("GROUP", str), ("RATE", (int, float)),
                ("PRIORITY", int)):
            if not hasattr(module, attr_name):
                continue # optional: the getters below supply a default
            value = getattr(module, attr_name)
            if not isinstance(value, expected_type):
                expected_name = expected_type.__name__ if isinstance(expected_type, type) \
                    else " or ".join(t.__name__ for t in expected_type)
                raise TypeError(
                    f"{attr_name} in {self} must be a {expected_name}, "
                    f"got {type(value).__name__}.")
        if isinstance(getattr(module, "RATE", None), bool) or \
                (getattr(module, "RATE", None) is not None and module.RATE <= 0):
            raise ValueError(f"RATE in {self} must be a positive number of updates per second.")

        # check for required methods
        update_method = getattr(module, "_update", None)
        if update_method is None:
            raise SyntaxError(f"Module {self} has no _update() method.")
        if not callable(update_method):
            raise SyntaxError(f"_update() in {self} is not callable.")

        # overwrite default print
        def print_with_name_tag(*args, **kwargs):
            # print only if global log level is INFO or less
            if logger.getEffectiveLevel() <= logging.INFO:
                # tag output with module name and group name
                group = "" if self.get_group() == "default" else f" (group:{self.get_group()})"
                print(f"MODULE {self.get_name()}{group}:", *args, **kwargs)
        module.print = print_with_name_tag

        if self._implementation_override is not None:
            self._implementation_override(self).install(module)
            logger.info("Module %s runs an applied model instead of its own _update().", self)

    def hot_reload(self):
        logger.debug("Hot-reloading %s..", self)
        old_component = self.get_component()
        try:
            self.build()
            # if this is a hot-reload, close old module and init new one
            if old_component is not None:
                if callable(getattr(old_component, "_close", None)):
                    old_component._close()
            self.call_init()    
        except Exception:
            logger.exception("Error when building module %s. Keeping former instance.", self)
            self._component = old_component
        else:
            logger.info("Hot-reload on %s was successful.", self)
    
    #def __str__(self):
    #    return f"<Module:{self.get_name()}>"

    """ Mandatory attributes """
    def call_update(self, bb):   
        logger.debug("Running call_update() on module %s", self)
        # TODO: Hot-reload only in mode DEVELOP and DEBUG, not in PERFORM!
        self.consider_hot_reload()
        
        self._is_running = True
        try:
            start_time = perf_counter()
            self.get_component()._update(bb)

            delta_secs = perf_counter() - start_time
            self._update_timing_stats(delta_secs)
        finally:
            self._is_running = False

    def is_running(self):
        return self._is_running

    """ Optional attributes"""
    def get_provides(self):
        return getattr(self.get_component(), "PROVIDES", [])
    
    def get_requires(self):
        return getattr(self.get_component(), "REQUIRES", [])

    def get_uses(self):
        return getattr(self.get_component(), "USES", [])

    def get_delayed(self):
        """ Representations this module reads as of the previous cycle. """
        return getattr(self.get_component(), "DELAYED", [])

    def get_internal_state_names(self):
        """
        Module-level values that are neither contract declarations nor imports.

        A module is free to keep state between updates, but anything learning
        from it only sees the blackboard, so state held here is invisible to a
        clone.
        """
        from bbmuse.engine.snapshot import data_attributes
        contract = {"PROVIDES", "REQUIRES", "USES", "DELAYED", "GROUP", "RATE", "ACTIVE"}
        return sorted(name for name in data_attributes(self.get_component())
            if name not in contract)

    def get_rate(self):
        """
        Requested updates per second, or None to run as often as the group can.

        Declaring a rate is how a module asks to be called periodically. Doing
        it with time.sleep() inside _update() instead stalls every module that
        shares a representation with it.
        """
        return getattr(self.get_component(), "RATE", None)

    def get_period(self):
        rate = self.get_rate()
        return None if rate is None else 1.0 / rate
        
    def get_group(self):
        return getattr(self.get_component(), "GROUP", "default")
        
    def is_active(self):
        return getattr(self.get_component(), "ACTIVE", True)

    def get_declared_level(self):
        """ An explicit LEVEL on the module, overriding what it provides. """
        return getattr(self.get_component(), "LEVEL", None)

    def get_priority(self):
        """
        Which of several ready modules goes first. Higher runs earlier.

        Only breaks ties: a module is never scheduled before something it
        REQUIRES, whatever its priority.
        """
        return getattr(self.get_component(), "PRIORITY", None)

    def has_trigger(self):
        return callable(getattr(self.get_component(), "_trigger", None))

    def should_update(self, trigger_view):
        """
        Whether this module wants to run this cycle.

        A module with no _trigger() runs every cycle, which is what every
        module did before triggers existed. A module with one is asked, and
        answers from the state of the blackboard -- that is what makes the
        control opportunistic rather than a fixed schedule.
        """
        trigger = getattr(self.get_component(), "_trigger", None)
        if not callable(trigger):
            return True
        return bool(trigger(trigger_view))

    def note_cycle(self, fired):
        """ Count how often this module was offered a cycle and took it. """
        if self.trigger_stats is None:
            self.trigger_stats = {"offered": 0, "fired": 0}
        self.trigger_stats["offered"] += 1
        if fired:
            self.trigger_stats["fired"] += 1

    def call_init(self):
        if callable(getattr(self.get_component(), "_init", None)):
            self.get_component()._init()
        else:
            logger.debug("No _init() function found in module %s.", self.get_name())

    def call_close(self):
        if callable(getattr(self.get_component(), "_close", None)):
            self.get_component()._close()
        else:
            logger.debug("No _close() function found in module %s.", self.get_name())

    def _update_timing_stats(self, delta_secs):
        delta = delta_secs * 1000 # sec -> ms
        period = self.get_period()
        overrun = period is not None and delta_secs > period
        if not self.timing_stats:
            self.timing_stats = {
                "n": 1,
                "mean": delta,
                "min": delta,
                "max": delta,
                "overruns": 1 if overrun else 0,
            }
        else:
            stats = self.timing_stats
            stats["n"] += 1
            if delta < stats["min"]:
                stats["min"] = delta
            if delta > stats["max"]:
                stats["max"] = delta
            stats["mean"] += (delta - stats["mean"]) / stats["n"]
            if overrun:
                stats["overruns"] += 1

    def print_timing_stats(self):
        if self.timing_stats:
            rate = self.get_rate()
            budget = ""
            if rate is not None:
                overruns = self.timing_stats.get("overruns", 0)
                budget = (f" rate={rate}Hz budget={1000.0 / rate:.3f}ms"
                    f" overruns={overruns}/{self.timing_stats['n']}")
            fired = ""
            if self.trigger_stats and self.has_trigger():
                fired = (f" triggered={self.trigger_stats['fired']}"
                    f"/{self.trigger_stats['offered']} cycles")
            logger.info("Timing statistics for %s: mean=%sms min=%sms max=%sms%s%s",
                self.get_name(),
                round(self.timing_stats["mean"], 3),
                round(self.timing_stats["min"], 3),
                round(self.timing_stats["max"], 3),
                budget,
                fired,
            )
        else:
            logger.info("No timing statistics available for %s.", self.get_name())
