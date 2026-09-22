import logging

from collections import defaultdict, deque
from contextlib import ExitStack
from time import monotonic, sleep, time

import threading

from bbmuse.engine.snapshot import diff_snapshots, snapshot_component

base_logger = logging.getLogger(__name__)

class ControlGroup:

    def __init__(self, module_handlers, blackboard):
        group_names = list(set([handler.get_group() for handler in module_handlers]))
        assert len(group_names) == 1, f"Got modules from {len(group_names)} groups: {group_names}. But all modules must be in the same group."

        self.name = group_names[0]
        self.module_handlers = module_handlers
        self.blackboard = blackboard

        self.thread = None

        self.logger = base_logger
        if self.name != "default":
            self.logger = logging.getLogger(__name__ + f" (group:{self.name})")
        
        self.logger.info("Created group '%s'. Members are: %s", self.name, self.module_handlers)

    def build(self, exec_order):
        self.execution_order = [handler for handler in exec_order if handler in self.module_handlers]
        self.build_blackboard_views()
        # resolved once, so the hot path only acquires ready-made locks
        self.data_locks = {handler: self.blackboard.data_locks_for(handler)
            for handler in self.module_handlers}
        # every representation any member of this group reads with a one-cycle delay
        self.delayed_names = sorted({name
            for handler in self.module_handlers
            for name in handler.get_delayed()})
        # modules that asked to be called at a fixed rate; the rest run every cycle
        self.periods = {handler: handler.get_period() for handler in self.module_handlers}
        self.free_running = all(period is None for period in self.periods.values())

    def build_blackboard_views(self):
        bb_views = {}
        for handler in self.module_handlers:
            bb_views[handler] = self.blackboard.create_view(handler)

        self.blackboard_views = bb_views

    def start(self, run_mode=0):
        self.run_mode = run_mode
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def halt_and_join(self, timeout=None):
        self.halt()
        self.thread.join(timeout=timeout)
    
    def is_alive(self):
        return self.thread.is_alive()

    def run(self):
        cycle_count = 0
        self.logger.info(f"Execution order: %s", self.execution_order)

        self._running = True
        # every rate-limited module is due immediately on the first cycle
        next_due = {handler: monotonic() for handler in self.module_handlers}

        self.logger.info("Start running..")
        while self._running:
            start_time = time()

            self.wait_until_due(next_due)

            # one snapshot per cycle, shared by every module in the group, so a
            # DELAYED read is the same value for all of them and cannot move
            # while the cycle runs
            self.take_delayed_snapshots()

            try:
                for mod_handler in self.execution_order:
                    #self.logger.debug("Call _update() on module %s", mod_handler)

                    period = self.periods[mod_handler]
                    if period is not None:
                        now = monotonic()
                        if now < next_due[mod_handler]:
                            continue # not due yet this cycle
                        # advance on the ideal grid so the rate does not drift,
                        # but never try to catch up a backlog after an overrun
                        next_due[mod_handler] = max(next_due[mod_handler] + period, now)

                    try:
                        if self.run_mode <= 0: # DEBUG or NORMAL mode
                            for rep_name in mod_handler.get_requires():
                                self.blackboard.get(rep_name).consider_hot_reload()
                    
                        if self._running and mod_handler.is_active():
                            with ExitStack() as locks:
                                for data_lock in self.data_locks[mod_handler]:
                                    locks.enter_context(data_lock)
                                before = self.snapshot_read_only(mod_handler)
                                mod_handler.call_update(self.blackboard_views[mod_handler])
                                self.check_read_only_untouched(mod_handler, before)

                        if self.run_mode < 0: # DEBUG mode
                            try:
                                for rep_name in mod_handler.get_provides():
                                    self.blackboard.get(rep_name).call_validate()
                            except Exception:
                                self.logger.exception(f"Representation {rep_name} did not pass validation check.")
                                self.halt()

                    except Exception:
                        # Stop in dev mode (normal), ignore in release mode (perform).
                        self.logger.exception(f"Module {mod_handler} produced an error.")
                        if self.run_mode <= 0: # DEBUG or NORMAL mode
                            self.logger.info("Not in PERFORM mode: trigger halt..")
                            self.halt()
                        else:
                            self.logger.debug("In PERFORM mode: ignoring the error.")

            except KeyboardInterrupt:
                self.logger.exception("KeyboardInterrupt detected: signal halt..")
                self.halt()

            delta_time = time() - start_time
            cycle_count += 1

            #self.logger.debug(f"End of cycle {cycle_count}, delta={delta_time:.5f}")

    def wait_until_due(self, next_due):
        """
        Sleep until the earliest rate-limited module is due.

        Nothing is held while waiting, so a slow or infrequent module no longer
        blocks anything. A group with no declared rates keeps running flat out,
        which is the behaviour projects had before rates existed.
        """
        if self.free_running:
            return
        waits = [next_due[handler] - monotonic()
            for handler, period in self.periods.items() if period is not None]
        unpaced = any(period is None for period in self.periods.values())
        if unpaced or not waits:
            return # something wants to run as fast as it can
        delay = min(waits)
        if delay > 0:
            sleep(delay)

    def snapshot_read_only(self, mod_handler):
        """
        DEBUG mode only: capture the representations this module may read but
        not write, so that writing through them can be detected afterwards.

        A read-only view blocks rebinding an attribute, but reading one hands
        back the object itself, so a module can still mutate a list, a dict or
        an array it only declared in REQUIRES. That cannot be prevented without
        copying every read; it can be caught, which is what this does.
        """
        if self.run_mode >= 0:
            return None
        return {name: snapshot_component(self.blackboard.get(name).get_component())
            for name in set(mod_handler.get_requires()) | set(mod_handler.get_uses())}

    def check_read_only_untouched(self, mod_handler, before):
        if not before:
            return
        for rep_name, old in before.items():
            changed = diff_snapshots(old,
                snapshot_component(self.blackboard.get(rep_name).get_component()))
            if changed:
                self.logger.error(
                    "Module %s wrote to %s, which it only declared as read-only: %s. "
                    "Move %s to PROVIDES, or stop mutating it.",
                    mod_handler, rep_name, ", ".join(changed), rep_name)
                self.halt()

    def take_delayed_snapshots(self):
        if not self.delayed_names:
            return
        snapshots = {}
        for rep_name in self.delayed_names:
            rep_handler = self.blackboard.get(rep_name)
            with rep_handler.get_data_lock():
                snapshots[rep_name] = snapshot_component(rep_handler.get_component())
        for bb_view in self.blackboard_views.values():
            bb_view._set_delayed_snapshots(snapshots)

    def halt(self):
        self._running = False
