import logging

from collections import defaultdict, deque
from time import time, sleep

import gc

from bbmuse.engine.blackboard import Blackboard
from bbmuse.engine.control_group import ControlGroup
from bbmuse.engine.snapshot import snapshot_component

logger = logging.getLogger(__name__)

class Controller:

    def __init__(self, module_handlers, blackboard: Blackboard, failed_representation_names=()):
        self.module_handlers = module_handlers
        self.blackboard = blackboard
        self.failed_representation_names = list(failed_representation_names)

        self.groups = self.make_groups()
        
    def make_groups(self):
        grouped_handlers = defaultdict(list)
        for handler in self.module_handlers:
            grouped_handlers[handler.get_group()].append(handler)

        groups = []
        for handlers in grouped_handlers.values():
            groups.append(ControlGroup(handlers, self.blackboard))
        return groups

    def build(self, strict=False):
         # test if dependency graph is is complete
        self.build_execution_order()
        self.report_deprecated_uses()
        self.report_cross_group_requires(strict=strict)

    def report_deprecated_uses(self):
        """
        USES is superseded by DELAYED. Both read without creating an ordering
        edge, but USES reads the live representation -- so what it returns
        depends on the arbitrary tie-breaking of the topological sort and on
        group membership -- while DELAYED reads a snapshot of the previous
        cycle, which is the same for everyone and reproducible.
        """
        for handler in self.module_handlers:
            if handler.get_uses():
                logger.warning(
                    "%s declares USES %s. USES is deprecated because what it reads is "
                    "not well defined: move these to DELAYED to read the previous cycle "
                    "reproducibly, or to REQUIRES to be ordered after the provider.",
                    handler, ", ".join(handler.get_uses()))

        for group in self.groups:
            group.build(self.execution_order)

    def report_cross_group_requires(self, strict=False):
        """
        A REQUIRES edge only orders two modules when they run in the same
        control group. Groups are separate threads, so an edge that crosses a
        group boundary gives no ordering and no one-to-one pairing at all: the
        consumer sees whichever value happens to be there, and may see the same
        one many times or miss most of them. Nothing about the declaration says
        so, hence this report.
        """
        crossing = []
        for provider, consumers in self.dependencies.items():
            for consumer in consumers:
                if provider.get_group() != consumer.get_group():
                    shared = sorted(set(provider.get_provides()) & set(consumer.get_requires()))
                    crossing.append((provider, consumer, shared))

        if not crossing:
            return

        for provider, consumer, shared in crossing:
            logger.warning(
                "%s requires %s from %s, but they are in different control groups "
                "('%s' and '%s'). The dependency is NOT ordered across groups: put both "
                "modules in one group for lockstep updates, or declare it in DELAYED to "
                "read the previous cycle deterministically.",
                consumer, ", ".join(shared), provider,
                consumer.get_group(), provider.get_group())

        if strict:
            raise RuntimeError(
                f"{len(crossing)} REQUIRES dependencies cross a control group boundary "
                "and are therefore unordered. Listed above; refused in DEBUG mode.")

    def build_execution_order(self):
        # construct mapping: repr -> provider
        provides_map = {}
        for handler in self.module_handlers:
            for repr in handler.get_provides():
                if not repr in self.blackboard._board.keys():
                    if repr in self.failed_representation_names:
                        raise RuntimeError(f"Representation {repr}, provided by module {handler}, failed to build. See the logged traceback above for the cause.")
                    raise RuntimeError(f"Representation {repr} is unknown to the blackboard, thus cannot be provided by module {handler}. No definition file for it was found.")
                if not repr in provides_map.keys():
                    provides_map[repr] = handler
                else:
                    raise RuntimeError(f"Duplicate provide: Representation {repr} provided by modules {handler} and {provides_map[repr]}.")
        logger.debug("Map repr -> provider: %s", provides_map)

        # DELAYED names must exist; unlike REQUIRES they add no ordering edge,
        # which is the whole point of declaring them that way
        for handler in self.module_handlers:
            for repr in handler.get_delayed():
                if not repr in self.blackboard._board.keys():
                    raise RuntimeError(
                        f"Module {handler} declares {repr} in DELAYED, but no such "
                        f"representation is on the blackboard.")

        # Build the graph: edges from providers -> consumers
        graph = defaultdict(list)
        num_of_consumers = {m: 0 for m in self.module_handlers}

        for handler in self.module_handlers:
            for req in handler.get_requires():
                provider = provides_map.get(req, None)
                if provider is None:
                    logger.debug("No module provides representation %s which module %s requires. Therefore it is irrelevant to the execution order.", req, handler)
                else:
                    if not handler in graph[provider]:
                        graph[provider].append(handler)
                        num_of_consumers[handler] += 1
        logger.debug("Map provider -> list of consumers: %s", graph)
        logger.debug("Num. of consumers per provider %s:", num_of_consumers)

        # Topological Sort: Kahn's algorithm (doi:10.1145/368996.369025)
        ready = deque([m for m, deg in num_of_consumers.items() if deg == 0])
        exec_order = []

        while ready:
            handler = ready.popleft()
            exec_order.append(handler)
            for neighbor in graph[handler]:
                num_of_consumers[neighbor] -= 1
                if num_of_consumers[neighbor] == 0:
                    ready.append(neighbor)
        logger.debug(f"Proposed execution order: %s", exec_order)

        if len(exec_order) != len(self.module_handlers):
            raise RuntimeError("Cycle detected in module dependencies")

        self.execution_order, self.dependencies = exec_order, graph

    def run(self, quit_after=-1, run_mode=0):

        logger.info("Init threads..")

        logger.info("Call _init() on all modules..")
        for mod_handler in self.module_handlers:
            mod_handler.call_init()

        if run_mode > 0: # PERFORM mode
            gc.disable()
        logger.debug("Garbage collector %s.", "enabled" if gc.isenabled() else "disabled")

        for group in self.groups:
            logger.info("Attempting to start thread '%s'..", group.name)
            group.start(run_mode=run_mode)

        self._running = True
        start_time = time()
        try:
            while self._running:
                sleep(0.5)
                if quit_after >= 0 and time() - start_time > quit_after:
                    self.halt()
                   
                # check hot-reload for all representations
                for rep in self.blackboard.list_content():
                    self.blackboard.get(rep).consider_hot_reload()
                    
                for group in self.groups:
                    if not group.is_alive():
                        if run_mode <= 0:# NORMAL mode
                            logger.error("Group %s stopped running. Not in PERFORM mode, hence halt program..", group.name)
                            self.halt()
                        else: # PERFORM mode
                            logger.warning("Group %s stopped running in PERFORM mode. Restarting..", group.name)
                            group.start(run_mode=run_mode)
        except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt detected: request halt and join..")
                self.halt()
        finally:
            # shut down from here on no matter how the loop was left, so that
            # modules are always closed and the gc is always turned back on
            for group in self.groups:
                group.halt()
            logger.debug(f"Requested halt after %.3f secs..", time() - start_time)

            for group in self.groups:
                group.halt_and_join()
                logger.debug("Group '%s' accepted join with main thread.", group.name)

            logger.info("All groups joined with main thread.")

            logger.info("Call _close() on all modules..")
            for mod_handler in self.module_handlers:
                try:
                    mod_handler.call_close()
                except Exception:
                    logger.exception("Error while closing module %s.", mod_handler)

            # if garbage collector has been disabled
            if run_mode > 0:
                gc.enable()

            for mod_handler in self.module_handlers:
                mod_handler.print_timing_stats()

    def step(self, n_cycles=1, run_mode=0, seed=None, seconds_per_cycle=0.01):
        """
        Run n_cycles of the whole project on this thread, deterministically.

        No control group threads are started: every module is called once per
        cycle in the single global execution order, so the result does not
        depend on thread scheduling and two runs of the same project with the
        same seed produce the same blackboard. RATE declarations are ignored,
        since there is no wall clock to be late against.

        If the blackboard's transport is virtual it is advanced by
        seconds_per_cycle each cycle, so scheduled events fire at reproducible
        cycles rather than at whatever the wall clock happened to say.

        This is what makes a project testable and a bblearn recording
        reproducible. Returns the number of cycles actually run.
        """
        transport = self.blackboard.get_transport()
        virtual = getattr(transport, "_virtual", False)
        if not virtual:
            logger.warning(
                "Stepping with a real-time transport: bb.transport.now still follows the "
                "wall clock, so anything driven by it will not be reproducible. Build the "
                "project with virtual_transport=True for a fully deterministic run.")
        if seed is not None:
            self.seed_random_sources(seed)

        logger.info("Call _init() on all modules..")
        for mod_handler in self.module_handlers:
            mod_handler.call_init()

        delayed_names = sorted({name
            for handler in self.module_handlers
            for name in handler.get_delayed()})
        views = {handler: self.blackboard.create_view(handler)
            for handler in self.module_handlers}

        self._running = True
        cycles_run = 0
        try:
            for _ in range(n_cycles):
                if not self._running:
                    break

                if virtual:
                    transport.advance_virtual(seconds_per_cycle)

                snapshots = {name: snapshot_component(self.blackboard.get(name).get_component())
                    for name in delayed_names}
                for view in views.values():
                    view._set_delayed_snapshots(snapshots)

                for mod_handler in self.execution_order:
                    if not mod_handler.is_active():
                        continue
                    mod_handler.call_update(views[mod_handler])
                    if run_mode < 0: # DEBUG mode
                        for rep_name in mod_handler.get_provides():
                            self.blackboard.get(rep_name).call_validate()
                cycles_run += 1
        finally:
            logger.info("Call _close() on all modules..")
            for mod_handler in self.module_handlers:
                try:
                    mod_handler.call_close()
                except Exception:
                    logger.exception("Error while closing module %s.", mod_handler)

        logger.info("Ran %s deterministic cycles.", cycles_run)
        for mod_handler in self.module_handlers:
            mod_handler.print_timing_stats()
        return cycles_run

    def seed_random_sources(self, seed):
        import random
        random.seed(seed)
        logger.debug("Seeded random with %s", seed)
        try:
            import numpy
            numpy.random.seed(seed)
            logger.debug("Seeded numpy.random with %s", seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            logger.debug("Seeded torch with %s", seed)
        except ImportError:
            pass

    def halt(self):
        self._running = False
