import logging

from collections import defaultdict, deque
from time import time, sleep

from bbmuse.engine.blackboard import Blackboard
from bbmuse.engine.control_group import ControlGroup

logger = logging.getLogger(__name__)

class Controller:

    def __init__(self, module_handlers, blackboard: Blackboard):
        self.module_handlers = module_handlers
        self.blackboard = blackboard

        self.groups = self.make_groups()
        
    def make_groups(self):
        grouped_handlers = defaultdict(list)
        for handler in self.module_handlers:
            grouped_handlers[handler.get_group()].append(handler)

        groups = []
        for handlers in grouped_handlers.values():
            groups.append(ControlGroup(handlers, self.blackboard))
        return groups

    def build(self):
         # test if dependency graph is is complete
        self.build_execution_order()

        for group in self.groups:
            group.build(self.execution_order)

    def build_execution_order(self):
        # construct mapping: repr -> provider
        provides_map = {}
        for handler in self.module_handlers:
            for repr in handler.get_provides():
                if not repr in self.blackboard._board.keys():
                    raise RuntimeError(f"Representation {repr} is unknown to the blackboard, thus cannot be provided by module {handler}.")
                if not repr in provides_map.keys():
                    provides_map[repr] = handler
                else:
                    raise RuntimeError(f"Duplicate provide: Representation {repr} provided by modules {handler} and {provides_map[repr]}.")
        logger.debug("Map repr -> provider: %s", provides_map)

        # Build the graph: edges from providers -> consumers
        graph = defaultdict(list)
        num_of_consumers = {m: 0 for m in self.module_handlers}

        for handler in self.module_handlers:
            for req in handler.get_requires():
                provider = provides_map.get(req, None)
                if provider is None:
                    raise RuntimeError(f"No module provides required representation: {req}")
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

    def run(self, quit_after=-1):

        logger.info("Init threads..")

        logger.info("Call _init() on all modules..")
        for mod_handler in self.module_handlers:
            mod_handler.call_init()

        for group in self.groups:
            logger.info("Attempting to start thread '%s'..", group.name)
            group.start()

        self._running = True
        start_time = time()
        try:
            while self._running:
                sleep(0.1)
                if quit_after >= 0 and time() - start_time > quit_after:
                    self.halt()
        except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt detected: request halt and join..")
                self.halt()
        finally:
            for group in self.groups:
                group.halt()
            logger.debug(f"Requested halt after %.3f secs..", time() - start_time)

        for group in self.groups:
            group.halt_and_join()
            logger.debug("Group '%s' accepted join with main thread.", group.name)

        logger.info("All groups joined with main thread.")

        logger.info("Call _close() on all modules..")
        for mod_handler in self.module_handlers:
            mod_handler.call_close()

    def halt(self):
        self._running = False
