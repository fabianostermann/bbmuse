import logging

from collections import defaultdict, deque
from time import time

logger = logging.getLogger(__name__)

class Controller:

    def __init__(self, module_handlers, blackboard):
        self.module_handlers = module_handlers
        self.blackboard = blackboard

    def build(self):
        self.execution_order, self.dependencies = self.calc_execution_order()

    def calc_execution_order(self):
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

        return exec_order, graph

    def run(self, limit=None):
        cycle_count = 0
        logger.info("Start running..")
        self.running = True
        while self.running and cycle_count != limit:
            start_time = time()
            
            try:
                for mod_handler in self.execution_order:
                    logger.debug("Run update on module %s", mod_handler)
                    mod_handler.run_update(self.blackboard)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt detected: preparing halt..")
                self.halt()

            delta_time = time() - start_time
            cycle_count += 1
            logger.info(f"End of cycle {cycle_count}{"/"+str(limit) if limit is not None else ""}, delta={delta_time:.5f}")

    def halt(self):
        self.running = False
