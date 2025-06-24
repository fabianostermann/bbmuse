import logging

from collections import defaultdict, deque
from time import time

from bbmuse.engine.config import syntax_keywords

logger = logging.getLogger(__name__)

class Controller:

    def __init__(self, modules, blackboard):
        self.modules = modules
        self.blackboard = blackboard

    def build(self):
        self.execution_order, self.dependencies = self.calc_execution_order()

    def calc_execution_order(self):
        # construct mapping: repr -> provider
        provides_map = {}
        for m in self.modules:
            for repr in getattr(m, syntax_keywords["PROVIDES"], []):
                if not repr in self.blackboard._board.keys():
                    raise RuntimeError(f"Representation {repr} is unknown to the blackboard, thus cannot be provided by module {m}.")
                if not repr in provides_map.keys():
                    provides_map[repr] = m
                else:
                    raise RuntimeError(f"Duplicate provide: Representation {repr} provided by modules {m} and {provides_map[repr]}.")
        logger.debug("Map repr -> provider: %s", provides_map)

        # Build the graph: edges from providers -> consumers
        graph = defaultdict(list)
        num_of_consumers = {m: 0 for m in self.modules}

        for m in self.modules:
            for req in getattr(m, syntax_keywords["REQUIRES"], []):
                provider = provides_map.get(req, None)
                if provider is None:
                    raise RuntimeError(f"No module provides required representation: {req}")
                else:
                    if not m in graph[provider]:
                        graph[provider].append(m)
                        num_of_consumers[m] += 1
        logger.debug(f"Map provider -> list of consumers: {graph}")
        logger.debug(f"Num. of consumers per provider: {num_of_consumers}")

        # Topological Sort: Kahn's algorithm (doi:10.1145/368996.369025)
        ready = deque([m for m, deg in num_of_consumers.items() if deg == 0])
        exec_order = []

        while ready:
            m = ready.popleft()
            exec_order.append(m)
            for neighbor in graph[m]:
                num_of_consumers[neighbor] -= 1
                if num_of_consumers[neighbor] == 0:
                    ready.append(neighbor)
        logger.debug(f"Proposed execution order: %s", exec_order)

        if len(exec_order) != len(self.modules):
            raise RuntimeError("Cycle detected in module dependencies")

        return exec_order, graph

    def run(self, limit=None):
        cycle_count = 0
        logger.info("Start running..")
        self.running = True
        while self.running and cycle_count != limit:
            start_time = time()
            
            try:
                for module in self.execution_order:
                    logger.debug("Update %s", module)
                    module.__class__.update(self.blackboard)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt detected: preparing halt..")
                self.halt()

            delta_time = time() - start_time
            cycle_count += 1
            logger.info(f"End of cycle {cycle_count}{"/"+str(limit) if limit is not None else ""}, delta={delta_time:.5f}")
            

    def halt(self):
        self.running = False
