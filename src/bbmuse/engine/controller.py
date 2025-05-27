from collections import defaultdict, deque
from time import time

class Controller:

    def __init__(self, modules, blackboard):
        self.modules = modules
        self.blackboard = blackboard

    def build(self):
        self.validate()
        self.execution_order = self.build_execution_order()

    def validate(self):
        pass # validate required module contents, etc.

    def build_execution_order(self):
        # Map: repr -> list of modules that provide it
        provides_map = defaultdict(list)
        for m in self.modules:
            for p in getattr(m, "provides", []):
                provides_map[p].append(m)

        # Build the graph: edges from providers to consumers
        graph = defaultdict(set)
        in_degree = {m: 0 for m in self.modules}

        for m in self.modules:
            for req in getattr(m, "requires", []):
                providers = provides_map.get(req, [])
                if not providers:
                    raise RuntimeError(f"No module provides required representation: {req}")
                for provider in providers:
                    if provider == m:
                        continue
                    graph[provider].add(m)
                    in_degree[m] += 1

        # Topological Sort (Kahn's algorithm)
        ready = deque([m for m, deg in in_degree.items() if deg == 0])
        execution_order = []

        while ready:
            m = ready.popleft()
            execution_order.append(m)
            for neighbor in graph[m]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    ready.append(neighbor)

        if len(execution_order) != len(self.modules):
            raise RuntimeError("Cycle detected in module dependencies")

        return execution_order

    def run(self):
        iteration_counter = 0
        print("Controller: Start loop..")
        while iteration_counter < 10:
            start_time = time()

            for mod in self.execution_order:
                mod.__class__.update(self.blackboard)

            delta_time = time() - start_time
            iteration_counter += 1
            print(f"End of loop {iteration_counter}, delta={delta_time}.")

