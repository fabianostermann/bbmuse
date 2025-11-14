import logging

from collections import defaultdict, deque
from time import time

import threading

base_logger = logging.getLogger(__name__)

class ControlGroup:

    def __init__(self, module_handlers, blackboard):
        group_names = list(set([handler.get_group() for handler in module_handlers]))
        assert len(group_names) == 1, f"Got modules from {len(group_names)} groups: {group_names}"

        self.name = group_names[0]
        self.module_handlers = module_handlers
        self.blackboard = blackboard

        self.thread = threading.Thread(target=self.run, daemon=True)

        self.logger = base_logger
        if self.name != "default":
            self.logger = logging.getLogger(__name__ + f" (group:{self.name})")
        
        self.logger.info("Created group '%s'. Members are: %s", self.name, self.module_handlers)

    def build(self, exec_order):
        self.execution_order = [handler for handler in exec_order if handler in self.module_handlers]
        self.build_blackboard_views()

    def build_blackboard_views(self):
        bb_views = {}
        for handler in self.module_handlers:
            bb_views[handler] = self.blackboard.create_view(handler)

        self.blackboard_views = bb_views

    def start(self):
        self.thread.start()

    def halt_and_join(self, timeout=None):
        self.halt()
        self.thread.join(timeout=timeout)
    
    def is_alive(self):
        self.thread.is_alive()

    def run(self):
        cycle_count = 0
        self.logger.info(f"Execution order: %s", self.execution_order)
        self.logger.info(f"Blackboard contents: %s", self.blackboard.list_content())

        self._running = True

        self.logger.info("Start running..")
        while self._running:
            start_time = time()
            
            try:
                for mod_handler in self.execution_order:
                    self.logger.debug("Call _update() on module %s", mod_handler)
                    
                    # TODO also, sanity check by pickling that read-only (required and used) representations are not altered
                    try:
                        if self._running: # check if not already halted
                            mod_handler.call_update(self.blackboard_views[mod_handler])
                    except Exception:
                        # TODO: Future improv: Break in dev mode, ignore in release mode.
                        self.logger.exception(f"Module {mod_handler} produced an error.")
            except KeyboardInterrupt:
                self.logger.exception("KeyboardInterrupt detected: signal halt..")
                self.halt()

            delta_time = time() - start_time
            cycle_count += 1

            self.logger.info(f"End of cycle {cycle_count}, delta={delta_time:.5f}")

    def halt(self):
        self._running = False
