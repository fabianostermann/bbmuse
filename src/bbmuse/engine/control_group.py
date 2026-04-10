import logging

from collections import defaultdict, deque
from time import time

import threading
GLOBAL_UPDATE_LOCK = threading.Lock()

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

        self.logger.info("Start running..")
        while self._running:
            start_time = time()
            
            try:
                for mod_handler in self.execution_order:
                    #self.logger.debug("Call _update() on module %s", mod_handler)

                    try:
                        if self.run_mode <= 0: # DEBUG or NORMAL mode
                            for rep_name in mod_handler.get_requires():
                                self.blackboard.get(rep_name).consider_hot_reload()
                    
                        if self._running and mod_handler.is_active():
                            with GLOBAL_UPDATE_LOCK:
                                mod_handler.call_update(self.blackboard_views[mod_handler])
                                
                        if self.run_mode < 0: # DEBUG mode
                            try:
                                for rep_name in mod_handler.get_provides():
                                    self.blackboard.get(rep_name).call_validate()
                            except Exception:
                                self.logger.exception(f"Representation {rep_name} did not pass validation check.")
                                self.halt()
                            # TODO in DEBUG mode: additional validation by pickling that read-only (required and used) representations are not altered

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

    def halt(self):
        self._running = False
