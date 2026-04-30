import logging
import sys, os

from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

from bbmuse.learn.module_listener import ModuleListener

class ListeningSession:
    def __init__(self, project, module_manager):
        self.project = project
        self.module_manager = module_manager

    def run(self, args):
        listeners = []

        for mh in self.project.get_module_handlers():
            if self.module_manager.is_armed(mh):
                listener = ModuleListener(mh, self.project.get_blackboard())
                listeners.append(listener)
                listener.activate_listen()

        if not listeners:
            logger.info("No modules are armed.")
            sys.exit()

        self.project.run(run_mode=0, quit_after=args.quit_after)

        for listener in listeners:
            rep_arrays = listener.flush()
            T = next(iter(rep_arrays.values())).shape[0]
            logger.debug(f"Listener on {listener.get_module_handler()} has finished with {T} timesteps.")

            if not args.dry_run:
                if rep_arrays:
                    ep_path = self.module_manager.get_next_episode_path(listener.get_module_handler(), tag=args.tag)
                    np.savez_compressed(ep_path, **rep_arrays)
                    logger.info("Record from ListeningSession stored at: %s", ep_path)
                else:
                    logger.warning("Rep_array was empty. Nothing to write to disk.")
