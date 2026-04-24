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

        self.project.run(run_mode=0)

        if listener in listeners:
            rep_arrays = listener.flush()
            if not args.dry_run:
                if rep_arrays:
                    ep_path = self.module_manager.get_next_episode_path(listener.get_module_handler())
                    np.savez_compressed(ep_path, **rep_arrays)
                else:
                    logger.warning("Rep_array was empty. Nothing to write to disk.")
