import logging
import sys, os

from pathlib import Path

logger = logging.getLogger(__name__)

class ListeningSession:
    def __init__(self, project, module_manager):
        self.project = project
        self.module_manager = module_manager

    def run(self, args):
        for mh in self.project.get_module_handlers():
            if self.module_manager.is_armed(mh):
                # wrap mh with ListeningModuleHandler
                logger.info("%s is ready to be listened to.", mh)
            else:
                logger.info("%s will not be listened to.", mh)