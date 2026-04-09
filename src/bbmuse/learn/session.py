import logging
import sys, os

from pathlib import Path

logger = logging.getLogger(__name__)

from bbmuse.learn.module_manager import ModuleManager
from bbmuse.learn.listening_session import ListeningSession

class Session():

    def __init__(self, project, args):
        logger.debug(args)

        # build project to setup modules, blackboard and dependency graph
        try:
            requested_log_level = logging.getLogger().level
            logger.debug("If on INFO log level, increase level to ERROR to suppress bbmuse project loading output.")
            if logging.getLogger().level == logging.INFO:
                logging.getLogger().setLevel(logging.ERROR)
            project.build_all()
        except Exception:
            logger.exception("Building project failed.")
            sys.exit(1)
        finally:
            logging.getLogger().setLevel(requested_log_level)

        self.project = project
        self.module_manager = ModuleManager(project)

        if hasattr(self, args.command):
            command_method = getattr(self, args.command)
            command_method(args)
        else:
            logger.error("Command '%s()' is unknown.", args.command)

    def arm(self, args):
        self.module_manager.arm(args)

    def disarm(self, args):
        self.module_manager.disarm(args)

    def status(self, args):
        self.module_manager.status(args)

    def listen(self, args):
        ls = ListeningSession(self.project, self.module_manager)
        ls.run(args)

    def clone(self, args):
        logger.error("clone() is not implemented yet.")

    def sculpt(self, args):
        logger.error("sculpt() is not implemented yet.")

    def apply(self, args):
        logger.error("apply() is not implemented yet.")

    def restore(self, args):
        logger.error("restore() is not implemented yet.")