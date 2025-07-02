import logging
import sys

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.config import Config
from bbmuse.engine.blackboard import Blackboard
from bbmuse.engine.controller import Controller
from bbmuse.engine.module_handler import ModuleHandler
from bbmuse.engine.representation_handler import RepresentationHandler

logger = logging.getLogger(__name__)

class BbMuseProject():

    def __init__(self, project_dir):
        self.controller = None
        self.config = Config(project_dir)

    def build_all(self):
        self.prepare_handlers()
        self.build_handlers()
        self.build_controller()
    
    def prepare_handlers(self):
        # Search for module defintion files
        mods_handlers = []
        for path in self.config["project_dir"].joinpath("Modules").glob("*.py"):
            handler = ModuleHandler(path)
            mods_handlers.append(handler)
        mods_handlers

        if not mods_handlers:
            raise RuntimeError("Did not find any module definitions.")
        logger.debug("Init modules: %s", mods_handlers)

        # Search for representation defintion files
        reps_handlers = []
        for path in self.config["project_dir"].joinpath("Representations").glob("*.py"):
            handler = RepresentationHandler(path)
            reps_handlers.append(handler)
        logger.debug("Init representations: %s", reps_handlers)

        self.potential_module_handlers = mods_handlers
        self.potential_representation_handlers = reps_handlers

        # run build() to fill the following
        self.module_handlers = []
        self.representation_handlers = []

    def build_handlers(self):
        all_provides = []
        mod_handlers = []
        for handler in self.potential_module_handlers:
            try:
                handler.build()
                all_provides += handler.get_provides()
                mod_handlers.append(handler)
            except Exception:
                logger.exception("Build failed for module %s. Skip and ignore.", handler)

        assert mod_handlers, "No modules were successfully build."
        if handler in mod_handlers:
            assert handler.get_build_status(), f"Build status is False for {handler}"

        self.module_handlers = mod_handlers
        logger.debug("List of all provided representations: %s", all_provides)

        rep_handlers = []
        for handler in self.potential_representation_handlers:
            if handler.get_name() in all_provides:
                try:
                    handler.build()
                    rep_handlers.append(handler)
                except Exception:
                    logger.warning("Build failed for representation %s. Skip and ignore.", handler)
            else:
                logger.warning("%s not found in provided representations. Skip import.", handler.get_name())

        assert rep_handlers, "No representations were successfully build."
        if handler in rep_handlers:
            assert handler.get_build_status(), f"Build status is False for {handler}"
        self.representation_handlers = rep_handlers

    def build_controller(self):
        # create blackboard
        blackboard = Blackboard(self.representation_handlers)

        # build controller
        self.controller = Controller(self.module_handlers, blackboard)
        self.controller.build()

    def run(self, *args, **kwargs):
        self.controller.run(*args, **kwargs)
        
