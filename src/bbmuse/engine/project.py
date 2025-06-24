import logging

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.config import Config, syntax_keywords
from bbmuse.engine.blackboard import Blackboard
from bbmuse.engine.controller import Controller

logger = logging.getLogger(__name__)

class BbMuseProject():

    def __init__(self, project_dir):
        self.controller = None
        self.config = Config(project_dir)

    def build(self):
        modules = self.init_modules()
        representations = self.init_representations(modules)

        # make blackboard
        blackboard = Blackboard(representations)

        # build controller
        self.controller = Controller(modules, blackboard)
        self.controller.build()

    def init_representations(self, modules):
        representations = []

        all_provides = [rep for module in modules for rep in module.provides]
        unused = []

        for path in self.config["project_dir"].joinpath("Representations").glob("*.py"):
            mod = self.dynamic_import_from_file(path)

            candidates = [
                obj for name, obj in inspect.getmembers(mod)
                if inspect.isclass(obj)
            ]

            if len(candidates) > 1:
                raise RuntimeError(
                    f"{path.name}' provides more than one class {[c.__name__ for c in candidates]}. Only one class is allowed per representation file."
                )
            elif len(candidates) == 1:
                # only initialize representations that are used
                if candidates[0].__name__ in all_provides:
                    instance = candidates[0]()
                    representations.append(instance)
                else:
                    unused.append(candidates[0])

        logger.info("Instantiated representations: %s", representations)
        logger.debug("Unused representation (not instantiated): %s", unused)
        return representations

    def init_modules(self):
        modules = []
        for path in self.config["project_dir"].joinpath("Modules").glob("*.py"):
            mod = self.dynamic_import_from_file(path)

            candidates = [
                obj for name, obj in inspect.getmembers(mod)
                if inspect.isclass(obj) and callable(getattr(obj, "update", None))
            ]

            if len(candidates) > 1:
                raise RuntimeError(
                    f"{path.name}' provides more than one class {[c.__name__ for c in candidates]}. Only one class is allowed per module file."
                )
            elif len(candidates) == 1:
                instance = candidates[0]()
                # check sanity of module syntax
                for attribute in [ syntax_keywords["REQUIRES"], syntax_keywords["PROVIDES"] ]:
                    if not hasattr(instance, attribute):
                        raise RuntimeError(f"Module {instance} has no '{attribute}'.")
                    if not isinstance(getattr(instance, attribute), list):
                        raise RuntimeError(f"'{attribute}' in module {instance} is not of type list.")
                modules.append(instance)

        logger.info("Instantiated modules: %s", modules)
        return modules

    def dynamic_import_from_file(self, filepath: Path):
        """ Perform dynamic import of a python module from a given filepath
        """
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        python_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(python_module)
        return python_module

    def run(self, *args, **kwargs):
        if self.controller is None:
            logger.info("No controller available. Building it first..")
            self.build()

        self.controller.run(*args, **kwargs)
        
