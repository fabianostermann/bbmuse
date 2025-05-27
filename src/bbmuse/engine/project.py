from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.config import Config
from bbmuse.engine.blackboard import Blackboard
from bbmuse.engine.controller import Controller

from bbmuse.engine.representation_base import RepresentationBase
from bbmuse.engine.module_base import ModuleBase

class BbMuseProject():

    def __init__(self, project_dir):
        # load config
        self.config = Config(project_dir)
        #print("Config loaded:", self.config)

        self.controller = None

    def build(self):
        representations = self.load_representations()
        #print("Representations:", representations)

        modules = self.load_modules()
        #print("Modules:", modules)

        # init blackboard
        blackboard = Blackboard(representations)
        print("Blackboard content:", blackboard._board)

        # init and build controller
        self.controller = Controller(modules, blackboard)
        self.controller.build()

    def load_representations(self):
        representations = []
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
                instance = candidates[0]()
                representations.append(instance)

        return representations

    def load_modules(self):
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
                modules.append(instance)

        return modules

    def dynamic_import_from_file(self, filepath: Path):
        """ Perform dynamic import of a python module from a given filepath
        """
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        python_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(python_module)
        return python_module

    def run(self):
        if self.controller is None:
            self.build()

        self.controller.run()
        
