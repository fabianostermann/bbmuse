import logging

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class ModuleHandler(BaseHandler):

    def build(self):
        mod = self.dynamic_import_from_file()

        # TODO check sanity

        # candidates = [
        #     obj for name, obj in inspect.getmembers(mod)
        #     if inspect.isclass(obj) and callable(getattr(obj, "update", None))
        # ]

        # if len(candidates) > 1:
        #     raise RuntimeError(
        #         f"{path.name}' provides more than one class {[c.__name__ for c in candidates]}. Only one class is allowed per module file."
        #     )
        # elif len(candidates) == 1:
        #     instance = candidates[0]()
        #     # check sanity of module syntax
        #     for attribute in [ syntax_keywords["REQUIRES"], syntax_keywords["PROVIDES"] ]:
        #         if not hasattr(instance, attribute):
        #             raise RuntimeError(f"Module {instance} has no '{attribute}'.")
        #         if not isinstance(getattr(instance, attribute), list):
        #             raise RuntimeError(f"'{attribute}' in module {instance} is not of type list.")
        #     modules.append(instance)

        self.set_component(mod)

    def get_provides(self):
        return self.get_component().PROVIDES
    
    def get_requires(self):
        return self.get_component().REQUIRES
    
    def run_update(self, bb):
        self.get_component().update(bb)
