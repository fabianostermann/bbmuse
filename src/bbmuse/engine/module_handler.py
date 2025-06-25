import logging

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class ModuleHandler(BaseHandler):

    def build(self):
        self.reset_build_status()
        module = self.dynamic_import_from_file(self.get_file_location())

        # check for required attributes
        for attribute in [ "REQUIRES", "PROVIDES" ]:
            if not hasattr(module, attribute):
                raise SyntaxError(f"Module {module} has no '{attribute}'.")
            if not isinstance(getattr(module, attribute), list):
                raise SyntaxError(f"'{attribute}' in module {module} is not of type list.")

        # check for required methods
        update_method = getattr(module, "_update", None)
        if update_method is None:
            raise SyntaxError(f"Module {self} has no _update() method.")
        if not callable(update_method):
            raise SyntaxError(f"_update() in {self} is not callable.")

        # add module name tag to print function
        def print_with_name_tag(*args, **kwargs):
            # print only if global log level is INFO or less
            if logger.getEffectiveLevel() <= logging.INFO:
                print(f"MODULE {self.get_name()}:", *args, **kwargs)
        module.print = print_with_name_tag

        self.set_component(module)

    def get_provides(self):
        return self.get_component().PROVIDES
    
    def get_requires(self):
        return self.get_component().REQUIRES
    
    def run_update(self, bb):
        self.get_component()._update(bb)
