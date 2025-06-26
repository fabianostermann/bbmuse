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
        assert isinstance(getattr(module, "PROVIDES", []), list)
        assert isinstance(getattr(module, "REQUIRES", []), list)
        assert isinstance(getattr(module, "USES", []), list)

        # check for required methods
        update_method = getattr(module, "_update", None)
        if update_method is None:
            raise SyntaxError(f"Module {self} has no _update() method.")
        if not callable(update_method):
            raise SyntaxError(f"_update() in {self} is not callable.")

        # overwrite default print
        def print_with_name_tag(*args, **kwargs):
            # print only if global log level is INFO or less
            if logger.getEffectiveLevel() <= logging.INFO:
                # tag output with module name
                print(f"MODULE {self.get_name()}:", *args, **kwargs)
        module.print = print_with_name_tag

        self.set_component(module) # also sets build_status to True
    
    #def __str__(self):
    #    return f"<Module:{self.get_name()}>"

    """ Mandatory attributes """
    def get_provides(self):
        return getattr(self.get_component(), "PROVIDES", [])
    
    def get_requires(self):
        return getattr(self.get_component(), "REQUIRES", [])
    
    def run_update(self, bb):
        self.get_component()._update(bb)

    """ Optional attributes"""
    def get_uses(self):
        return getattr(self.get_component(), "USES", [])
    