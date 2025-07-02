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

        # TODO delete attributes (delattr) PROVIDES, REQUIRES and USES from module object

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

        self.known_bindings = dict()

        self._set_component(module) # also sets build_status to True
    
    def bind_representation(self, name, representation):
        assert self.get_build_status()
        assert name not in self.known_bindings.keys()

        module = self.get_component()
        setattr(module, name, representation)
        self._set_component(module)
        # register for sanity checks
        self.known_bindings[name] = representation

    #def __str__(self):
    #    return f"<Module:{self.get_name()}>"

    """ bbmuse features """
    def get_provides(self):
        return getattr(self.get_component(), "PROVIDES", [])
    
    def get_requires(self):
        return getattr(self.get_component(), "REQUIRES", [])
    
    def get_uses(self):
        return getattr(self.get_component(), "USES", [])
    
    def run_update(self):
        self.get_component(read_only=True)._update()
        # sanity checks:
        for name, rep in self.known_bindings.items():
            print("->>>>", getattr(self.get_component(), name))
            assert getattr(self.get_component(), name) is rep, "representation was overwritten!"
