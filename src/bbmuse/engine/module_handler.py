import logging

from pathlib import Path
import importlib.util
import inspect

import threading
GLOBAL_UPDATE_LOCK = threading.Lock()

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class ModuleHandler(BaseHandler):

    def build(self):
        self.reset_build_status()
        module = self.dynamic_import_from_file(self.get_file_location())

        # attributes
        self._is_running = False

        # check for required attributes
        assert isinstance(self.get_provides(), list)
        assert isinstance(self.get_requires(), list)
        assert isinstance(self.get_uses(), list)
        assert isinstance(self.get_group(), str)

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
                # tag output with module name and group name
                print(f"MODULE {self.get_name()}{"" if self.get_group() == "default" else f" (group:{self.get_group()})"}:", *args, **kwargs)
        module.print = print_with_name_tag

        self.set_component(module) # also sets build_status to True
    
    #def __str__(self):
    #    return f"<Module:{self.get_name()}>"

    """ Mandatory attributes """
    def call_update(self, bb):
        # make updates atomic
        with GLOBAL_UPDATE_LOCK:
            self._is_running = True
            try:
                self.get_component()._update(bb)
            finally:
                self._is_running = False

    def is_running(self):
        return self._is_running

    """ Optional attributes"""
    def get_provides(self):
        return getattr(self.get_component(), "PROVIDES", [])
    
    def get_requires(self):
        return getattr(self.get_component(), "REQUIRES", [])

    def get_uses(self):
        return getattr(self.get_component(), "USES", [])
        
    def get_group(self):
        return getattr(self.get_component(), "GROUP", "default")
        
    def is_active(self):
        return getattr(self.get_component(), "ACTIVE", True)

    def call_init(self):
        if callable(getattr(self.get_component(), "_init", None)):
            self.get_component()._init()
        else:
            logger.debug("No _init() function found in module %s.", self.get_name())

    def call_close(self):
        if callable(getattr(self.get_component(), "_close", None)):
            self.get_component()._close()
        else:
            logger.debug("No _close() function found in module %s.", self.get_name())