import logging

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class RepresentationHandler(BaseHandler):

    def build(self):
        rep = self.dynamic_import_from_file(self.get_file_location())
        self.set_component(rep) # also sets build_status to True

    #def __str__(self):
    #    return f"<Repr:{self.get_name()}>"

    def create_view(self, read_only=False):
        return _RepresentationView(self.get_component(), read_only=read_only)
    
class _RepresentationView():
    def __init__(self, representation, read_only=False):
        object.__setattr__(self, "_representation", representation)
        if read_only:
            object.__setattr__(self, "_allowed", set())
        else:
            object.__setattr__(self, "_allowed", set(dir(representation)))

    def __getattr__(self, name):
        return getattr(self._representation, name)

    def __setattr__(self, name, value):
        if name not in self._allowed:
            raise AttributeError(f"Setting attribute '{name}' on <{self._representation.__name__}> is not allowed.")
        setattr(self._representation, name, value)

    def __delattr__(self, name):
        raise AttributeError(f"Deleting attribute '{name}' from {self._representation.__name__} is not allowed.")
    