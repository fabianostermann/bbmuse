import logging

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class RepresentationHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation_views = []

    def build(self):
        rep = self.dynamic_import_from_file(self.get_file_location())
        self.call_validate()
        self.set_component(rep) # also sets build_status to True

        # overwrite default print
        def print_with_name_tag(*args, **kwargs):
            # print only if global log level is INFO or less
            if logger.getEffectiveLevel() <= logging.INFO:
                # tag output with module name and group name
                print(f"REPRESENTATION {self.get_name()}:", *args, **kwargs)
        rep.print = print_with_name_tag
        
    def hot_reload(self):
        logger.debug("Hot-reloading %s..", self)
        old_component = self.get_component()
        try:
            self.build() 
        except Exception:
            logger.exception("Error when building representation %s. Keeping former instance.", self)
            self._component = old_component
            
        for rep_view in self.representation_views:
            rep_view._rebind(self._component)
            
        logger.info("Hot-reload on %s was successful.", self)

    #def __str__(self):
    #    return f"<Repr:{self.get_name()}>"

    def call_validate(self):
        if callable(getattr(self.get_component(), "_validate", None)):
            self.get_component()._validate()
            
    def create_view(self, read_only=False):
        rep_view = _RepresentationView(self.get_component(), read_only=read_only)
        self.representation_views.append(rep_view)
        logger.debug("%s, %s", self, self.representation_views)
        return rep_view
    
class _RepresentationView():
    def __init__(self, representation, read_only=False):
        self._rebind(representation, read_only=read_only)

    def _rebind(self, representation, read_only=None):
        """Point this view at a (new) underlying component, in place."""
        if read_only is None:
            read_only = (self._allowed == set())  # preserve current mode
        object.__setattr__(self, "_representation", representation)
        object.__setattr__(self, "_allowed", set() if read_only else set(dir(representation)))

    def __getattr__(self, name):
        return getattr(self._representation, name)

    def __setattr__(self, name, value):
        if name not in self._allowed:
            raise AttributeError(f"Setting attribute '{name}' on <{self._representation.__name__}> is not allowed.")
        setattr(self._representation, name, value)

    def __delattr__(self, name):
        raise AttributeError(f"Deleting attribute '{name}' from {self._representation.__name__} is not allowed.")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"RepresentationView(name={self._representation.__name__},read_only={self._allowed == set()})"
