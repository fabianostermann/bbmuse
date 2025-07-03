import logging

from bbmuse.engine.module_handler import ModuleHandler
from bbmuse.engine.representation_handler import RepresentationHandler

logger = logging.getLogger(__name__)

class Blackboard:

    def __init__(self, representation_handlers=[]):
        self._board = {}
        for rep in representation_handlers:
            self.register(rep)

        logger.debug("Blackboard initialized: %s", self.list_content())

    def register(self, rep_handler: RepresentationHandler):
        rep_name = rep_handler.get_name()
        if rep_name.lower() in [name.lower() for name in self._board.keys()]:
            raise ValueError(f"Duplicate representation name (case ignored): {rep_name}")

        # add actual representation object (in readonly view) to blackboard
        self._board[rep_name] = rep_handler

    #def remove(self, representation):
    #    """ remove representation from board either by name or object """
    #    for key, value in self._board.items():
    #        if key == representation or value == representation:
    #            del self._board[key]
    #            break

    def list_content(self):
        return list(self._board.keys())

    def get(self, name):
        """ returns a representation handler by name """
        return self._board[name]

    def create_view(self, module_handler: ModuleHandler):
        readable_keys = module_handler.get_requires() + module_handler.get_uses()
        writable_keys = module_handler.get_provides()
        return _BlackboardView(self, readable_keys, writable_keys)

class _BlackboardView:

    def __init__(self, blackboard: Blackboard, readable_keys=None, writable_keys=None):
        rep_views = {}
        for readable_key in readable_keys:
            rep_views[readable_key] = blackboard.get(readable_key).create_view(read_only=True)
        for writable_key in writable_keys:
            rep_views[writable_key] = blackboard.get(writable_key).create_view(read_only=False)
        object.__setattr__(self, "_rep_views", rep_views)
        
    def __getattr__(self, name):
        return self._rep_views[name]

    def __setattr__(self, name, value):
        raise AttributeError(f"Setting attribute '{name}' on a BlackboardView is not allowed.")

    def __delattr__(self, name):
        raise AttributeError(f"Deleting attribute '{name}' from a BlackboardView is not allowed.")
    