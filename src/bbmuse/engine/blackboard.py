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
        self._board[rep_name] = rep_handler.get_component(read_only=True)

    #def remove(self, representation):
    #    """ remove representation from board either by name or object """
    #    for key, value in self._board.items():
    #        if key == representation or value == representation:
    #            del self._board[key]
    #            break

    def list_content(self):
        return list(self._board.keys())

    def get(self, key):
        return self._board[key]

    def __getitem__(self, key):
        raise RuntimeError("__getitem__ (syntax: bb[key]) is not allowed for the global blackboard, only for views.")

    def create_view(self, module_handler: ModuleHandler):
        readable_keys = module_handler.get_requires() + module_handler.get_uses()
        writable_keys = module_handler.get_provides()
        return _BlackboardView(self, readable_keys, writable_keys)
        
class _BlackboardView:
    def __init__(self, board: dict, readable_keys=None, writable_keys=None):
        self._board = board
        self._readable = set(readable_keys or [])
        self._writable = set(writable_keys or [])
    
    def get(self, key):
        raise RuntimeError("get() is only allowed in the global blackboard, not for views.")

    def __getitem__(self, key):
        if key not in self._readable and key not in self._writable:
            raise KeyError(f"Obtaining representation '{key}' is not allowed. No use, require or provide was specified by module definition.")
        return self._board.get(key)


