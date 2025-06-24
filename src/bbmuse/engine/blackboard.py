import logging

logger = logging.getLogger(__name__)

class Blackboard:

    def __init__(self, representations=[]):
        self._board = {}
        for r in representations:
            self.register(r)

        logger.info("Blackboard initialized: %s", self._board)

    def register(self, representation):
        rep_name = representation.__class__.__name__
        if rep_name.lower() in [name.lower() for name in self._board.keys()]:
            raise ValueError(f"Duplicate representation name (case ignored): {rep_name}")

        # add representation to blackboard
        self._board[rep_name] = representation

    #def remove(self, representation):
    #    """ remove representation from board either by name or object """
    #    for key, value in self._board.items():
    #        if key == representation or value == representation:
    #            del self._board[key]
    #            break

    def get(self, name):
        return self._board[name]

    def __getitem__(self, name):
        return self.get(name)  # allows blackboard["<key>"]

#     def create_view(self, allowed_keys):
#         return BlackboardView(self, allowed_keys)
        
# class BlackboardView:
#     def __init__(self, blackboard, allowed_keys):
#         self._bb = blackboard
#         self._allowed = set(allowed_keys)

#     def get(self, key):
#         if key not in self._allowed:
#             raise KeyError(f"Access denied: {key}")
#         return self._bb.require(key)

#     def set(self, key, value):
#         if key not in self._allowed:
#             raise KeyError(f"Access denied: {key}")
#         self._bb.provide(key, value)

