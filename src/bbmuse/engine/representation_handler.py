import logging

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class RepresentationHandler(BaseHandler):

    def build(self):
        rep = self.dynamic_import_from_file()

        # TODO check sanity

        self.set_component(rep)

    # def handle_representations(self, modules):
    #     representations = []

    #     all_provides = [rep for module in modules for rep in module.provides]
    #     unused = []

    #     for path in self.config["project_dir"].joinpath("Representations").glob("*.py"):
    #         mod = self.dynamic_import_from_file(path)

    #         candidates = [
    #             obj for name, obj in inspect.getmembers(mod)
    #             if inspect.isclass(obj)
    #         ]

    #         if len(candidates) > 1:
    #             raise RuntimeError(
    #                 f"{path.name}' provides more than one class {[c.__name__ for c in candidates]}. Only one class is allowed per representation file."
    #             )
    #         elif len(candidates) == 1:
    #             # only initialize representations that are used
    #             if candidates[0].__name__ in all_provides:
    #                 instance = candidates[0]()
    #                 representations.append(instance)
    #             else:
    #                 unused.append(candidates[0])

    #     logger.info("Instantiated representations: %s", representations)
    #     logger.debug("Unused representation (not instantiated): %s", unused)
    #     return representations
