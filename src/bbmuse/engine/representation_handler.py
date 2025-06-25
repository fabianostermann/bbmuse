import logging

from pathlib import Path
import importlib.util
import inspect

from bbmuse.engine.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class RepresentationHandler(BaseHandler):

    def build(self):
        rep = self.dynamic_import_from_file(self.get_file_location())

        # TODO check sanity
        # nothing to do for now

        self.set_component(rep)