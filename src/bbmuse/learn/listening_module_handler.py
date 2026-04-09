import logging

logger = logging.getLogger(__name__)

from bbmuse.engine.module_handler import ModuleHandler

class WrappedModuleHandler(ModuleHandler):
    def call_update(self, bb):
        # <before_hook>
        super().call_update(bb)
        # <after_hook>