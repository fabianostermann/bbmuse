import logging

from pathlib import Path
import importlib.util
import inspect

logger = logging.getLogger(__name__)

class BaseHandler():

    def __init__(self, path):
        path = Path(path)
        if not path.exists() or not path.is_file:
            raise FileNotFoundError(f"File path is no valid file: {path}")
        self._file_location = path.absolute()
        self._name = path.stem

        self.reset_build_status()
        logger.debug(f"Init handler for {self.get_name()} located at {self.get_file_location()}")

    def reset_build_status(self):
        self._build_success = False
        self._component = None

    def get_build_status(self):
        return self._build_success

    def get_file_location(self):
        return self._file_location
    
    def get_name(self):
        return self._name

    def build(self):
        raise NotImplementedError("Do not use base class. Use subclasses.")

    def dynamic_import_from_file(self, filepath):
        """
        Perform dynamic import of a python module from a given file location
        """
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        python_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(python_module)
        return python_module
    
    def get_component(self):
        return self._component
    
    def set_component(self, c):
        """
        Only call this if the component was successfully build.
        It sets the build status to True.
        """
        self._component = c
        self._build_success = True

    def __repr__(self):
        return f"<{self.get_name()}>"
    
    def __str__(self):
        return f"<{self.get_name()}>"