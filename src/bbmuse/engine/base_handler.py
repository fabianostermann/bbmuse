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
        
        self.consider_hot_reload()
        
        self._component = None
        logger.debug(f"Init handler for {self.get_name()} located at {self.get_file_location()}")

    def get_file_location(self):
        return self._file_location
    
    def get_name(self):
        return self._name

    def build(self):
        raise NotImplementedError("Do not use base class. Use subclasses.")
        
    def hot_reload(self):
        raise NotImplementedError("Do not use base class. Use subclasses.")

    def dynamic_import_from_file(self, filepath):
        """
        Perform dynamic import of a python module from a given file location
        """
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        python_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(python_module)
        return python_module
        
    def consider_hot_reload(self):
        if not hasattr(self, "_last_mtime"):
            self._last_mtime = self.get_mtime()
            return
    	
        curr_mtime = self.get_mtime()
        # reload if last modification time is older than 1 second
        if self._last_mtime + 1 < curr_mtime:
            self.hot_reload()
            self._last_mtime = curr_mtime
    	
    def get_mtime(self):
        if self._file_location:
            return self._file_location.stat().st_mtime
        else:
            return None
        
    def get_component(self):
        return self._component
    
    def set_component(self, c):
        assert c is not None
        self._component = c

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f"<{self.get_name()}>"
