import logging

logger = logging.getLogger(__name__)

import numpy as np

from bbmuse.engine.module_handler import ModuleHandler
from bbmuse.engine.blackboard import Blackboard

class ModuleListener:

    def __init__(self, mod_handler: ModuleHandler, blackboard: Blackboard):
        self._mod_handler = mod_handler
        self._blackboard = blackboard

        self._active = False
        self._original_call_update = None
        self._expected_shapes = {} # rep_name -> list of shapes
        self._requires_buffer = {}  # rep_name -> list of arrays
        self._uses_buffer = {}  # rep_name -> list of arrays
        self._provides_buffer = {}  # rep_name -> list of arrays

    def _check_requirements(self):
        # check for required methods
        for required_rep_name in self._mod_handler.get_requires():
            rh = self._blackboard.get(required_rep_name)
            self._check_function_exists(rh, "_pack")
        for used_rep_name in self._mod_handler.get_uses():
            rh = self._blackboard.get(used_rep_name)
            self._check_function_exists(rh, "_pack")
        for provided_rep_name in self._mod_handler.get_provides():
            rh = self._blackboard.get(provided_rep_name)
            self._check_function_exists(rh, "_pack")
    
    def _check_function_exists(self, rep_handler, func_name):
        func = getattr(rep_handler.get_component(), func_name, None)
        if func is None:
            raise SyntaxError(f"Repr. {rep_handler} has no {func_name}() method.")
        if not callable(func):
            raise SyntaxError(f"{func_name}() in {rep_handler} is not callable.")

    def activate_listen(self):
        self._check_requirements()

        if self._active:
            return
        self._original_call_update = self._mod_handler.call_update  # capture bound method
        
        original = self._original_call_update
        listener = self  # capture self for use inside wrapper

        def call_update_wrapped(bb):
            listener._before_hook()
            result = original(bb)
            listener._after_hook()
            return result

        # monkey-patching the wrapped update method to the handler instance
        self._mod_handler.call_update = call_update_wrapped
        self._active = True

        logger.debug("Listening activated on %s", self._mod_handler)

    def deactivate_listen(self):
        if not self._active:
            return
        self._mod_handler.call_update = self._original_call_update  # restore
        self._original_call_update = None
        self._active = False

        logger.debug("Listening deactivated on %s", self._mod_handler)

    def _before_hook(self):
        logger.debug("Running _before_hook() on module %s", self._mod_handler)
        for required_rep_name in self._mod_handler.get_requires():
            rh = self._blackboard.get(required_rep_name)
            self._store(rh, self._requires_buffer)
        for used_rep_name in self._mod_handler.get_uses():
            rh = self._blackboard.get(used_rep_name)
            self._store(rh, self._uses_buffer)

    def _after_hook(self):
        logger.debug("Running _after_hook() on module %s", self._mod_handler)
        for provided_rep_name in self._mod_handler.get_provides():
            rh = self._blackboard.get(provided_rep_name)
            self._store(rh, self._provides_buffer)

    def _store(self, rep_handler, buffer):
        rep_array = self._perform_pack(rep_handler)
        rep_name = rep_handler.get_name()
        if rep_name not in buffer:
            buffer[rep_name] = []
        buffer[rep_name].append(rep_array)

    def _perform_pack(self, rep_handler):
        rep_array = rep_handler.get_component()._pack()
        assert isinstance(rep_array, np.ndarray), \
            f"_pack() in {rep_handler} must return a numpy array"
        
        rep_name = rep_handler.get_name()
        if rep_name not in self._expected_shapes:
            self._expected_shapes[rep_name] = rep_array.shape  # first call, store it
        else:
            assert rep_array.shape == self._expected_shapes[rep_name], \
                f"_pack() in {rep_handler} returned shape {rep_array.shape}, " \
                f"expected {self._expected_shapes[rep_name]}"
        
        return rep_array

    def flush(self):
        rep_arrays = {}
        rep_arrays |= {f"requires__{k}": np.stack(v) for k, v in self._requires_buffer.items()}
        rep_arrays |= {f"uses__{k}": np.stack(v) for k, v in self._uses_buffer.items()}
        rep_arrays |= {f"provides__{k}": np.stack(v) for k, v in self._provides_buffer.items()}
        # shape per rep: (n_timesteps, *rep_shape)

        if not rep_arrays:
            logger.warning("Flush called on %s but buffers are empty.", self._mod_handler)
            return {}

        timestep_counts = {k: v.shape[0] for k, v in rep_arrays.items()}
        min_timesteps = min(timestep_counts.values())
        if len(set(timestep_counts.values())) > 1:
            logger.warning(
                "Inconsistent timestep counts in %s, truncating to %d: %s",
                self._handler.get_name(), min_timesteps, timestep_counts
            )
            rep_arrays = {k: v[:min_timesteps] for k, v in rep_arrays.items()}

        self._requires_buffer.clear()
        self._uses_buffer.clear()
        self._provides_buffer.clear()
        return rep_arrays

    def get_module_handler(self):
        return self._mod_handler