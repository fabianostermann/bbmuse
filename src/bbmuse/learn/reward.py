import logging

import importlib.util
from pathlib import Path

logger = logging.getLogger(__name__)

class Reward:
    def __init__(self, reward_filepath: str | Path):
        self.reward_filepath = Path(reward_filepath).resolve()
        self.name = self.reward_filepath.stem

        if not self.reward_filepath.exists():
            raise FileNotFoundError(f"Reward file not found: {self.reward_filepath}")

        if self.reward_filepath.suffix != ".py":
            raise ValueError(f"Expected a .py file, got: {self.reward_filepath.suffix}")

        self.module = self._import_module()

        if not hasattr(self.module, "_reward"):
            raise AttributeError(
                f"Reward module '{self.name}' must define a '_reward' function."
            )

    def _import_module(self):
        spec = importlib.util.spec_from_file_location(self.name, self.reward_filepath)
        if spec is None:
            raise ImportError(f"Could not load spec from: {self.reward_filepath}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def call_reward(self, bb):
        return self.module._reward(bb)

    def get_name(self):
        return self.name