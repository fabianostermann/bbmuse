import logging

import importlib.util
from pathlib import Path
import shutil

import torch

from bbmuse.engine.blackboard import _BlackboardView

logger = logging.getLogger(__name__)

class RewardCollector:
    """Evaluates all rewards once per timestep, after every module has written."""

    def __init__(self, project, reward_fpaths, log_path=None, device=torch.device("cpu")):
        self.project = project
        self.device = device

        self.buffer = {}          # reward_name -> list of scalars
        self._active = False

        # load reward functions from disk
        self.rewards = []
        for path in reward_fpaths:
            try:
                reward = Reward(path)
                self.rewards.append(reward)
                if log_path: # copy reward files for logging purposes
                    destination = Path(log_path) / "rewards" / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
            except Exception:
                logger.exception("Ignored reward at: %s", path)
        if not self.rewards:
            logger.warning("No reward functions in place.")

        # create representation-complete blackboard view
        blackboard = self.project.get_blackboard()
        self.bb_view = _BlackboardView(blackboard, readable_keys=blackboard.list_content())

        # determine last_handler from project
        exec_order = self.project.get_controller().execution_order
        last_handler = exec_order[-1]
        logger.debug("Activating reward collection on mod_handler %s. Exec order is: %s", last_handler, exec_order)
        self._activate(last_handler)

    def _activate(self, last_handler):

        self._original = last_handler.call_update
        original, collector = self._original, self

        def wrapped(bb):
            result = original(bb)
            collector._collect()   # after the final module of the cycle
            return result

        last_handler.call_update = wrapped
        self._handler = last_handler
        self._active = True

    def _collect(self):
        for reward in self.rewards:
            name = reward.get_name()
            self.buffer.setdefault(name, []).append(reward.call_reward(self.bb_view))

    def flush(self):
        # Rewards are scalars -> convert to a 1D torch tensor per rep
        out = {
            f"rewards__{k}": torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in self.buffer.items()
        }
        self.buffer.clear()
        return out

    def override_weights(self, weight_dict: dict):
        for reward in self.rewards:
            if reward.name in weight_dict.keys():
                reward._weight = weight_dict[reward.name]


class Reward:
    def __init__(self, reward_filepath: str | Path):
        self.reward_filepath = Path(reward_filepath).resolve()
        self.name = self.reward_filepath.stem

        if not self.reward_filepath.exists():
            raise FileNotFoundError(f"Reward file not found: {self.reward_filepath}")

        if self.reward_filepath.suffix != ".py":
            raise ValueError(f"Expected a .py file, got: {self.reward_filepath.suffix}")

        self.module = self._import_module()

        if not hasattr(self.module, "_reward") or not callable(getattr(self.module, "_reward")):
            raise AttributeError(
                f"Reward module '{self.name}' must define a '_reward' function."
            )

        self._weight = float(getattr(self.module, "_weight", 1.0))

    def _import_module(self):
        spec = importlib.util.spec_from_file_location(self.name, self.reward_filepath)
        if spec is None:
            raise ImportError(f"Could not load spec from: {self.reward_filepath}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def call_reward(self, bb):
        return self.module._reward(bb)

    def get_weight(self):
        return self._weight

    def get_name(self):
        return self.name