import logging

import inspect
import torch
import torch.nn as nn

from pathlib import Path

logger = logging.getLogger(__name__)

from bbmuse.learn.module_clone import ModuleClone

class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = None
        self._model = None
        self._optimizer = None

    def save(self, model: nn.Module, module_handler, epoch: int, loss: float, optimizer: torch.optim.Optimizer = None) -> None:
        self._data = {
            "model_class":      model.__class__.__name__,
            "model_config":     model.config,
            "model_state_dict": model.state_dict(),
            "epoch":            epoch,
            "loss":             loss,
            "bbmuse_module":    module_handler.get_name(),
        }
        if optimizer:
            self._data["optimizer_state_dict"] = optimizer.state_dict()
            self._data["optimizer_class"] = optimizer.__class__.__name__

        # TODO: save list of episode IDs that this was trained on

        # TODO maybe later: Store backbone source for audit purposes
        #if model.config.get("path_to_backbone"):
        #    try:
        #        with open(model.metadata["path_to_backbone"]) as f:
        #            self._data["backbone_source"] = f.read()
        #    except OSError:
        #        pass  # don't fail the save if source is unreadable

        torch.save(self._data, self.path)
        return self

    def load(self):
        if not self.path.exists():
            logger.error("Path does not exists: %s", self.path)
            raise FileNotFoundError(path)

        self._data = torch.load(self.path, weights_only=False)
        return self

    def make_model(self):
        if self._model:
            return self._model

        assert self._data, "Checkpoint must be saved or loaded first."
        assert self._data["model_class"] == ModuleClone.__name__

        model = ModuleClone(**self._data["model_config"])
        model.load_state_dict(self._data["model_state_dict"])

        return model

    def make_optimizer(self):
        if self._optimizer:
            return self._optimizer

        assert self._data, "Checkpoint must be saved or loaded first."

        optimizer = torch.optim.Adam(self.make_model().parameters())

        if "optimizer_state_dict" in self._data.keys():
            if not self._data["optimizer_class"] == torch.optim.Adam.__name__:
                logger.warning("optimizer_class seems to be other than Adam, got: %s", self._data["optimizer_class"])
            optimizer.load_state_dict(self._data["optimizer_state_dict"])
        else:
            logger.info("Optimizer state_dict not available from checkpoint '%s'. Returning blank optimizer.", self.path)

        return optimizer

    def get_epoch(self):
        assert self._data, "Checkpoint must be saved or loaded first."
        return self._data["epoch"]

    def get_loss(self):
        assert self._data, "Checkpoint must be saved or loaded first."
        return self._data["loss"]

    def get_module_name(self):
        assert self._data, "Checkpoint must be saved or loaded first."
        return self._data["bbmuse_module"]