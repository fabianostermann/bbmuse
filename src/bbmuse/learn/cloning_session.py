import logging
import sys, os

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

logger = logging.getLogger(__name__)

from bbmuse.learn.module_clone import ModuleClone

class CloningSession:
    def __init__(self, project, module_manager):
        self.project = project
        self.blackboard = self.project.get_blackboard()
        self.module_manager = module_manager

        # TODO: load backbone library from py file (.bblearn/backbones/)

    def run(self, args):
        module_handler = self.module_manager.identify_module(args.module[0])
        if not module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

        # load packed representations from recorded episodes
        ep_paths = self.module_manager.get_available_episode_paths(module_handler)
        ep_path = ep_paths[-1] # just last episode for now
        episode = load_episode(ep_path)

        # check if episode matches the given module handler
        assert module_handler.get_requires() == list(episode["requires"].keys()),\
            f"Requires in module handler and episode does not match, got: {module_handler.get_requires()} and {list(episode["requires"].keys())}"
        assert module_handler.get_uses() == list(episode["uses"].keys()),\
            f"Uses in module handler and episode does not match, got: {module_handler.get_uses()} and {list(episode["uses"].keys())}"
        assert module_handler.get_provides() == list(episode["provides"].keys()),\
            f"Provides in module handler and episode does not match, got: {module_handler.get_provides()} and {list(episode["provides"].keys())}"

        logger.debug("Loaded episode from: %s", ep_path)
        shapes = {
            group_name: { rep_name: arr.shape for rep_name, arr in group.items() }
            for group_name, group in episode.items()
        }
        logger.debug("Shapes are: %s", shapes)
        # TODO: assert num of timesteps are equal

        # TODO: init network that will be used for behavior cloning
        #ModuleClone(input_dims, output_dims) # determine input_dims and output_dims from shapes (expected form: Dict[str, int])

        # TODO: run training (TODO: allow use of a config file for hyperparameters)

def load_episode(ep_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    grouped = {
        "requires": {},
        "uses": {},
        "provides": {},
    }

    with np.load(ep_path) as data:
        for key in data.files:
            if key.startswith("requires__"):
                grouped["requires"][key[len("requires__"):]] = data[key]
            elif key.startswith("uses__"):
                grouped["uses"][key[len("uses__"):]] = data[key]
            elif key.startswith("provides__"):
                grouped["provides"][key[len("provides__"):]] = data[key]
            else:
                raise ValueError(f"Unexpected key in episode archive: {key}")

    return grouped