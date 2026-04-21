import logging
import sys, os

from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

logger = logging.getLogger(__name__)

from bbmuse.learn.module_clone import ModuleClone
from bbmuse.learn.checkpoint import Checkpoint

class CloningSession:
    def __init__(self, project, module_manager):
        self.project = project
        self.blackboard = self.project.get_blackboard()
        self.module_manager = module_manager

        self.fallback_loss_function = F.mse_loss

    def run(self, args):
        self.module_handler = self.module_manager.identify_module(args.module[0])
        if not self.module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

        # load packed representations from recorded episodes
        ep_paths = self.module_manager.get_available_episode_paths(self.module_handler)
        ep_path = ep_paths[-1] # TODO: load all episodes, just loading last episode for now
        episode = self.load_episode(ep_path)

        # check if episode matches the given module handler
        assert self.module_handler.get_requires() == list(episode["requires"].keys()),\
            f"Requires in module handler and loaded episode does not match, got: {self.module_handler.get_requires()} and {list(episode["requires"].keys())}"
        assert self.module_handler.get_uses() == list(episode["uses"].keys()),\
            f"Uses in module handler and loaded episode does not match, got: {self.module_handler.get_uses()} and {list(episode["uses"].keys())}"
        assert self.module_handler.get_provides() == list(episode["provides"].keys()),\
            f"Provides in module handler and loaded episode does not match, got: {self.module_handler.get_provides()} and {list(episode["provides"].keys())}"

        logger.info("Loaded episode from: %s", ep_path)
        shapes = {
            group_name: { rep_name: arr.shape for rep_name, arr in group.items() }
            for group_name, group in episode.items()
        }
        logger.debug("Shapes are: %s", shapes)
        assert len({
            arr.shape[0]
            for group in episode.values()
            for arr in group.values()
        }) == 1, "Inconsistent timestep counts across episode arrays"

        # init network that will be used for behavior cloning
        input_dims_dict = {k: v[1:] for k, v in (shapes["uses"] | shapes["requires"]).items()}
        output_dims_dict = {k: v[1:] for k, v in shapes["provides"].items()}
        path_to_backbone = self.get_path_to_backbone(args.backbone)
        clone_model = ModuleClone(input_dims_dict, output_dims_dict, path_to_backbone)

        loss_functions = self.load_loss_functions(self.module_handler)

        # run training (TODO: allow use of a config file for hyperparameters)
        self.train(clone_model, episode, loss_functions)

    def load_episode(self, ep_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
        episode = {
            "requires": {},
            "uses": {},
            "provides": {},
        }

        with np.load(ep_path) as data:
            for key in data.files:
                if key.startswith("requires__"):
                    episode["requires"][key[len("requires__"):]] = data[key]
                elif key.startswith("uses__"):
                    episode["uses"][key[len("uses__"):]] = data[key]
                elif key.startswith("provides__"):
                    episode["provides"][key[len("provides__"):]] = data[key]
                else:
                    raise ValueError(f"Unexpected key in episode archive: {key}")

        return episode

    def get_path_to_backbone(self, backbone_name: str | None):
        if backbone_name is None: 
            return None
        ptb = self.module_manager.get_backbones_dir() / (backbone_name+".py")
        if ptb.exists() and ptb.is_file():
            logger.debug("Found backbone file: %s", ptb)
            return ptb
        else:
            raise FileNotFoundError(f"Backbone file not found: {ptb}")

    def load_loss_functions(self, mod_handler):
        logger.info("Load loss functions for target representations of module %s", mod_handler)
        loss_functions = {}
        for provided_rep_name in mod_handler.get_provides():
            rh = self.blackboard.get(provided_rep_name)
            loss_candidate = getattr(rh.get_component(), "_loss", None)
            if loss_candidate and callable(loss_candidate):
                logger.debug("Found custom loss function for %s.", rh)
                loss_functions[provided_rep_name] = loss_candidate
            else:
                logger.debug("No custom loss function found for %s. Using fallback (%s)", rh, self.fallback_loss_function)
                loss_functions[provided_rep_name] = self.fallback_loss_function

        return loss_functions

    def train(self,
        clone_model: torch.nn.Module,
        episode: Dict[str, Dict[str, object]],
        loss_functions: Dict[str, callable],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        
        # init run & checkpoint directory
        curr_run_dir = self.module_manager.create_next_clone_run_dir(self.module_handler)

        clone_model.train()
        optimizer = torch.optim.Adam(clone_model.parameters(), lr=lr)

        input_arrays = episode["uses"] | episode["requires"]
        target_arrays = episode["provides"]

        inputs = {
            name: torch.as_tensor(arr, dtype=torch.float32)
            for name, arr in input_arrays.items()
        }
        targets = {
            name: torch.as_tensor(arr, dtype=torch.float32)
            for name, arr in target_arrays.items()
        }

        logger.info("Starting training for %s epochs.", epochs)
        with tqdm(range(epochs+1)) as pbar:
            for epoch in pbar:
                loss = 0.0

                if epoch > 0:
                    optimizer.zero_grad()
                    preds = clone_model(inputs)

                    for name, target in targets.items():
                        loss = loss + loss_functions[name](preds[name], target)

                    loss.backward()
                    optimizer.step()

                    pbar.set_description(f"epoch={epoch:04d} loss={loss:.6f}")

                # save checkpoints every 10 epochs
                if epoch % 10 == 0:
                    ckpt_path = self.module_manager.get_checkpoint_path(curr_run_dir, epoch)
                    ckpt = Checkpoint(ckpt_path)
                    ckpt.save(clone_model, epoch, loss, optimizer)

        final_path = self.module_manager.get_final_model_path(curr_run_dir)
        pt = Checkpoint(final_path)
        pt.save(clone_model, epoch, loss, optimizer)
        