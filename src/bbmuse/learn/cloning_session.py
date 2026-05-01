import logging
import sys, os

from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from typing import Dict

from time import time

logger = logging.getLogger(__name__)

from bbmuse.learn.module_clone import ModuleClone
from bbmuse.learn.checkpoint import Checkpoint
from bbmuse.learn.session_logger import SessionLogger

class CloningSession:
    def __init__(self, project, module_manager, device=torch.device("cpu")):
        self.project = project
        self.blackboard = self.project.get_blackboard()
        self.module_manager = module_manager
        self.device = device

    def build(self, args):
        self.module_handler = self.module_manager.identify_module(args.module[0])
        if not self.module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

        self.tag = args.tag
        self.dry_run = args.dry_run

        # load packed representations from recorded episodes
        ep_paths = self.module_manager.get_available_episode_paths(self.module_handler)
        ep_path = ep_paths[-1] # TODO: load all episodes, just loading last episode for now
        self.episode = self.load_episode(ep_path)

        # check if episode matches the given module handler
        assert self.module_handler.get_requires() == list(self.episode["requires"].keys()),\
            f"Requires in module handler and loaded episode does not match, got: {self.module_handler.get_requires()} and {list(self.episode["requires"].keys())}"
        assert self.module_handler.get_uses() == list(self.episode["uses"].keys()),\
            f"Uses in module handler and loaded episode does not match, got: {self.module_handler.get_uses()} and {list(self.episode["uses"].keys())}"
        assert self.module_handler.get_provides() == list(self.episode["provides"].keys()),\
            f"Provides in module handler and loaded episode does not match, got: {self.module_handler.get_provides()} and {list(self.episode["provides"].keys())}"

        logger.info("Loaded episode from: %s", ep_path)
        shapes = {
            group_name: { rep_name: arr.shape for rep_name, arr in group.items() }
            for group_name, group in self.episode.items()
        }
        logger.debug("Shapes are: %s", shapes)
        assert len({
            arr.shape[0]
            for group in self.episode.values()
            for arr in group.values()
        }) == 1, "Inconsistent timestep counts across episode arrays"

        # init network that will be used for behavior cloning
        input_dims_dict = {k: v[1:] for k, v in (shapes["uses"] | shapes["requires"]).items()}
        output_dims_dict = {k: v[1:] for k, v in shapes["provides"].items()}
        path_to_backbone = self.get_path_to_backbone(args.backbone)
        self.clone_model = ModuleClone(input_dims_dict, output_dims_dict, path_to_backbone)

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

    def load_loss_functions(self, mod_handler, fallback_loss_function):
        logger.info("Load loss functions for target representations of module %s", mod_handler)
        loss_functions = {}
        for provided_rep_name in mod_handler.get_provides():
            rh = self.blackboard.get(provided_rep_name)
            loss_candidate = getattr(rh.get_component(), "_loss", None)
            if loss_candidate and callable(loss_candidate):
                logger.debug("Found custom loss function for %s.", rh)
                loss_functions[provided_rep_name] = loss_candidate
            else:
                logger.debug("No custom loss function found for %s. Will fallback to: %s", rh, fallback_loss_function)
                loss_functions[provided_rep_name] = fallback_loss_function

        return loss_functions

    def run(self,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 512,
        fallback_loss_function = F.mse_loss,
        checkpoint_interval: int = None,
    ) -> None:
        
        session_logger = SessionLogger()

        if not self.dry_run:
            curr_run_dir = self.module_manager.create_next_clone_run_dir(self.module_handler, self.tag)

        loss_functions = self.load_loss_functions(self.module_handler, fallback_loss_function)

        self.clone_model.to(self.device)
        self.clone_model.train()
        optimizer = torch.optim.Adam(self.clone_model.parameters(), lr=lr)

        input_arrays = self.episode["uses"] | self.episode["requires"]
        target_arrays = self.episode["provides"]

        inputs = {
            name: torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            for name, arr in input_arrays.items()
        }
        targets = {
            name: torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            for name, arr in target_arrays.items()
        }

        input_keys = list(inputs.keys())
        target_keys = list(targets.keys())
        dataset = TensorDataset(*inputs.values(), *targets.values()) # TODO: do this manually without torch loaders. Copy from sculpt_session
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        start_walltime = time()
        logger.info("Starting training for %s epochs.", epochs)
        with tqdm(range(epochs+1)) as pbar:
            for epoch in pbar:
            
                epoch_loss = 0.0
                DEBUG_ONLY_accuracy = []

                if epoch > 0:

                    for batch in loader:
                        n_inputs = len(input_keys)
                        batch_inputs = dict(zip(input_keys, batch[:n_inputs]))
                        batch_targets = dict(zip(target_keys, batch[n_inputs:]))

                        optimizer.zero_grad()
                        preds = self.clone_model(batch_inputs)
                        loss = 0.0

                        for name, target in batch_targets.items():
                            repr_loss = loss_functions[name](preds[name], target)

                            DEBUG_ONLY_accuracy.append(self._DEBUG_ONLY_accuracy(preds[name], target))

                            session_logger.log({f"loss__{name}": repr_loss})
                            loss = loss + repr_loss

                        loss = loss / len(batch_targets)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    # >>> DEBUG
                    DEBUG_ONLY_accuracy = sum(DEBUG_ONLY_accuracy) / len(DEBUG_ONLY_accuracy)
                    session_logger.log({"accuracy": DEBUG_ONLY_accuracy})
                    # <<< DEBUG

                    epoch_loss /= len(loader)
                    session_logger.log({"epoch": epoch, "loss": epoch_loss, "walltime": time()-start_walltime}).step()
                    pbar.set_description(f"epoch={epoch:04d} loss={epoch_loss:.6f}")
                
                # save checkpoints
                if not self.dry_run:
                    if checkpoint_interval and epochs % checkpoint_interval == 0:
                        ckpt_path = self.module_manager.get_checkpoint_path(curr_run_dir, epoch)
                        ckpt = Checkpoint(ckpt_path)
                        ckpt.save(self.clone_model, epoch, epoch_loss, optimizer)
                    session_logger.write_to_disk(curr_run_dir)

        if not self.dry_run:
            final_path = self.module_manager.get_final_model_path(curr_run_dir)
            pt = Checkpoint(final_path)
            pt.save(self.clone_model, epoch, epoch_loss, optimizer)
            session_logger.write_to_disk(curr_run_dir)
        

    def _DEBUG_ONLY_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pc_correct  = pred[:, :12].argmax(-1) == target[:, :12].argmax(-1)
        oct_correct = pred[:, 12:].argmax(-1) == target[:, 12:].argmax(-1)
        return (pc_correct & oct_correct).float().mean()