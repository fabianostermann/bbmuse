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

class SculptingSession:
    def __init__(self, project, module_manager, device=torch.device("cpu")):
        self.project = project
        self.blackboard = self.project.get_blackboard()
        self.module_manager = module_manager
        self.device = device

    def run(self, args):
        self.module_handler = self.module_manager.identify_module(args.module[0])
        if not self.module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

        # load clone from disk
        clone_dirs = self.module_manager.get_available_clone_run_dirs(self.module_handler)
        clone_final_path = self.module_manager.get_final_model_path(clone_dirs[-1])
        clone_model = Checkpoint(clone_final_path, self.device).load().make_model()
        logger.info("Loaded model from: %s", clone_final_path)

        # run training (TODO: allow use of a config file for hyperparameters)
        self.train(clone_model)

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

    def train(self,
        clone_model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 1e-3,
        fallback_loss_function = F.mse_loss,
        checkpoint_interval: int = 10,
    ) -> None:
        
        # init run & checkpoint directory
        curr_run_dir = self.module_manager.create_next_sculpt_run_dir(self.module_handler)

        loss_functions = self.load_loss_functions(self.module_handler, fallback_loss_function)

        clone_model.to(self.device)
        clone_model.train()
        optimizer = torch.optim.Adam(clone_model.parameters(), lr=lr)

        logger.info("Starting training for %s epochs.", epochs)
        with tqdm(range(epochs+1)) as pbar:
            for epoch in pbar:
                loss = 0.0

                if epoch > 0:
                    optimizer.zero_grad()

                    # TODO: Put PPO (RL with constraints and RLHF) here
                    #preds = clone_model(inputs)

                    #for name, target in targets.items():
                    #    loss = loss + loss_functions[name](preds[name], target)

                    #loss.backward()
                    optimizer.step()

                    desc = f"epoch={epoch:04d} loss={loss:.6f}"
                    pbar.set_description(desc)
                    logger.debug(desc)

                # save checkpoints every 10 epochs (default)
                if epoch % checkpoint_interval == 0:
                    ckpt_path = self.module_manager.get_checkpoint_path(curr_run_dir, epoch)
                    ckpt = Checkpoint(ckpt_path)
                    ckpt.save(clone_model, epoch, loss, optimizer)

        final_path = self.module_manager.get_final_model_path(curr_run_dir)
        pt = Checkpoint(final_path)
        pt.save(clone_model, epoch, loss, optimizer)
        