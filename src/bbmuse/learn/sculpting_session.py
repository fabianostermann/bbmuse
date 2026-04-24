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

from bbmuse.engine.project import BbMuseProject

from bbmuse.learn.module_clone import ModuleClone
from bbmuse.learn.checkpoint import Checkpoint
from bbmuse.learn.policy_prober import PolicyProber

class SculptingSession:
    def __init__(self, project: BbMuseProject, module_manager, device=torch.device("cpu")):
        self.project = project
        self.blackboard = self.project.get_blackboard()
        self.module_manager = module_manager
        self.device = device

    def build(self, args):
        self.module_handler = self.module_manager.identify_module(args.module[0])
        if not self.module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

        self.dry_run = args.dry_run

        # load clone from disk
        clone_dirs = self.module_manager.get_available_clone_run_dirs(self.module_handler)
        clone_final_path = self.module_manager.get_final_model_path(clone_dirs[-1])
        self.clone_model = Checkpoint(clone_final_path, self.device).load().make_model()
        logger.info("Loaded model from: %s", clone_final_path)

        # make module prober
        self.prober = PolicyProber(self.clone_model, self.module_handler, self.project.get_blackboard())
        self.prober.activate_listen()


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
        num_updates: int = 3, # 100,
        epochs: int = 10,
        lr: float = 1e-3,
        fallback_loss_function = F.l1_loss, # mean absolute error
        checkpoint_interval: int = 10,
    ) -> None:
        
        # init run & checkpoint directory
        if not self.dry_run:
            curr_run_dir = self.module_manager.create_next_sculpt_run_dir(self.module_handler)

        loss_functions = self.load_loss_functions(self.module_handler, fallback_loss_function)

        self.clone_model.to(self.device)
        optimizer = torch.optim.Adam(self.clone_model.parameters(), lr=lr)

        with tqdm(range(num_updates+1)) as pbar:
            for num_updates in pbar:
                loss = 0.0

                if num_updates > 0:

                    # collect trajectories with old policy
                    trajectories = self.collect(self.clone_model, self.project, self.prober)
                    advantages = self.compute_advantages(trajectories)

                    self.clone_model.train()
                    optimizer.zero_grad()

                    # TODO: make the following real code
                    # Multiple gradient updates on the same data
                    for epoch in range(epochs):
                        for minibatch in trajectories:
                            r = π_θ(a|s) / π_θ_alt(a|s) # Importance Ratio
                            clipped = clip(r, 1-ε, 1+ε)
                            
                            loss = -mean(min(r * A, clipped * A)) # PPO clip loss
                            
                            loss.backward()
                            optimizer.step()

                    # Update old policy
                    π_θ_alt = π_θ
                    
                    loss.backward()
                    optimizer.step()

                    desc = f"num_updates={num_updates:04d} loss={loss:.6f}"
                    pbar.set_description(desc)
                    logger.debug(desc)

                # save policy checkpoints every 10 num_updates (default)
                if not self.dry_run and num_updates % checkpoint_interval == 0:
                    ckpt_path = self.module_manager.get_checkpoint_path(curr_run_dir, num_updates)
                    ckpt = Checkpoint(ckpt_path)
                    ckpt.save(self.clone_model, num_updates, loss, optimizer)

        if not self.dry_run:
            final_path = self.module_manager.get_final_model_path(curr_run_dir)
            pt = Checkpoint(final_path)
            pt.save(self.clone_model, num_updates, loss, optimizer)
        
    def collect(self, policy_model, env: BbMuseProject, prober: PolicyProber):
        # run policy -> collect episodes
        policy_model.eval() # deactivate dropout, BatchNorm etc.
        with torch.no_grad():
            env.run(quit_after=0.1, run_mode=0)

        trajectories = prober.flush()
        return trajectories

    def compute_advantages(self, trajectories, discount_factor = 0.99, gae_lambda = 0.95):
        # TODO (1) Baseline (noisy): A = G, advantage as pure return value
        # TODO (2) PPO standard: Critic + GAE
        raise NotImplementedError()