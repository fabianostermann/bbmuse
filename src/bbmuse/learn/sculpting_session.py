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
from bbmuse.learn.policy_model import PolicyModel

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
        clone_model = Checkpoint(clone_final_path, self.device).load().make_model()
        self.policy_model = PolicyModel(clone_model)
        logger.info("Loaded model from: %s", clone_final_path)

        # make module prober
        self.prober = PolicyProber(self.policy_model, self.module_handler, self.project.get_blackboard())
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
        checkpoint_interval: int = None,
    ) -> None:
        
        # init run & checkpoint directory
        if not self.dry_run:
            curr_run_dir = self.module_manager.create_next_sculpt_run_dir(self.module_handler)

        if checkpoint_interval is None:
            checkpoint_interval = epochs // 10 # default: 10 checkpoints per training run

        loss_functions = self.load_loss_functions(self.module_handler, fallback_loss_function)

        self.policy_model.to(self.device)
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)

        with tqdm(range(num_updates+1)) as pbar:
            for num_updates in pbar:

                if num_updates > 0:

                    # collect trajectories with current policy
                    trajectories = self.collect(self.policy_model, self.project, self.prober)
                    advantages = self.compute_advantages(trajectories)

                    # extract what we need once, outside the loop
                    states = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('requires__') or k.startswith('uses__')}
                    old_log_probs = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('log_probs__')}
                    actions = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('actions__')}

                    self.policy_model.train()

                    for epoch in range(epochs):
                        # recompute log probs of OLD actions under CURRENT policy
                        new_log_probs = self.policy_model.log_prob(states, actions)

                        total_loss = 0.0
                        for head_name in new_log_probs.keys():
                            A = advantages[f'{head_name}']
                            old_lp = old_log_probs[head_name]
                            new_lp = new_log_probs[head_name]

                            # importance ratio in log space for numerical stability

                            r = torch.exp(new_lp - old_lp)

                            # PPO clip loss
                            eps = 0.2
                            clipped = torch.clamp(r, 1 - eps, 1 + eps)
                            loss = -torch.mean(torch.min(r * A, clipped * A))
                            total_loss = total_loss + loss  # accumulate across heads

                        optimizer.zero_grad()
                        total_loss.backward()  # one backward through the full shared graph
                        optimizer.step()

                    desc = f"num_updates={num_updates:04d} total_loss={loss:.6f}"
                    pbar.set_description(desc)
                    logger.debug(desc)

                # save policy checkpoints every 10 num_updates (default)
                if not self.dry_run and num_updates % checkpoint_interval == 0:
                    raise NotImplementedError("checkpoints do not handle PolicyModel objects yet. Make it so")
                    ckpt_path = self.module_manager.get_checkpoint_path(curr_run_dir, num_updates)
                    ckpt = Checkpoint(ckpt_path)
                    ckpt.save(self.policy_model, num_updates, loss, optimizer)

        if not self.dry_run:
            final_path = self.module_manager.get_final_model_path(curr_run_dir)
            pt = Checkpoint(final_path)
            pt.save(self.policy_model, num_updates, loss, optimizer)

            # TODO: track loss and other measures/parameters and save to disk
        
    def collect(self, policy_model, env: BbMuseProject, prober: PolicyProber):
        # run policy -> collect episodes
        policy_model.eval() # deactivate dropout, BatchNorm etc.
        with torch.no_grad():
            env.run(quit_after=0.1, run_mode=0)

        trajectories = prober.flush()
        return trajectories

    def compute_advantages(self, trajectories, discount_factor = 0.99, gae_lambda = 0.95):
        # TODO extend advantage calculation to PPO standard: Critic with loss + GAE
        advantages = {}

        reward_keys = [k for k in trajectories.keys() if k.startswith('rewards__')]

        for reward_key in reward_keys:
            head = reward_key.split('__')[1]  # e.g. 'ProvRep'
            rewards = trajectories[reward_key]  # shape: (T,)

            T = len(rewards)
            returns = torch.zeros(T, device=rewards.device)

            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + discount_factor * G
                returns[t] = G

            advantages[f'{head}'] = returns

        return advantages