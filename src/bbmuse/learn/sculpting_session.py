import logging
import sys, os

from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from time import time

logger = logging.getLogger(__name__)

from bbmuse.engine.project import BbMuseProject

from bbmuse.learn.module_clone import ModuleClone
from bbmuse.learn.checkpoint import Checkpoint
from bbmuse.learn.policy_prober import PolicyProber
from bbmuse.learn.reward_collector import RewardCollector
from bbmuse.learn.policy_model import PolicyModel
from bbmuse.learn.session_logger import SessionLogger
from bbmuse.learn.action_spaces import make_ce_loss
from bbmuse.learn.metrics import estimate_entropy_floor

class SculptingSession:
    def __init__(self, project: BbMuseProject, module_manager, device=torch.device("cpu")):
        self.project = project
        self.module_manager = module_manager
        self.device = device

    def build(self, args, skip_reward_collector=False):
        self.module_handler = self.module_manager.identify_module(args.module[0])
        if not self.module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

        self.tag = args.tag
        self.dry_run = args.dry_run

        self.curr_run_dir = None
        if not self.dry_run:
            self.curr_run_dir = self.module_manager.create_next_sculpt_run_dir(self.module_handler, self.tag)
        self.session_logger = SessionLogger(self.curr_run_dir)

        # load clone from disk -- TODO: create mode that runs without BC model (init just a random model)
        clone_dirs = self.module_manager.get_available_clone_run_dirs(self.module_handler)
        clone_final_path = self.module_manager.get_final_model_path(clone_dirs[-1])
        self.loaded_checkpoint = Checkpoint(clone_final_path, self.device).load()
        clone_model = self.loaded_checkpoint.make_model()
        self.policy_model = PolicyModel(clone_model)
        self.policy_model.to(self.device)
        logger.info("Loaded model from: %s", clone_final_path)

        # make module prober
        self.prober = PolicyProber(self.policy_model, self.module_handler, self.project.get_blackboard())
        self.prober.activate_listen()

        # make reward collector
        if skip_reward_collector: # useful to avoid duplicated monkey patches
            self.reward_collector = None
        else:
            reward_fpaths = self.module_manager.get_available_rewards_filepaths()
            self.reward_collector = RewardCollector(self.project, reward_fpaths, log_path=self.curr_run_dir, device=self.device)

    def run(self,
        num_updates: int = 100,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 256,
        entropy_coef = 0.0,
        bc_coef = 0.1,
        checkpoint_interval: int = 10,
        log_context = {},
        log_global_offset = 0,
    ) -> None:
        kwargs = {k: v for k, v in locals().items() if k != 'self'}

        clone_info = {"clone_info": {"clone_epochs": self.loaded_checkpoint.get_epoch(), "clone_loss": self.loaded_checkpoint.get_loss()}}
        self.session_logger.write_config_to_disk( { "run_kwargs": kwargs } | clone_info, overwrite_dir=self.curr_run_dir)

        loss_functions = {name: make_ce_loss(nvec)
            for name, nvec in self.policy_model.model.config["action_spaces"].items()}

        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)

        epoch_loss = 0.0
        with tqdm(range(num_updates+1)) as pbar:
            for num_updates in pbar:

                if num_updates > 0:
                    logger.debug("Start collecting trajectories (exploration phase)..")

                    # collect trajectories with current policy
                    trajectories = self.collect(self.policy_model, self.project, self.prober, self.reward_collector)
                    advantages, named_returns = self.compute_advantages(trajectories)
                    mean_returns = {f"rew_{name}": v.mean().item() for name, v in named_returns.items()}

                    self.session_logger.log(mean_returns)

                    # extract what we need once, outside the loop
                    states = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('requires__') or k.startswith('uses__')}
                    old_log_probs = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('log_probs__')}
                    actions = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('actions__')}
                    
                    # and the actions that the original module would have chosen (used for BC)
                    oracled_actions = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('provides__')}

                    floors, _, mean_group = estimate_entropy_floor(
                        {k: v.detach().cpu().numpy() for k, v in states.items()},
                        {k: v.detach().cpu().numpy() for k, v in oracled_actions.items()},
                        self.policy_model.model.config["action_spaces"],
                        miller_madow=True,   # groups are ~1000x smaller than in cloning
                    )
                    floor = sum(floors.values()) / len(floors)

                    self.policy_model.train()

                    logger.debug("Train policy model (learning phase)..")

                    T = next(iter(states.values())).shape[0]
                    for epoch in range(epochs):
                        indices = torch.randperm(T, device=self.device)

                        epoch_loss = 0.0
                        n_batches = 0
                        epoch_policy_loss = []
                        epoch_entropy = []
                        epoch_bc_loss = []

                        for batch_start in range(0, T, batch_size):
                            idx = indices[batch_start:batch_start+batch_size]

                            batch_states     = {k: v[idx] for k, v in states.items()}
                            batch_old_lp     = {k: v[idx] for k, v in old_log_probs.items()}
                            batch_actions    = {k: v[idx] for k, v in actions.items()}
                            batch_oracle     = {k: v[idx] for k, v in oracled_actions.items()}
                            batch_advantages = advantages[idx] if not advantages is None else None

                            # recompute log probs of OLD actions under CURRENT policy
                            new_log_probs, entropies = self.policy_model.log_prob_with_entropy(batch_states, batch_actions)
                            pred_actions = self.policy_model(batch_states) # TODO remove duplicate forward call

                            batch_loss = 0.0

                            for head_name in new_log_probs.keys():

                                policy_loss = torch.zeros((), device=self.device)
                                if not batch_advantages is None:
                                    A = batch_advantages
                                    old_lp = batch_old_lp[head_name]
                                    new_lp = new_log_probs[head_name]

                                    # importance ratio in log space for numerical stability
                                    r = torch.exp(new_lp - old_lp)

                                    # PPO clip loss
                                    eps = 0.2
                                    clipped = torch.clamp(r, 1 - eps, 1 + eps)
                                    policy_loss = -torch.mean(torch.min(r * A, clipped * A))
                                epoch_policy_loss.append(policy_loss.item())

                                # entropy loss
                                entropy = torch.mean(entropies[head_name])  # negative because we want to maximize entropy
                                epoch_entropy.append(entropy.item())
                                
                                # BC loss
                                bc_pred = pred_actions[head_name]           # what policy did
                                bc_target = batch_oracle[head_name] # what original module did
                                bc_loss = loss_functions[head_name](bc_pred, bc_target)
                                epoch_bc_loss.append(bc_loss.item() * len(idx) / len(new_log_probs))

                                loss_contribution = sum([
                                    policy_loss,
                                    entropy_coef * -entropy,
                                    bc_coef * bc_loss,
                                ]) / len(new_log_probs) # important because decouples task count from hyperparameter tuning

                                batch_loss += loss_contribution

                            optimizer.zero_grad()
                            batch_loss.backward()  # one backward through the full shared graph
                            optimizer.step()

                            epoch_loss += batch_loss.item()
                            n_batches += 1

                    epoch_loss /= n_batches

                    bc_mean = sum(epoch_bc_loss) / T # devided by T because weighted by len(idx) above -> per-sample mean

                    self.session_logger.log({
                        "num_updates": num_updates,
                        "weighted_loss": epoch_loss,
                        "policy_loss": sum(epoch_policy_loss)/len(epoch_policy_loss),
                        "entropy": sum(epoch_entropy)/len(epoch_entropy),
                        "bc_loss": bc_mean,
                        "entropy_floor": floor,
                        "kl_to_symbolic": bc_mean - floor,
                        "mean_group_size": mean_group,
                        "walltime": time(),
                    } | log_context | {"global_update": num_updates+log_global_offset}
                    ).step()

                    desc = f"num_updates={num_updates:04d} loss={epoch_loss:.6f}"
                    pbar.set_description(desc)

                # save intermediate policy checkpoints
                if not self.dry_run:
                    if checkpoint_interval and num_updates % checkpoint_interval == 0:
                        ckpt_path = self.module_manager.get_checkpoint_path(self.curr_run_dir, num_updates+log_global_offset)
                        ckpt = Checkpoint(ckpt_path)
                        ckpt.save(self.policy_model.model, num_updates, epoch_loss, optimizer)
                    self.session_logger.write_to_disk()

        # save final policy
        if not self.dry_run:
            final_path = self.module_manager.get_final_model_path(self.curr_run_dir)
            pt = Checkpoint(final_path)
            pt.save(self.policy_model.model, num_updates, epoch_loss, optimizer)
            self.session_logger.write_to_disk()
        
    def collect(self, policy_model, env: BbMuseProject, prober: PolicyProber, reward_collector: RewardCollector):
        # run policy -> collect episodes
        policy_model.eval() # deactivate dropout, BatchNorm etc.
        with torch.no_grad():
            env.run(quit_after=8, run_mode=0)

        trajectories = prober.flush()
        trajectories |= reward_collector.flush()

        # prober fires mid-cycle, collector fires at end of execution order:
        # a truncated final cycle desynchronizes them by one sample
        lengths = {k: v.shape[0] for k, v in trajectories.items()}
        T = min(lengths.values())
        if len(set(lengths.values())) > 1:
            logger.warning("Trajectory lengths differ, truncating to %d: %s", T, lengths)
        return {k: v[:T] for k, v in trajectories.items()}

    def compute_advantages(self, trajectories, discount_factor=0.9): # gae_lambda = 0.95):
        # TODO extend advantage calculation to PPO standard: Critic with loss + GAE
        # TODO: truncated GAE (we have endless episodes)
        reward_keys = [k for k in trajectories.keys() if k.startswith('rewards__')]

        # Named raw rewards for external aggregation/logging
        named_returns = {k.split('__')[1]: trajectories[k] for k in reward_keys}

        if not named_returns:
            logger.warning("Received no rewards.")
            return None, {}

        # Average across all rewards into a single signal
        stacked_rewards = torch.stack([trajectories[k] for k in reward_keys], dim=0)
        # Normalize each reward signal over the episode length before combining
        stacked_rewards = (stacked_rewards - stacked_rewards.mean(dim=1, keepdim=True)) #/ (stacked_rewards.std(dim=1, keepdim=True) + 1e-8)
        combined_rewards = stacked_rewards.mean(dim=0)  # shape: (T,)

        # Compute discounted returns
        T = len(combined_rewards)
        returns = torch.zeros(T, device=combined_rewards.device)
        G = 0.0
        for t in reversed(range(T)):
            G = combined_rewards[t] + discount_factor * G
            returns[t] = G

        # Normalize
        returns = (returns - returns.mean()) #/ (returns.std() + 1e-8)

        return returns, named_returns
