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
from bbmuse.learn.policy_model import PolicyModel
from bbmuse.learn.session_logger import SessionLogger
from bbmuse.learn.reward import Reward

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

        self.tag = args.tag
        self.dry_run = args.dry_run

        # load clone from disk
        clone_dirs = self.module_manager.get_available_clone_run_dirs(self.module_handler)
        clone_final_path = self.module_manager.get_final_model_path(clone_dirs[-1])
        clone_model = Checkpoint(clone_final_path, self.device).load().make_model()
        self.policy_model = PolicyModel(clone_model)
        logger.info("Loaded model from: %s", clone_final_path)

        reward_fpaths = self.module_manager.get_available_rewards_filepaths()
        self.rewards = []
        for path in reward_fpaths:
            try:
                reward = Reward(path)
                self.rewards.append(reward)
            except Exception:
                logger.exception("Ignored reward at: %s", path)

        # make module prober
        self.prober = PolicyProber(self.policy_model, self.module_handler, self.project.get_blackboard(), self.rewards)
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
        num_updates: int = 50,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 256,
        entropy_coef = 0.0,
        bc_coef = 1.0,
        fallback_loss_function = F.mse_loss,
        checkpoint_interval: int = None,
    ) -> None:
        
        session_logger = SessionLogger()
        
        # init run & checkpoint directory
        if not self.dry_run:
            curr_run_dir = self.module_manager.create_next_sculpt_run_dir(self.module_handler, self.tag)

        loss_functions = self.load_loss_functions(self.module_handler, fallback_loss_function)

        self.policy_model.to(self.device)
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)

        with tqdm(range(num_updates+1)) as pbar:
            start_walltime = time()
            for num_updates in pbar:

                if num_updates > 0:
                    logger.debug("Start collecting trajectories (exploration phase)..")

                    # collect trajectories with current policy
                    trajectories = self.collect(self.policy_model, self.project, self.prober)
                    advantages, mean_reward = self.compute_advantages(trajectories)

                    session_logger.log({"mean_reward": mean_reward})

                    # extract what we need once, outside the loop
                    states = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('requires__') or k.startswith('uses__')}
                    old_log_probs = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('log_probs__')}
                    actions = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('actions__')}
                    
                    # and the actions that the original module would have chosen (used for BC)
                    oracled_actions = {k.split('__')[1]: v for k, v in trajectories.items() if k.startswith('provides__')}

                    self.policy_model.train()

                    logger.debug("Train policy model (learning phase)..")

                    T = next(iter(states.values())).shape[0]
                    for epoch in range(epochs):
                        indices = torch.randperm(T, device=self.device)

                        epoch_loss = 0.0
                        epoch_entropy = []
                        epoch_bc_loss = []

                        for batch_start in range(0, T, batch_size):
                            idx = indices[batch_start:batch_start+batch_size]

                            batch_states     = {k: v[idx] for k, v in states.items()}
                            batch_old_lp     = {k: v[idx] for k, v in old_log_probs.items()}
                            batch_actions    = {k: v[idx] for k, v in actions.items()}
                            batch_oracle     = {k: v[idx] for k, v in oracled_actions.items()}
                            batch_advantages = {k: v[idx] for k, v in advantages.items()}

                            # recompute log probs of OLD actions under CURRENT policy
                            new_log_probs, entropies = self.policy_model.log_prob_with_entropy(batch_states, batch_actions)
                            pred_actions = self.policy_model(batch_states) # TODO remove duplicate forward call

                            batch_loss = 0.0
                            batch_entropy = 0.0

                            for head_name in new_log_probs.keys():
                                A = batch_advantages[head_name] # TODO advantages are for all heads!
                                old_lp = batch_old_lp[head_name]
                                new_lp = new_log_probs[head_name]

                                # importance ratio in log space for numerical stability
                                r = torch.exp(new_lp - old_lp)

                                # PPO clip loss
                                eps = 0.2
                                clipped = torch.clamp(r, 1 - eps, 1 + eps)
                                policy_loss = -torch.mean(torch.min(r * A, clipped * A))

                                # entropy loss
                                entropy = torch.mean(entropies[head_name])  # negative because we want to maximize entropy
                                epoch_entropy.append(entropy)
                                
                                # BC loss
                                bc_pred = pred_actions[head_name]           # what policy did
                                bc_target = batch_oracle[head_name] # what original module did
                                bc_loss = loss_functions[head_name](bc_pred, bc_target)
                                epoch_bc_loss.append(bc_loss)
                                
                                loss_contribution = sum([
                                    policy_loss,
                                    entropy_coef * -entropy,
                                    bc_coef * bc_loss,
                                ]) / len(new_log_probs) # important because decouples task count from hyperparameter tuning

                                batch_loss += loss_contribution

                            optimizer.zero_grad()
                            batch_loss.backward()  # one backward through the full shared graph
                            optimizer.step()

                            epoch_loss += batch_loss / len(indices)

                    session_logger.log({
                        "num_updates": num_updates,
                        "last_epoch_loss": epoch_loss,
                        "entropy": sum(epoch_entropy)/len(epoch_entropy),
                        "bc_loss": sum(epoch_bc_loss)/len(epoch_bc_loss),
                        "walltime": time()-start_walltime,
                    }).step()

                    desc = f"num_updates={num_updates:04d} loss={epoch_loss:.6f}"
                    pbar.set_description(desc)

                # save policy checkpoints every 10 num_updates (default)
                if not self.dry_run:
                    if checkpoint_interval and num_updates % checkpoint_interval == 0:
                        pass
                        # TODO checkpoints do not handle PolicyModel objects yet. Make it so!
                        #ckpt_path = self.module_manager.get_checkpoint_path(curr_run_dir, num_updates)
                        #ckpt = Checkpoint(ckpt_path)
                        #ckpt.save(self.policy_model, num_updates, loss, optimizer)
                    session_logger.write_to_disk(curr_run_dir)

        if not self.dry_run:
            #final_path = self.module_manager.get_final_model_path(curr_run_dir)
            #pt = Checkpoint(final_path)
            #pt.save(self.policy_model, num_updates, loss, optimizer)
            session_logger.write_to_disk(curr_run_dir)
        
    def collect(self, policy_model, env: BbMuseProject, prober: PolicyProber):
        # run policy -> collect episodes
        policy_model.eval() # deactivate dropout, BatchNorm etc.
        with torch.no_grad():
            env.run(quit_after=2, run_mode=0)

        trajectories = prober.flush()
        return trajectories

    def compute_advantages(self, trajectories, discount_factor = 0.99, gae_lambda = 0.95):
        # TODO extend advantage calculation to PPO standard: Critic with loss + GAE
        # TODO: truncated GAE (we have endless episodes)
        advantages = {}

        reward_keys = [k for k in trajectories.keys() if k.startswith('rewards__')]
        mean_rewards = []

        for reward_key in reward_keys:
            head = reward_key.split('__')[1]  # e.g. 'ProvRep'
            rewards = trajectories[reward_key]  # shape: (T,)

            mean_rewards.append(rewards.mean())

            T = len(rewards)
            returns = torch.zeros(T, device=rewards.device)

            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + discount_factor * G
                returns[t] = G

            returns = (returns - returns.mean()) / (returns.std() + 1e-8) # normalize returns
            advantages[f'{head}'] = returns

        mean_reward = sum(mean_rewards) / len(reward_keys)
        return advantages, mean_reward
