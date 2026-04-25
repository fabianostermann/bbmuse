import logging

logger = logging.getLogger(__name__)

import numpy as np
import torch

from bbmuse.engine.module_handler import ModuleHandler
from bbmuse.engine.blackboard import Blackboard

from bbmuse.learn.module_listener import ModuleListener
from bbmuse.learn.policy_model import PolicyModel

class PolicyProber(ModuleListener):

    def __init__(self, policy_model: PolicyModel, mod_handler: ModuleHandler, blackboard: Blackboard):
        super().__init__(mod_handler, blackboard)

        self.policy_model = policy_model
        self.device = next(policy_model.parameters()).device

        self._actions_buffer = {}  # rep_name -> list of arrays
        self._log_probs_buffer = {}  # rep_name -> list of arrays
        self._rewards_buffer = {}  # rep_name -> list of scalars

    def _check_requirements(self):
        super()._check_requirements()

        # check that unpack() exists for provided representations
        for provided_rep_name in self._mod_handler.get_provides():
            rh = self._blackboard.get(provided_rep_name)
            assert self._check_function_exists(rh, "_unpack")
    
    def _before_hook(self):
        super()._before_hook() # packs and stores requires and provides

        # get last requires and uses from buffer
        last_required = {rep_name: torch.as_tensor(rep_array[-1], dtype=torch.float32, device=self.device) for rep_name, rep_array in self._requires_buffer.items()}
        last_used = {rep_name: torch.as_tensor(rep_array[-1], dtype=torch.float32, device=self.device) for rep_name, rep_array in self._uses_buffer.items()}
        last_inputs = last_required | last_used

        # let policy model predict actions
        with torch.no_grad():
            actions, log_probs = self.policy_model.sample_with_log_prob(last_inputs)

        # write model outputs to buffer (predictions for provides)
        for rep_name, action_array in actions.items():
            if rep_name not in self._actions_buffer:
                self._actions_buffer[rep_name] = []
            self._actions_buffer[rep_name].append(action_array)

        # write log_probs to buffer (needed for PPO importance ratio)
        for rep_name, log_prob_array in log_probs.items():
            if rep_name not in self._log_probs_buffer:
                self._log_probs_buffer[rep_name] = []
            self._log_probs_buffer[rep_name].append(log_prob_array)
        
    def _after_hook(self):
        # logger.debug("Running _after_hook() on module %s", self._mod_handler)
        # for provided_rep_name in self._mod_handler.get_provides():
        #     rh = self._blackboard.get(provided_rep_name)
        #     self._store(rh, self._provides_buffer)
        super()._after_hook() # packs and stores provides from original module

        # get last actions from buffer
        last_actions = {rep_name: rep_array[-1] for rep_name, rep_array in self._actions_buffer.items()}
        
        for rep_name in last_actions.keys():

            # unpack actions to original representation
            rep_handler = self._blackboard.get(rep_name)
            rep_handler.get_component()._unpack(last_actions[rep_name])
            
            # collect available rewards (self._check_function_exists(rh, "_reward"))
            if self._check_function_exists(rep_handler, "_reward"):
                reward = rep_handler.get_component()._reward()
            else:
                reward = 0 # default entry, will not contribute to any sum
            if rep_name not in self._rewards_buffer:
                self._rewards_buffer[rep_name] = []
            self._rewards_buffer[rep_name].append(reward)

    def flush(self):
        rep_arrays = super().flush()  # shapes requires__, uses__, and provides__ buffers

        # convert numpy arrays to torch tensors # TODO: bottleneck?
        rep_arrays = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in rep_arrays.items()
        }

        # Actions are already torch tensors in lists -> stack directly
        rep_arrays |= {
            f"actions__{k}": torch.stack(v)
            for k, v in self._actions_buffer.items()
        }

        # Actions are already torch tensors in lists -> stack directly
        rep_arrays |= {
            f"log_probs__{k}": torch.stack(v)
            for k, v in self._log_probs_buffer.items()
        }

        # Rewards are scalars .> convert to a 1D torch tensor per rep
        rep_arrays |= {
            f"rewards__{k}": torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in self._rewards_buffer.items()
        }

        assert len(set([v.shape[0] for v in rep_arrays.values()])) == 1,\
            "Episode lengths do not match."
        self._actions_buffer.clear()
        self._log_probs_buffer.clear()
        self._rewards_buffer.clear()
        return rep_arrays
