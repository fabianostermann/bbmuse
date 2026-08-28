import logging

import torch

logger = logging.getLogger(__name__)


class RolloutCollector:
    """
    Owns ONE rollout pipeline over N probed agents: runs the environment,
    flushes every prober plus the shared RewardCollector, reconciles
    trajectory lengths, and returns everything AGENT-KEYED.

    Every collect() flushes ALL probers, so frozen agents (IBR) no longer
    accumulate buffers across a phase -- their data is simply returned and
    ignored by the caller. IPPO uses all slices; IBR uses one.

    compute_advantages() also returns an agent-keyed dict. With the current
    shared team reward every agent receives the SAME advantage tensor, but
    the agent-keyed return type is deliberate: per-agent difference rewards
    or a MAPPO critic baseline later only change this method (or its
    caller), not any signature downstream.
    """

    def __init__(self, project, probers: dict, reward_collector, device=torch.device("cpu")):
        """
        probers:          dict agent_name -> PolicyProber (already activated;
                          patch order relative to the RewardCollector is the
                          caller's responsibility: probers first, collector last).
        reward_collector: the single shared RewardCollector.
        """
        self.project = project
        self.probers = dict(probers)
        self.reward_collector = reward_collector
        self.device = device

    # ---------------------------------------------------------------- collect

    def collect(self, rollout_cycles: float = 2000):
        """
        Run one rollout and return (per_agent, rewards):

          per_agent[name] = {
              "states":        rep -> [T, ...]  (requires + uses, packed floats)
              "actions":       rep -> [T, n_segments]  (sampled indices)
              "old_log_probs": rep -> [T]
              "expert":        rep -> [T, ...]  (symbolic module's one-hot output)
          }
          rewards[reward_name] = [T] raw reward tensor

        All tensors are truncated to one common T: probers fire mid-cycle and
        the reward collector at the end of the execution order, so a halted
        final cycle desynchronizes them by one sample.
        """
        for prober in self.probers.values():
            prober.policy_model.eval()  # deactivate dropout/BatchNorm everywhere

        with torch.no_grad():
            self.project.run(quit_after_cycles=rollout_cycles, run_mode=0)

        raw = {name: prober.flush() for name, prober in self.probers.items()}
        raw_rewards = self.reward_collector.flush()

        # reconcile lengths across ALL agents and rewards
        lengths = {f"{name}.{k}": v.shape[0] for name, arrs in raw.items() for k, v in arrs.items()}
        lengths |= {k: v.shape[0] for k, v in raw_rewards.items()}
        T = min(lengths.values())
        spread = max(lengths.values()) - T
        if spread:
            # a 1-sample gap is the expected mid-cycle halt; more indicates a real problem
            log = logger.debug if spread <= 1 else logger.warning
            log("Trajectory lengths differ by %d, truncating to %d: %s", spread, T, lengths)

        per_agent = {}
        for name, arrs in raw.items():
            arrs = {k: v[:T] for k, v in arrs.items()}
            per_agent[name] = {
                "states":        {k.split("__", 1)[1]: v for k, v in arrs.items()
                                  if k.startswith(("requires__", "uses__"))},
                "actions":       {k.split("__", 1)[1]: v for k, v in arrs.items()
                                  if k.startswith("actions__")},
                "old_log_probs": {k.split("__", 1)[1]: v for k, v in arrs.items()
                                  if k.startswith("log_probs__")},
                "expert":        {k.split("__", 1)[1]: v for k, v in arrs.items()
                                  if k.startswith("provides__")},
            }

        rewards = {k.split("__", 1)[1]: v[:T] for k, v in raw_rewards.items()}
        return per_agent, rewards

    # ----------------------------------------------------------- advantages

    def compute_advantages(self, rewards: dict, discount_factor: float = 0.9) -> dict:
        """
        rewards: dict reward_name -> [T] raw reward tensor (from collect()).
        Returns dict agent_name -> [T] advantage tensor, or None per agent if
        there is no reward signal.

        Shared team signal: signals are centered (NOT std-normalized -- the
        absolute reward scale is load-bearing for the KL-regularized optimum
        and for constant-reward robustness), averaged, discounted, and the
        returns centered again. Every agent currently receives the same
        tensor (read-only downstream); difference rewards later replace only
        this mapping.
        """
        # TODO extend advantage calculation to PPO standard: Critic with loss + GAE
        # TODO: truncated GAE (we have endless episodes)
        if not rewards:
            logger.warning("Received no rewards.")
            return {name: None for name in self.probers}

        weights = {reward.name: reward.get_weight() for reward in self.reward_collector.rewards} or {}
        unknown = set(weights) - set(rewards)
        if unknown:
            logger.warning("Weights given for rewards that are not loaded: %s", unknown)

        names = list(rewards)                      # single source of truth for order
        w = torch.tensor([float(weights.get(n)) for n in names],
                         dtype=torch.float32, device=next(iter(rewards.values())).device)
        logger.debug("Reward weights: %s", dict(zip(names, w.tolist())))

        if not w.any():
            logger.warning("All reward weights are zero -> advantages will be zero.")

        stacked = torch.stack([rewards[n] for n in names], dim=0)
        stacked = stacked - stacked.mean(dim=1, keepdim=True)   # center each signal
        combined = (w[:, None] * stacked).sum(dim=0)            # weighted SUM, shape (T,)

        T = len(combined)
        returns = torch.zeros(T, device=combined.device)
        G = 0.0
        for t in reversed(range(T)):
            G = combined[t] + discount_factor * G
            returns[t] = G
        returns = returns - returns.mean()  # center only

        return {name: returns for name in self.probers}
