import logging

import torch

logger = logging.getLogger(__name__)

from bbmuse.learn.policy_model import PolicyModel
from bbmuse.learn.action_spaces import make_ce_loss
from bbmuse.learn.metrics import estimate_entropy_floor


class PPOUpdater:
    """
    Owns ONE agent's PPO optimization: the policy model reference, its Adam
    optimizer, the per-head CE loss functions, and the anchor diagnostics
    (entropy floor / KL to the symbolic policy). Knows nothing about
    blackboards, probers, rollouts, or logging -- everything crossing this
    boundary is plain tensors in, a metrics dict out.

    Shared by all learning modes (standalone sculpt, round-robin/IBR,
    simultaneous/IPPO/MAPPO), so a fix to the clip, the BC weighting, or the
    loss normalization lands in exactly one place -- which is also the
    methodological argument that scheduling differences between modes are
    not implementation differences.

    The anchor diagnostics live here deliberately: the floor is computed
    from the SAME states/oracle tensors and the SAME action-space
    normalization as bc_loss, so the two can never silently diverge in
    scale or data. (To drop them later: delete the floor block in update()
    and the three metric keys.)

    Critics, when they come:
      - LOCAL critic (IPPO): will live inside this class -- it owns a value
        net over this agent's states and computes `baseline` itself.
      - CENTRAL critic (MAPPO): lives at the coordinator, which evaluates it
        once over the full blackboard state and passes the result in via
        `baseline`. Either way this signature does not change again.

    The optimizer persists across update() calls, so in round-robin mode an
    agent keeps its Adam momentum across rounds instead of resetting each
    phase.
    """

    def __init__(self,
        policy_model: PolicyModel,
        lr: float = 1e-3,
        bc_coef: float = 0.1,
        entropy_coef: float = 0.0,
        clip_eps: float = 0.2,
        device=torch.device("cpu"),
    ):
        self.policy_model = policy_model
        self.device = device
        self.bc_coef = bc_coef
        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps

        self.action_spaces = policy_model.model.config["action_spaces"]
        self.loss_functions = {name: make_ce_loss(nvec)
            for name, nvec in self.action_spaces.items()}
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)

    def update(self,
        states: dict,           # rep_name -> [T, ...] float tensors (packed inputs)
        actions: dict,          # rep_name -> [T, n_segments] index tensors
        old_log_probs: dict,    # rep_name -> [T] log probs at collection time
        oracle: dict,           # rep_name -> [T, ...] one-hot targets from the symbolic module
        returns,                # [T] (discounted, centered) return tensor, or None if no reward signal
        baseline=None,          # optional [T] value estimate; advantages = returns - baseline
        epochs: int = 5,
        batch_size: int = 256,
    ) -> dict:
        """
        Runs the full epoch/minibatch PPO(+BC+entropy) optimization on one
        rollout and returns a metrics dict. Optimization metrics reflect the
        LAST epoch only (deliberate: the state after the update, not a mean
        over a moving policy); the anchor diagnostics are computed on the
        full rollout before training.
        """
        assert epochs >= 1, "update() needs at least one epoch"

        advantages = None
        if returns is not None:
            advantages = returns if baseline is None else returns - baseline

        # entropy floor of the symbolic program AT THE STATES OF THIS ROLLOUT
        # (recomputed per update: the visited-state mix moves as agents
        # drift, so the floor is not a constant)
        floors, _, mean_group = estimate_entropy_floor(
            {k: v.detach().cpu().numpy() for k, v in states.items()},
            {k: v.detach().cpu().numpy() for k, v in oracle.items()},
            self.action_spaces,
            miller_madow=True,   # groups are far smaller than in cloning
        )
        floor = sum(floors.values()) / len(floors)

        self.policy_model.train()

        n_heads = len(self.loss_functions)
        T = next(iter(states.values())).shape[0]

        for epoch in range(epochs):
            indices = torch.randperm(T, device=self.device)

            epoch_loss = 0.0
            n_batches = 0
            epoch_policy_loss = []
            epoch_entropy = []
            epoch_bc_loss = []

            for batch_start in range(0, T, batch_size):
                idx = indices[batch_start:batch_start + batch_size]

                batch_states     = {k: v[idx] for k, v in states.items()}
                batch_old_lp     = {k: v[idx] for k, v in old_log_probs.items()}
                batch_actions    = {k: v[idx] for k, v in actions.items()}
                batch_oracle     = {k: v[idx] for k, v in oracle.items()}
                batch_advantages = advantages[idx] if advantages is not None else None

                # recompute log probs of OLD actions under CURRENT policy
                new_log_probs, entropies = self.policy_model.log_prob_with_entropy(batch_states, batch_actions)
                pred_actions = self.policy_model(batch_states)  # TODO remove duplicate forward call

                batch_loss = 0.0

                for head_name in new_log_probs.keys():

                    policy_loss = torch.zeros((), device=self.device)
                    if batch_advantages is not None:
                        A = batch_advantages
                        # importance ratio in log space for numerical stability
                        r = torch.exp(new_log_probs[head_name] - batch_old_lp[head_name])
                        # PPO clip loss
                        clipped = torch.clamp(r, 1 - self.clip_eps, 1 + self.clip_eps)
                        policy_loss = -torch.mean(torch.min(r * A, clipped * A))
                    epoch_policy_loss.append(policy_loss.item())

                    entropy = torch.mean(entropies[head_name])
                    epoch_entropy.append(entropy.item())

                    # BC loss: what the policy predicts vs what the symbolic module did
                    bc_loss = self.loss_functions[head_name](pred_actions[head_name], batch_oracle[head_name])
                    # METRIC ONLY: weighted by sample count and per-head share, so
                    # sum(epoch_bc_loss)/T is the per-sample mean CE averaged over
                    # heads -- the same scale as the entropy floor it gets
                    # subtracted from. The GRADIENT below stays unweighted on
                    # purpose: each batch should step with comparable magnitude
                    # regardless of (partial) batch size.
                    epoch_bc_loss.append(bc_loss.item() * len(idx) / n_heads)

                    batch_loss += sum([
                        policy_loss,
                        self.entropy_coef * -entropy,
                        self.bc_coef * bc_loss,
                    ]) / n_heads  # decouples head count from hyperparameter tuning

                self.optimizer.zero_grad()
                batch_loss.backward()  # one backward through the full shared graph
                self.optimizer.step()

                epoch_loss += batch_loss.item()
                n_batches += 1

        bc_mean = sum(epoch_bc_loss) / T  # per-sample mean (see weighting above)

        return {
            "weighted_loss": epoch_loss / n_batches,
            "policy_loss": sum(epoch_policy_loss) / len(epoch_policy_loss),
            "entropy": sum(epoch_entropy) / len(epoch_entropy),
            "bc_loss": bc_mean,
            "entropy_floor": floor,
            "kl_to_symbolic": bc_mean - floor,
            "mean_group_size": mean_group,
        }
