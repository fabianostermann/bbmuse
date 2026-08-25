import logging

import torch

logger = logging.getLogger(__name__)

from bbmuse.learn.policy_model import PolicyModel
from bbmuse.learn.action_spaces import make_ce_loss, make_kl_losses
from bbmuse.learn.metrics import estimate_entropy_floor


class PPOUpdater:
    """
    Owns ONE agent's PPO optimization: the policy model reference, its Adam
    optimizer, the per-head loss functions, and the anchor diagnostics.
    Knows nothing about blackboards, probers, rollouts, or logging --
    everything crossing this boundary is plain tensors in, a metrics dict out.

    Shared by all learning modes (standalone sculpt, round-robin/IBR,
    simultaneous/IPPO/MAPPO), so a fix to the clip, the expert weighting, or the
    loss normalization lands in exactly one place -- which is also the
    methodological argument that scheduling differences between modes are
    not implementation differences.

    TWO ANCHORS, one axis
    ---------------------
    symbolic (expert_coef): CE against the one-hot action the symbolic module
        actually took this rollout. Reported as
        kl_to_expert = expert_loss - entropy_floor = KL(p_symbolic || q).
        The expert is queried live at policy-visited states (DAgger).
    reference (ref_coef): analytic KL against a FROZEN clone's predicted
        distribution. The ablation arm: does anchoring to a learned
        approximation suffice?

    KL DIRECTION is itself an ablation factor (`kl_direction`):
      "forward" = KL(ref || policy), mass-covering. Same direction as
          kl_to_expert, so the two anchors sit on one Pareto axis.
          Predicted to PRESERVE the anchor's entropy.
      "reverse" = KL(policy || ref), mode-seeking. The RLHF convention,
          where it is chosen because it is estimable from on-policy samples;
          here both directions are exact, so the choice is purely about the
          objective. Predicted to COLLAPSE entropy faster.
    Both directions are always computed and logged (kl_to_ref_forward /
    kl_to_ref_reverse) regardless of which one enters the loss, and
    regardless of ref_coef -- so a symbolic-anchored run still reports how
    far it drifted from its own starting clone, at no cost to the objective.
    kl_to_ref_forward is the one comparable to kl_to_expert.

    NOTE (ablation fairness): the reference term uses the clone's full soft
    distribution while the symbolic term uses one-hot samples, so the
    reference gradient has strictly lower variance. If the ref arm wins,
    rule that out before crediting the anchor itself -- e.g. by feeding the
    symbolic term the grouped soft targets from estimate_entropy_floor.

    The anchor diagnostics live here deliberately: the floor is computed
    from the SAME states/expert tensors and the SAME action-space
    normalization as expert_loss, so the two can never silently diverge in
    scale or data.

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
        ref_model: PolicyModel = None,   # frozen initial clone; None disables the arm
        ref_coef: float = 0.0,           # weight for the reference (clone) KL anchor
        kl_direction: str = "forward",   # "forward" = KL(ref||policy), "reverse" = KL(policy||ref)
        expert_coef: float = 0.1,            # weight for the symbolic anchor
        entropy_coef: float = 0.0,
        lr: float = 1e-3,
        clip_eps: float = 0.2,
        device=torch.device("cpu"),
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.ref_coef = ref_coef
        if kl_direction not in ("forward", "reverse"):
            raise ValueError(f"kl_direction must be 'forward' or 'reverse', got {kl_direction!r}")
        self.kl_direction = kl_direction
        self.expert_coef = expert_coef
        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps
        self.device = device

        self.action_spaces = policy_model.model.config["action_spaces"]
        self.loss_functions = {name: make_ce_loss(nvec)
            for name, nvec in self.action_spaces.items()}

        self.kl_functions = None
        if ref_model is not None:
            # only the parts that must match for the per-segment KL to be
            # meaningful; path_to_backbone and friends are irrelevant here
            for key in ("action_spaces", "input_dims", "output_dims"):
                assert ref_model.model.config[key] == policy_model.model.config[key], (
                    f"Reference model '{key}' does not match the policy model: "
                    f"{ref_model.model.config[key]} vs {policy_model.model.config[key]}")
            self.kl_functions = {name: make_kl_losses(nvec)
                for name, nvec in self.action_spaces.items()}
            # frozen: never trained, never in the optimizer, no grads retained
            self.ref_model.to(device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        elif ref_coef:
            raise ValueError("ref_coef > 0 but no ref_model was given.")

        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)

    def update(self,
        states: dict,           # rep_name -> [T, ...] float tensors (packed inputs)
        actions: dict,          # rep_name -> [T, n_segments] index tensors
        old_log_probs: dict,    # rep_name -> [T] log probs at collection time
        expert: dict,           # rep_name -> [T, ...] one-hot targets from the symbolic module
        returns,                # [T] (discounted, centered) return tensor, or None if no reward signal
        baseline=None,          # optional [T] value estimate; advantages = returns - baseline
        epochs: int = 5,
        batch_size: int = 256,
    ) -> dict:
        """
        Runs the full epoch/minibatch PPO(+expert+refKL+entropy) optimization on
        one rollout and returns a metrics dict. Optimization metrics reflect
        the LAST epoch only (deliberate: the state after the update, not a
        mean over a moving policy); the entropy floor is computed on the full
        rollout before training.
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
            {k: v.detach().cpu().numpy() for k, v in expert.items()},
            self.action_spaces,
            miller_madow=True,   # groups are far smaller than in cloning
        )
        floor = sum(floors.values()) / len(floors)

        # reference logits for the WHOLE rollout, once: they never change
        # during the update, so recomputing them per batch (and per epoch)
        # would be pure waste. Detached -> gradient flows to the policy only.
        ref_logits = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = {k: v.detach() for k, v in self.ref_model(states).items()}

        self.policy_model.train()

        n_heads = len(self.loss_functions)
        T = next(iter(states.values())).shape[0]

        for epoch in range(epochs):
            indices = torch.randperm(T, device=self.device)

            epoch_loss = 0.0
            n_batches = 0
            epoch_policy_loss = []
            epoch_entropy = []
            epoch_entropy_per_head = {}
            epoch_expert_loss = []
            epoch_ref_fwd = []
            epoch_ref_rev = []

            for batch_start in range(0, T, batch_size):
                idx = indices[batch_start:batch_start + batch_size]

                batch_states     = {k: v[idx] for k, v in states.items()}
                batch_old_lp     = {k: v[idx] for k, v in old_log_probs.items()}
                batch_actions    = {k: v[idx] for k, v in actions.items()}
                batch_expert     = {k: v[idx] for k, v in expert.items()}
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
                    epoch_entropy_per_head.setdefault(head_name, []).append(entropy)
                    epoch_entropy.append(entropy.item())

                    # symbolic anchor: what the policy predicts vs what the
                    # symbolic module actually did (one-hot samples)
                    expert_loss = self.loss_functions[head_name](pred_actions[head_name], batch_expert[head_name])
                    # METRIC ONLY: weighted by sample count and per-head share, so
                    # sum(...)/T is the per-sample mean averaged over heads -- the
                    # same scale as the entropy floor it gets subtracted from. The
                    # GRADIENT below stays unweighted on purpose: each batch should
                    # step with comparable magnitude regardless of (partial) batch size.
                    epoch_expert_loss.append(expert_loss.item() * len(idx) / n_heads)

                    # reference anchor: analytic KL against the frozen clone.
                    # BOTH directions are always computed and logged when a
                    # ref_model exists -- even at ref_coef = 0, where neither
                    # contributes to the loss. Only the selected direction is
                    # added to the objective.
                    ref_kl = torch.zeros((), device=self.device)
                    if ref_logits is not None:
                        fwd, rev = self.kl_functions[head_name](
                            ref_logits[head_name][idx], pred_actions[head_name])
                        epoch_ref_fwd.append(fwd.item() * len(idx) / n_heads)
                        epoch_ref_rev.append(rev.item() * len(idx) / n_heads)
                        ref_kl = fwd if self.kl_direction == "forward" else rev

                    batch_loss += sum([
                        policy_loss,
                        self.entropy_coef * -entropy,
                        self.expert_coef * expert_loss,
                        self.ref_coef * ref_kl,
                    ]) / n_heads  # decouples head count from hyperparameter tuning

                self.optimizer.zero_grad()
                batch_loss.backward()  # one backward through the full shared graph
                self.optimizer.step()

                epoch_loss += batch_loss.item()
                n_batches += 1

        expert_mean = sum(epoch_expert_loss) / T  # per-sample mean (see weighting above)

        metrics = {
            "weighted_loss": epoch_loss / n_batches,
            "policy_loss": sum(epoch_policy_loss) / len(epoch_policy_loss),
            "entropy": sum(epoch_entropy) / len(epoch_entropy),
            "expert_loss": expert_mean,
            "entropy_floor": floor,
            "kl_to_expert": expert_mean - floor,
            "mean_group_size": mean_group,
        }
        metrics |= { "entropy__"+name: sum(entropy)/len(entropy) for name, entropy in epoch_entropy_per_head.items() }
        if epoch_ref_fwd:
            # kl_to_ref_forward is the one on the same axis as kl_to_expert
            metrics["kl_to_ref_forward"] = sum(epoch_ref_fwd) / T
            metrics["kl_to_ref_reverse"] = sum(epoch_ref_rev) / T
            metrics["kl_direction"] = self.kl_direction   # constant, but keeps
            #   concatenated ablation CSVs self-describing
        return metrics
