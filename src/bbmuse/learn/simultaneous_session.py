import logging
from copy import copy

from tqdm import tqdm

import torch

from time import time

logger = logging.getLogger(__name__)

from bbmuse.engine.project import BbMuseProject
from bbmuse.learn.sculpting_session import SculptingSession
from bbmuse.learn.checkpoint import Checkpoint
from bbmuse.learn.ppo_updater import PPOUpdater
from bbmuse.learn.reward_collector import RewardCollector
from bbmuse.learn.rollout_collector import RolloutCollector
from bbmuse.learn.session_logger import SessionLogger


class SimultaneousSculptingSession:
    """
    Simultaneous PPO over N agents: per global update, ONE shared rollout is
    collected and EVERY agent runs one PPO update on its own slice of it.
    "IPPO-shaped": independent per-agent updates, shared team reward, no
    critic yet -- each agent faces N-1 concurrently moving teammates, which
    is exactly the non-stationarity that round robin (IBR) avoids by
    freezing.

    Structure mirrors RoundRobinSculptingSession: one SculptingSession per
    agent (clone loading, prober patching, per-agent sculpt dirs), one
    shared RewardCollector (patched last) and one shared RolloutCollector.
    Unlike round robin, this coordinator does NOT call session.run(): it
    uses the built sessions as agent containers and drives each agent's
    PPOUpdater directly -- the per-update body (PPO + anchor diagnostics)
    lives inside PPOUpdater.update(), shared with every other scheduler.

    Critics later: a LOCAL critic (IPPO) moves into PPOUpdater; a CENTRAL
    critic (MAPPO) is evaluated here once per update over the full
    blackboard state and passed to each update() via `baseline`.

    Update order within a step is irrelevant: all agents update from data
    collected under the pre-update policies, and gradients do not cross
    agents, so iterating the sessions dict in insertion order is exact.

    Every global update produces N metric rows sharing one `global_update`
    value (long format, distinguished by the `agent` column) -- and, unlike
    round robin, `global_update` counts ROLLOUTS, so comparing the two
    schedulers on this axis compares them at equal environment interaction.
    """

    def __init__(self, project: BbMuseProject, module_manager, device=torch.device("cpu")):
        self.project = project
        self.module_manager = module_manager
        self.device = device

        self.sessions = {}          # canonical agent name -> SculptingSession
        self.reward_collector = None
        self.rollout_collector = None

    # ------------------------------------------------------------------ build
    # NOTE: keep in sync with RoundRobinSculptingSession.build() -- the patch
    # order invariant (all probers first, reward collector last) lives here.

    def build(self, args, module_names):
        """
        args:         the parsed sculpt args (module/tag/dry_run/...); the
                      `module` field is overridden per agent.
        module_names: agent names, e.g. ["AgentA", "AgentB", "AgentC", "AgentD"].
                      Order does NOT need to match execution order; sessions
                      are keyed by the handler's canonical name regardless of
                      how the user typed it.
        """
        if self.sessions:
            raise RuntimeError("build() called twice -- probers would be double-patched.")
        if not module_names:
            raise RuntimeError("No modules were given.")

        # 1. all sessions + probers first (each patches its own handler once)
        for name in module_names:
            agent_args = copy(args)
            agent_args.module = [name]

            session = SculptingSession(self.project, self.module_manager, self.device)
            session.build(agent_args, skip_collectors=True)
            self.sessions[session.agent_name] = session

            logger.info("Built session for %s (prober active).", session.agent_name)

        experiment_dir = self.module_manager.create_next_experiments_dir(tag=args.tag)
        self.session_logger = SessionLogger(experiment_dir)

        # 2. reward collector last -> outermost wrapper on the final module
        reward_fpaths = self.module_manager.get_available_rewards_filepaths()
        self.reward_collector = RewardCollector(self.project, reward_fpaths,
                                                log_path=experiment_dir, device=self.device)

        # 3. one shared rollout pipeline over all probers
        probers = {name: session.prober for name, session in self.sessions.items()}
        self.rollout_collector = RolloutCollector(self.project, probers,
                                                  self.reward_collector, device=self.device)

        # experiment report: what runs, with which rewards, and where each
        # agent's sculpt dir lives; completed with the schedule parameters
        # in run()
        self._experiment_info = {
            "mode": "simultaneous",
            "agents": list(self.sessions),
            "rewards": [p.name for p in reward_fpaths],
            "sculpt_run_dirs": {n: str(s.curr_run_dir) for n, s in self.sessions.items()},
        }
        self.session_logger.write_config_to_disk(self._experiment_info)

        for session in self.sessions.values():
            session.rollout_collector = self.rollout_collector
            session.session_logger = self.session_logger

        logger.info("Simultaneous session built for %s agents: %s",
                    len(self.sessions), ", ".join(self.sessions))
        return self

    # -------------------------------------------------------------------- run

    def run(self,
        num_updates: int = 100,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 256,
        entropy_coef = 0.0,
        bc_coef = 0.1,
        checkpoint_interval: int = 10,
        rollout_seconds: float = 8,
    ) -> None:
        """
        num_updates counts GLOBAL updates (= rollouts); every agent updates
        at each one. For an equal-environment-interaction comparison with
        round robin, match this against rounds * agents * per-phase updates.
        """
        if not self.sessions:
            raise RuntimeError("Call build() before run().")

        kwargs = {k: v for k, v in locals().items() if k != 'self'}

        self.session_logger.write_config_to_disk(self._experiment_info | {"run_kwargs": kwargs})

        # per-agent config into each sculpt dir (analogous to what
        # SculptingSession.run() writes when it drives the loop itself)
        for session in self.sessions.values():
            clone_info = {"clone_info": {"clone_epochs": session.loaded_checkpoint.get_epoch(),
                                         "clone_loss": session.loaded_checkpoint.get_loss()}}
            self.session_logger.write_config_to_disk({"run_kwargs": kwargs} | clone_info,
                                                     overwrite_dir=session.curr_run_dir)

        # one persistent updater (and optimizer) per agent -- same lazy
        # pattern as SculptingSession.run(), so a session sculpted before
        # keeps its Adam momentum here as well
        for session in self.sessions.values():
            if session.ppo_updater is None:
                session.ppo_updater = PPOUpdater(session.policy_model, lr=lr, bc_coef=bc_coef,
                                                 entropy_coef=entropy_coef, device=self.device)

        last_loss = {name: 0.0 for name in self.sessions}
        with tqdm(range(num_updates + 1)) as pbar:
            for update_i in pbar:

                if update_i > 0:
                    logger.debug("Start collecting trajectories (exploration phase)..")

                    # ONE rollout, then every agent updates on its own slice.
                    # NOTE: the collector's "advantages" are baseline-free
                    # (discounted, centered) returns; a central critic later
                    # turns into a `baseline=` argument below.
                    per_agent, rewards = self.rollout_collector.collect(quit_after=rollout_seconds)
                    returns = self.rollout_collector.compute_advantages(rewards)

                    logger.debug("Train policy models (learning phase)..")
                    for name, session in self.sessions.items():
                        mine = per_agent[name]

                        self.session_logger.log({f"rew_{rname}": v.mean().item()
                                                 for rname, v in rewards.items()})

                        metrics = session.ppo_updater.update(
                            states=mine["states"],
                            actions=mine["actions"],
                            old_log_probs=mine["old_log_probs"],
                            oracle=mine["oracle"],
                            returns=returns[name],
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                        last_loss[name] = metrics["weighted_loss"]

                        self.session_logger.log(metrics | {
                            "num_updates": update_i,
                            "walltime": time(),
                            "agent": name,
                            "global_update": update_i,
                        }).step()

                    mean_loss = sum(last_loss.values()) / len(last_loss)
                    pbar.set_description(f"update={update_i:04d} mean_loss={mean_loss:.6f}")

                # save intermediate policy checkpoints (per agent, shared metrics file)
                if checkpoint_interval and update_i % checkpoint_interval == 0:
                    for name, session in self.sessions.items():
                        if not session.dry_run:
                            ckpt_path = self.module_manager.get_checkpoint_path(session.curr_run_dir, update_i)
                            Checkpoint(ckpt_path).save(session.policy_model.model, update_i,
                                                       last_loss[name], session.ppo_updater.optimizer)
                self.session_logger.write_to_disk()

        # save final policies
        for name, session in self.sessions.items():
            if not session.dry_run:
                final_path = self.module_manager.get_final_model_path(session.curr_run_dir)
                Checkpoint(final_path).save(session.policy_model.model, update_i,
                                            last_loss[name], session.ppo_updater.optimizer)
        self.session_logger.write_to_disk()

        logger.info("Simultaneous sculpting finished after %s updates.", num_updates)
