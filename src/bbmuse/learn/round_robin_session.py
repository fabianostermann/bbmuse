import logging
import random
from copy import copy

import torch

logger = logging.getLogger(__name__)

from bbmuse.engine.project import BbMuseProject
from bbmuse.learn.sculpting_session import SculptingSession
from bbmuse.learn.reward_collector import RewardCollector
from bbmuse.learn.session_logger import SessionLogger

class RoundRobinSculptingSession:
    """
    Iterated Best Response (IBR) over N agents.

    Per phase, exactly one session's run() is called (scheduled round robin);
    that session is the only one that flushes the prober and the shared RewardCollector.
    """

    def __init__(self, project: BbMuseProject, module_manager, device=torch.device("cpu")):
        self.project = project
        self.module_manager = module_manager
        self.device = device

        self.sessions = {}          # module name -> SculptingSession
        self.reward_collector = None

    # ------------------------------------------------------------------ build

    def build(self, args, module_names):
        """
        args:         the parsed sculpt args (module/tag/dry_run/...); the
                      `module` field is overridden per agent.
        module_names: agent names, e.g. ["AgentA", "AgentB", "AgentC", "AgentD"].
                      Order here does NOT need to match execution order.
        """
        if self.sessions:
            raise RuntimeError("build() called twice -- probers would be double-patched.")

        if not module_names:
            raise RuntimeError("No modules were given.")

        # --- 1. all sessions + probers first (each patches its own handler once)
        for name in module_names:
            agent_args = copy(args)
            agent_args.module = [name]

            session = SculptingSession(self.project, self.module_manager, self.device)
            session.build(agent_args, skip_reward_collector=True)
            self.sessions[name] = session

            logger.info("Built session for %s (prober active).", name)

        # --- 2. reward collector last -> outermost wrapper on the final module
        reward_fpaths = self.module_manager.get_available_rewards_filepaths()
        self.reward_collector = RewardCollector(self.project, reward_fpaths, self.device)

        # TODO: replace placeholder experiment directory
        experiment_dir = "/home/osterman/Nextcloud/Uni/Dissertation/bbmuse_dev/bbmuse/tests/LearnProject/.bblearn"
        self.session_logger = SessionLogger(experiment_dir)

        for session in self.sessions.values():
            session.reward_collector = self.reward_collector
            session.session_logger = self.session_logger

        logger.info("RoundRobin built for %s agents: %s",
                    len(self.sessions), ", ".join(self.sessions))
        return self

    # -------------------------------------------------------------------- run

    def run(self,
        rounds: int = 4,
        shuffle: bool = True,
        **run_kwargs
    ):
        """
        rounds:     how many full passes over all agents.
        shuffle:    randomize agent order within each round. HAPPO uses a
                    random permutation per update for exactly this reason --
                    a fixed order biases who gets to move first.
        run_kwargs: forwarded to SculptingSession.run() (num_updates, epochs,
                    lr, batch_size, entropy_coef, bc_coef, ...). These are
                    the PER-PHASE budget, not the total.
        """
        if not self.sessions:
            raise RuntimeError("Call build() before run().")

        names = list(self.sessions)
        global_update = 0

        for rnd in range(1, rounds + 1):
            order = names[:]
            if shuffle:
                random.shuffle(order)
            logger.info("=== Round %s/%s -- order: %s ===", rnd, rounds, " -> ".join(order))

            for name in order:
                logger.info("--- Round %s: sculpting %s (others frozen) ---", rnd, name)
                self._set_active(name)

                self.sessions[name].run(
                    log_context={"round": rnd, "agent": name},
                    log_global_offset=global_update,
                    **run_kwargs,
                )
                global_update += run_kwargs.get("num_updates")

        logger.info("Round-robin finished after %s rounds.", rounds)

    # ---------------------------------------------------------------- internals

    def _set_active(self, active_name):
        for name, session in self.sessions.items():
            prober = session.prober
            if name == active_name:
                prober.clear()   # start the phase clean
                session.policy_model.train()
            else:
                prober.clear()
                # TODO: allow probers to be set inactive (meaning to skip writing to buffers)
                session.policy_model.eval()
