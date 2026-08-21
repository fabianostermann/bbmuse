import logging
import random
from copy import copy

import torch

logger = logging.getLogger(__name__)

from bbmuse.engine.project import BbMuseProject
from bbmuse.learn.sculpting_session import SculptingSession
from bbmuse.learn.reward_collector import RewardCollector
from bbmuse.learn.rollout_collector import RolloutCollector
from bbmuse.learn.session_logger import SessionLogger


class RoundRobinSculptingSession:
    """
    Iterated Best Response (IBR) over N agents.

    One SculptingSession per agent, all built once (each patches its own
    module handler exactly once). One shared RolloutCollector runs the
    environment and flushes ALL probers every rollout, so frozen agents act
    with their current learned policies but never accumulate buffers -- the
    active session simply uses only its own slice. Consequently there is no
    per-phase activation step anymore: eval/train modes are handled by the
    collector (eval before every rollout) and the PPOUpdater (train before
    every update).
    """

    def __init__(self, project: BbMuseProject, module_manager, device=torch.device("cpu")):
        self.project = project
        self.module_manager = module_manager
        self.device = device

        self.sessions = {}          # canonical agent name -> SculptingSession
        self.reward_collector = None
        self.rollout_collector = None

    # ------------------------------------------------------------------ build

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
        # agent's sculpt dir lives (so results can be relocated later);
        # completed with the schedule parameters in run()
        self._experiment_info = {
            "mode": "round_robin (iterated best response)",
            "agents": list(self.sessions),
            "rewards": [p.name for p in reward_fpaths],
            "sculpt_run_dirs": {n: str(s.curr_run_dir) for n, s in self.sessions.items()},
        }
        self.session_logger.write_config_to_disk(self._experiment_info)

        for session in self.sessions.values():
            session.rollout_collector = self.rollout_collector
            session.session_logger = self.session_logger

        logger.info("RoundRobin built for %s agents: %s",
                    len(self.sessions), ", ".join(self.sessions))
        return self

    # -------------------------------------------------------------------- run

    def run(self,
        rounds: int = 5,
        num_updates: int = 20,
        shuffle: bool = False,
        seed: int = None,
        **run_kwargs,
    ):
        """
        rounds:      how many full passes over all agents.
        num_updates: PER-PHASE update budget (explicit here so the global
                     step counter can advance without guessing the session
                     default).
        shuffle:     randomize agent order within each round; HAPPO uses a
                     random permutation per update for exactly this reason --
                     a fixed order biases who gets to move first. `seed`
                     makes shuffled orders reproducible across runs.
        run_kwargs:  forwarded to SculptingSession.run() (epochs, lr,
                     batch_size, entropy_coef, bc_coef, rollout_seconds, ...).
        """
        if not self.sessions:
            raise RuntimeError("Call build() before run().")

        self.session_logger.write_config_to_disk(self._experiment_info | {
            "rounds": rounds, "num_updates_per_phase": num_updates,
            "shuffle": shuffle, "seed": seed, "run_kwargs": dict(run_kwargs),
        })

        rng = random.Random(seed)
        names = list(self.sessions)
        global_update = 0

        for rnd in range(1, rounds + 1):
            order = names[:]
            if shuffle:
                rng.shuffle(order)
            logger.info("=== Round %s/%s -- order: %s ===", rnd, rounds, " -> ".join(order))

            for name in order:
                logger.info("--- Round %s: sculpting %s (others frozen) ---", rnd, name)

                self.sessions[name].run(
                    num_updates=num_updates,
                    log_context={"round": rnd, "agent": name},
                    log_global_offset=global_update,
                    **run_kwargs,
                )
                global_update += num_updates

        logger.info("Round-robin finished after %s rounds.", rounds)
