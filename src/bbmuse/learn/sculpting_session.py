import logging
import sys

from tqdm import tqdm

import torch

from time import time

logger = logging.getLogger(__name__)

from bbmuse.engine.project import BbMuseProject

from bbmuse.learn.checkpoint import Checkpoint
from bbmuse.learn.policy_prober import PolicyProber
from bbmuse.learn.reward_collector import RewardCollector
from bbmuse.learn.rollout_collector import RolloutCollector
from bbmuse.learn.ppo_updater import PPOUpdater
from bbmuse.learn.policy_model import PolicyModel
from bbmuse.learn.session_logger import SessionLogger


class SculptingSession:
    """
    Single-agent composition: loads the clone, owns this agent's prober and
    PPOUpdater, and per update asks the RolloutCollector for a rollout, runs
    the PPO update (which also computes the anchor diagnostics), logs, and
    checkpoints.

    Standalone (`bblearn sculpt X`): build() creates its own RewardCollector
    and a RolloutCollector over just this one prober.
    Coordinated (round robin): build(skip_collectors=True) creates and
    patches ONLY the prober; the coordinator assigns a shared
    `rollout_collector` and `session_logger` before run().
    The simultaneous coordinator uses built sessions as agent containers
    (policy model, prober, updater, run dir) and drives their
    `ppo_updater.update()` directly instead of calling run().
    """

    def __init__(self, project: BbMuseProject, module_manager, device=torch.device("cpu")):
        self.project = project
        self.module_manager = module_manager
        self.device = device

        # created lazily on the first run() (needs lr/expert_coef/... from run
        # kwargs) and then REUSED, so Adam momentum survives across IBR
        # rounds. Changed hyperparameters on a later run() call are ignored.
        self.ppo_updater = None

    # ------------------------------------------------------------------ build

    def build(self, args, skip_collectors=False):
        """
        skip_collectors=True is for coordinators: the session then patches
        only its own module handler, so the coordinator controls the global
        patch order (all probers first, RewardCollector last -> the
        collector's wrapper ends up outermost on the final module of the
        execution order and scores the policy's action, not the symbolic one).
        """
        self.module_handler = self.module_manager.identify_module(args.module[0])
        if not self.module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

        self.agent_name = self.module_handler.get_name()  # canonical key into RolloutCollector output
        self.tag = args.tag
        self.dry_run = args.dry_run

        self.curr_run_dir = None
        if not self.dry_run:
            self.curr_run_dir = self.module_manager.create_next_sculpt_run_dir(self.module_handler, self.tag)
        self.session_logger = SessionLogger(self.curr_run_dir)

        # load clone from disk -- TODO: create mode that runs without expert model (init just a random model)
        clone_dirs = self.module_manager.get_available_clone_run_dirs(self.module_handler)
        clone_final_path = self.module_manager.get_final_model_path(clone_dirs[-1])
        self.loaded_checkpoint = Checkpoint(clone_final_path, self.device).load()
        clone_model = self.loaded_checkpoint.make_model()
        self.policy_model = PolicyModel(clone_model)
        self.policy_model.to(self.device)
        logger.info("Loaded policy model to be trained from: %s", clone_final_path)

        # load reference model from same checkpoint
        loaded_ref_checkpoint = Checkpoint(clone_final_path, self.device).load()
        clone_ref_model = loaded_ref_checkpoint.make_model()
        self.ref_model = PolicyModel(clone_ref_model)
        self.ref_model.to(self.device)
        logger.info("Loaded ref model for anchoring from: %s", clone_final_path)

        # make module prober (patches this module's call_update exactly once)
        self.prober = PolicyProber(self.policy_model, self.module_handler, self.project.get_blackboard())
        self.prober.activate_listen()

        if skip_collectors:
            self.reward_collector = None
            self.rollout_collector = None   # assigned by the coordinator
        else:
            reward_fpaths = self.module_manager.get_available_rewards_filepaths()
            self.reward_collector = RewardCollector(self.project, reward_fpaths,
                                                    log_path=self.curr_run_dir, device=self.device)
            self.rollout_collector = RolloutCollector(self.project, {self.agent_name: self.prober},
                                                      self.reward_collector, device=self.device)

    # -------------------------------------------------------------------- run

    def run(self,
        # num_updates: int = 100,
        # epochs: int = 5,
        # lr: float = 1e-3,
        # batch_size: int = 256,
        # entropy_coef = 0.0,
        # expert_coef = 0.1,
        # checkpoint_interval: int = 10,
        # rollout_seconds: float = 8,
        # log_context = {},
        # log_global_offset = 0,
    ) -> None:
        raise NotImplementedError("Do not call, currently not maintained.")

        # kwargs = {k: v for k, v in locals().items() if k != 'self'}

        # clone_info = {"clone_info": {"clone_epochs": self.loaded_checkpoint.get_epoch(),
        #                              "clone_loss": self.loaded_checkpoint.get_loss()}}
        # self.session_logger.write_config_to_disk({"run_kwargs": kwargs} | clone_info,
        #                                          overwrite_dir=self.curr_run_dir)

        # if self.rollout_collector is None:
        #     raise RuntimeError("No rollout_collector assigned. Either build without "
        #                        "skip_collectors or let the coordinator assign a shared one.")

        # if self.ppo_updater is None:
        #     self.ppo_updater = PPOUpdater(self.policy_model, lr=lr, expert_coef=expert_coef,
        #                                   ref_model = self.ref_model,
        #                                   entropy_coef=entropy_coef, device=self.device)

        # metrics = {"weighted_loss": 0.0}
        # with tqdm(range(num_updates + 1)) as pbar:
        #     for update_i in pbar:

        #         if update_i > 0:
        #             logger.debug("Start collecting trajectories (exploration phase)..")

        #             # one shared rollout; this session uses only its own slice.
        #             # NOTE: the collector's "advantages" are baseline-free
        #             # (discounted, centered) returns; the updater subtracts a
        #             # baseline if one exists (critics, later).
        #             per_agent, rewards = self.rollout_collector.collect(quit_after=rollout_seconds)
        #             returns = self.rollout_collector.compute_advantages(rewards)

        #             mine = per_agent[self.agent_name]

        #             self.session_logger.log({f"rew_{name}": v.mean().item() for name, v in rewards.items()})

        #             logger.debug("Train policy model (learning phase)..")
        #             metrics = self.ppo_updater.update(
        #                 states=mine["states"],
        #                 actions=mine["actions"],
        #                 old_log_probs=mine["old_log_probs"],
        #                 expert=mine["expert"],
        #                 returns=returns[self.agent_name],
        #                 epochs=epochs,
        #                 batch_size=batch_size,
        #             )

        #             self.session_logger.log(metrics | {
        #                 "num_updates": update_i,
        #                 "walltime": time(),
        #             } | log_context | {"global_update": update_i + log_global_offset}
        #             ).step()

        #             pbar.set_description(f"update={update_i:04d} loss={metrics['weighted_loss']:.6f}")

        #         # save intermediate policy checkpoints
        #         if not self.dry_run:
        #             if checkpoint_interval and update_i % checkpoint_interval == 0:
        #                 ckpt_path = self.module_manager.get_checkpoint_path(
        #                     self.curr_run_dir, update_i + log_global_offset)
        #                 Checkpoint(ckpt_path).save(self.policy_model.model, update_i,
        #                                            metrics["weighted_loss"], self.ppo_updater.optimizer)
        #             self.session_logger.write_to_disk()

        # # save final policy
        # if not self.dry_run:
        #     final_path = self.module_manager.get_final_model_path(self.curr_run_dir)
        #     Checkpoint(final_path).save(self.policy_model.model, update_i,
        #                                 metrics["weighted_loss"], self.ppo_updater.optimizer)
        #     self.session_logger.write_to_disk()
