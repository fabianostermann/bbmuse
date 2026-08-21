import logging
import sys, os

from pathlib import Path

logger = logging.getLogger(__name__)

from bbmuse.learn.module_manager import ModuleManager

class Session():

    def __init__(self, project, args):
        logger.debug(args)

        try: # build project to setup modules, blackboard and dependency graph
            project.build_all()
        except Exception:
            logger.exception("Building project failed.")
            sys.exit(1)

        self.project = project
        self.module_manager = ModuleManager(project)
        
        # TODO: somehow set the project to "training mode" (maybe a hook on representations that hack in high tempos etc.)

        command = args.command.replace("-", "_")
        if hasattr(self, command):
            command_method = getattr(self, command)
            command_method(args)
        else:
            logger.error("Command '%s()' is unknown.", command)

        self.module_manager.clean()

    def arm(self, args):
        self.module_manager.arm(args)

    def disarm(self, args):
        self.module_manager.disarm(args)

    def status(self, args):
        self.module_manager.clean()
        self.module_manager.status(args)

    def listen(self, args):
        logger.debug("Starting ListeningSession..")
        from bbmuse.learn.listening_session import ListeningSession
        ls = ListeningSession(self.project, self.module_manager)
        ls.run(args)

    def clone(self, args):
        device = self.get_desired_torch_device(args.device)
        logger.debug("Starting CloningSession..")
        from bbmuse.learn.cloning_session import CloningSession
        cs = CloningSession(self.project, self.module_manager, device=device)
        cs.build(args)
        cs.run(epochs=args.epochs) # TODO build smart parameter loader (config file? also a formatted command-line option?)

    def sculpt(self, args):
        device = self.get_desired_torch_device(args.device)
        logger.debug("Starting SculptingSession..")
        from bbmuse.learn.sculpting_session import SculptingSession
        cs = SculptingSession(self.project, self.module_manager, device=device)
        cs.build(args)
        cs.run()
        
    def sculpt_rr(self, args):
        device = self.get_desired_torch_device(args.device)
        logger.debug("Starting SculptingSession with RoundRobin..")
        from bbmuse.learn.round_robin_session import RoundRobinSculptingSession
        rr = RoundRobinSculptingSession(self.project, self.module_manager, device=device)
        rr.build(args, module_names=args.modules)
        rr.run()
        
    def sculpt_sim(self, args):
        device = self.get_desired_torch_device(args.device)
        logger.debug("Starting SculptingSession with simultaneous updates..")
        from bbmuse.learn.simultaneous_session import SimultaneousSculptingSession
        sim = SimultaneousSculptingSession(self.project, self.module_manager, device=device)
        sim.build(args, module_names=args.modules)
        sim.run()

    def apply(self, args):
        from bbmuse.learn.apply_restore_session import ApplyRestoreSession
        aps = ApplyRestoreSession(self.project, self.module_manager)
        aps.apply(args)

    def restore(self, args):
        from bbmuse.learn.apply_restore_session import ApplyRestoreSession
        aps = ApplyRestoreSession(self.project, self.module_manager)
        aps.restore(args)

    def get_desired_torch_device(self, device_name):
        import torch

        # Check if GPU is available
        logger.debug("CUDA available: %s", torch.cuda.is_available())
        logger.debug("Current device: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

        # Check which device a tensor is on
        x = torch.tensor([1, 2, 3])
        logger.debug("Default tensor device: %s", x.device)

        if device_name:
            # Explicitly move tensors to desired device if provided
            device = torch.device(device_name)
        else:
            # Explicitly move tensors to GPU if available
            logger.debug("No desired device provided, trying: cuda (fallback: cpu).")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        logger.debug("Tensor device after moving to desired device (%s): %s", device_name, x.device)

        return device