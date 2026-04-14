import logging
import sys, os

from pathlib import Path

logger = logging.getLogger(__name__)

class ModuleManager():

    def __init__(self, project):
        self.project = project
        self.prepare_working_dir()

    def prepare_working_dir(self):
        self._working_dir = Path(self.project.config["path"]["bblearn"])
        self._working_dir.mkdir(parents=True, exist_ok=True)
        
        self._modules_dir = Path(self._working_dir, "modules/")
        self._modules_dir.mkdir(parents=True, exist_ok=True)

    def disarm(self, args):
        self.arm(args, disarm=True)

    def arm(self, args, disarm=False):
        modules = args.modules
        if len(modules) == 0:
            logger.info("No modules given to %s.", "disarm" if disarm else "arm")
            sys.exit(0)

        logger.debug("args: %s", args)
        modules_to_arm = []
        for mod in modules:
            found_mod = self.identify_module(mod)
            if found_mod:
                modules_to_arm.append(found_mod)
            else:
                logger.info("Unable to identify module: %s", mod)
                sys.exit(1)
                
        # remove duplicates
        modules_to_arm = list(set(modules_to_arm))
        logger.debug("Modules to %s: %s", "disarm" if disarm else "arm", modules_to_arm)

        for mod in modules_to_arm:
            mod_dir = self.get_module_dir(mod)
            if not mod_dir.exists():
                if disarm:
                    logger.info("Module %s is untracked.", mod)
                    continue
                else: # arm
                    mod_dir.mkdir()
            armed_indicator = mod_dir / "_armed" # TODO write this to a config toml file
            if disarm:
                armed_indicator.unlink(missing_ok=True)
                logger.info("Module %s disarmed.", mod)
            else: # arm
                armed_indicator.touch()
                logger.info("Module %s armed.", mod)

    def get_module_dir(self, module_handler):
        assert not module_handler is None, "Argument should be a valid ModuleHandler object"
        return self._modules_dir / module_handler.get_name().lower()

    def get_next_episode_path(self, module_handler):
        episodes_dir = self.get_module_dir(module_handler) / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        
        existing = sorted(episodes_dir.glob("ep_*.npz"))
        if existing:
            last_number = int(existing[-1].stem.split("_")[1])
            next_number = last_number + 1
        else:
            next_number = 1
    
        return episodes_dir / f"ep_{next_number:03d}.npz"

    def is_armed(self, module_handler):
        mod_dir = self.get_module_dir(module_handler)
        return (mod_dir / "_armed").exists()

    def identify_module(self, mod):
        """
        mod: Name or path of module
        """
        for mh in self.project.get_module_handlers():
            try:
                if mh.get_file_location().samefile(Path(mod)):
                    logger.debug("Found path equivalence: %s", mod)
                    return mh
            except Exception:
                pass
            if mh.get_name().lower() == mod.lower():
                logger.debug("Found name equivalence: %s", mod)
                return mh
        return None

    def status(self, args):
        modules = args.modules
        if len(modules) == 0:
            for mh in self.project.get_module_handlers():
                self.print_module_info(mh.get_name())
        else:
            for mod in modules:
                self.print_module_info(mod)

    def print_module_info(self, module):
        # collect info
        mh = self.identify_module(module)
        if mh is None:
            # TODO module is present but skipped in the dependency graph, could run BaseHandler(path) manually here in the future
            logger.info("Unable to find module: %s (not existing or not in the current dependency graph)", module)
            return
            
        mod_dir = self.get_module_dir(mh)
        if not mod_dir.exists():
            logger.info("%s is untracked. Run 'bblearn arm %s' first.", mh, mh.get_name())
            return
        
        is_armed = self.is_armed(mh)
        logger.info("%s %s", mh, "is armed." if is_armed else "is not armed.")
