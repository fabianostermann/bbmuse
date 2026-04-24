import logging
import sys, os

from pathlib import Path

logger = logging.getLogger(__name__)

class ModuleManager():

    def __init__(self, project):
        self.project = project
        self.prepare_working_dir()

    def prepare_working_dir(self):
        self._working_dir = Path(self.project.config["bblearn"]["work"])
        self._working_dir.mkdir(parents=True, exist_ok=True)
        
        self._modules_dir = Path(self._working_dir, "modules/")
        self._modules_dir.mkdir(parents=True, exist_ok=True)
        
        self._backbones_dir = Path(self.project.config["bblearn"]["backbones"])
        self._backbones_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Working dir set to: %s", self._working_dir)

    def get_working_dir(self):
        return self._working_dir

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

    # TODO later: change function names and variables called "episodes" for listen & clone to "records" 
    def get_episodes_dir(self, module_handler):
        episodes_dir = self.get_module_dir(module_handler) / "records"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        return episodes_dir

    def get_next_episode_path(self, module_handler):
        existing = self.get_available_episode_paths(module_handler)
        if existing:
            last_number = int(existing[-1].stem)
            next_number = last_number + 1
        else:
            next_number = 1
    
        return self.get_episodes_dir(module_handler) / f"{next_number:03d}.npz"

    def get_available_episode_paths(self, module_handler):
        episodes_dir = self.get_episodes_dir(module_handler)
        return sorted(episodes_dir.glob("*.npz"))

    def get_clones_dir(self, module_handler):
        clones_dir = self.get_module_dir(module_handler) / "clones"
        clones_dir.mkdir(parents=True, exist_ok=True)
        return clones_dir

    def create_next_clone_run_dir(self, module_handler):
        existing = self.get_available_clone_run_dirs(module_handler)
        if existing:
            last_number = int(existing[-1].stem)
            next_number = last_number + 1
        else:
            next_number = 1
    
        next_clone_run_dir = self.get_clones_dir(module_handler) / f"{next_number:03d}"
        next_clone_run_dir.mkdir(parents=False, exist_ok=False)
        return next_clone_run_dir

    def get_available_clone_run_dirs(self, module_handler):
        clones_dir = self.get_clones_dir(module_handler)
        return sorted(clones_dir.glob("*"))

    def get_sculpts_dir(self, module_handler):
        sculpts_dir = self.get_module_dir(module_handler) / "sculpts"
        sculpts_dir.mkdir(parents=True, exist_ok=True)
        return sculpts_dir

    def create_next_sculpt_run_dir(self, module_handler):
        existing = self.get_available_sculpt_run_dirs(module_handler)
        if existing:
            last_number = int(existing[-1].stem)
            next_number = last_number + 1
        else:
            next_number = 1
    
        next_sculpt_run_dir = self.get_sculpts_dir(module_handler) / f"{next_number:03d}"
        next_sculpt_run_dir.mkdir(parents=False, exist_ok=False)
        return next_sculpt_run_dir

    def get_available_sculpt_run_dirs(self, module_handler):
        sculpts_dir = self.get_sculpts_dir(module_handler)
        return sorted(sculpts_dir.glob("*"))

    def get_checkpoint_path(self, run_dir: str | Path, epoch: int):
        """ Intended for use with clones and sculpt directories """
        checkpoint_dir = Path(run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / f"epoch_{epoch:04d}.ckpt"

    def get_final_model_path(self, run_dir: str | Path):
        """ Intended for use with clones and sculpt directories """
        return Path(run_dir) / "final.pt"

    def get_backbones_dir(self):
        return self._backbones_dir

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
        logger.info("───STATUS───")
        modules = args.modules
        if len(modules) == 0:
            modules = [m.get_name() for m in self.project.get_module_handlers()]
        for mod_name in modules:
            self.print_module_info(mod_name, short=args.short)

    def print_module_info(self, module_name, short=False):
        # collect info
        mh = self.identify_module(module_name)
        if mh is None:
            # TODO module is present but skipped in the dependency graph, could run BaseHandler(path) manually here in the future
            logger.info("Unable to find module: %s (not existing or not in the current dependency graph)", module_name)
            return
            
        mod_dir = self.get_module_dir(mh)
        if not mod_dir.exists():
            logger.info("%s is untracked. Run 'bblearn arm %s' first.", mh, mh.get_name())
            return
        
        is_armed = self.is_armed(mh)
        avail_episodes_names = [path.stem for path in self.get_available_episode_paths(mh)]
        avail_clones_names = [path.stem for path in self.get_available_clone_run_dirs(mh)]
        avail_sculpts_names = [path.stem for path in self.get_available_sculpt_run_dirs(mh)]

        if short:
            short_info = ", ".join([x for x in [
                "armed" if is_armed else "not armed",
                str(len(avail_episodes_names))+" records" if len(avail_episodes_names) > 0 else None,
                str(len(avail_clones_names))+" clones" if len(avail_clones_names) > 0 else None,
                str(len(avail_sculpts_names))+" sculpts" if len(avail_sculpts_names) > 0 else None,
            ] if x is not None])
            logger.info("%s: %s",
                mh,
                short_info
            )
        else:
            logger.info("%s %s", mh, "is armed" if is_armed else "is not armed.")
            if avail_episodes_names:
                logger.info("├── %s records: %s", len(avail_episodes_names), ", ".join(avail_episodes_names))
            else:
                logger.info("├── no records")

            
            if avail_clones_names:
                logger.info("├── %s clones: %s", len(avail_clones_names), ", ".join(avail_clones_names))
            else:
                logger.info("├── no clones")

            
            if avail_sculpts_names:
                logger.info("└── %s sculpts: %s", len(avail_sculpts_names), ", ".join(avail_sculpts_names))
            else:
                logger.info("└── no sculpts")
