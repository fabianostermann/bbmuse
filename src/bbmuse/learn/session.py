import logging
import sys, os

from pathlib import Path

logger = logging.getLogger(__name__)

class Session():

    def __init__(self, project, args):
        logger.debug(args)

        try:
            requested_log_level = logging.getLogger().level
            logger.debug("If on INFO log level, increase level to ERROR to suppress bbmuse project loading output.")
            if logging.getLogger().level == logging.INFO:
                logging.getLogger().setLevel(logging.ERROR)
            project.build_all()
            logging.getLogger().setLevel(requested_log_level)
        except Exception:
            logging.getLogger().setLevel(desired_log_level)
            logger.exception("Building project failed.")
            sys.exit(1)

        self.project = project
        self.prepare_working_dir()

        if hasattr(self, args.command):
            command_method = getattr(self, args.command)
            command_method(args)
        else:
            logger.error("Method for command '%s()' is not implemented yet.", args.command)

        #print(project.get_module_handlers())
        #print(project.get_representation_handlers())
        #print(project.get_controller())
        #print(project.get_blackboard())

    def prepare_working_dir(self):
        self._working_dir = Path(self.project.config["path"]["bblearn"])
        if not self._working_dir.exists():
            self._working_dir.mkdir()
        self._modules_dir = Path(self._working_dir, "modules/")
        if not self._modules_dir.exists():
            self._modules_dir.mkdir()

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
                logger.error("Unable to find module: %s", mod)
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
            armed_indicator = mod_dir / ".armed" # TODO write this to a config toml file
            if disarm:
                armed_indicator.unlink(missing_ok=True)
                logger.info("Module %s disarmed.", mod)
            else: # arm
                armed_indicator.touch()
                logger.info("Module %s armed.", mod)

    def get_module_dir(self, module_handler):
        assert not module_handler is None, "Argument should be a valid ModuleHandler object"
        return self._modules_dir / module_handler.get_name().lower()

    def is_armed(self, module_handler):
        mod_dir = self.get_module_dir(module_handler)
        return (mod_dir / ".armed").exists()

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

    def disarm(self, args):
        self.arm(args, disarm=True)

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

    def collect(self, args):
        logger.error("collect() is not implemented yet.")

    def train(self, args):
        logger.error("train() is not implemented yet.")

    def apply(self, args):
        logger.error("apply() is not implemented yet.")

    def restore(self, args):
        logger.error("restore() is not implemented yet.")