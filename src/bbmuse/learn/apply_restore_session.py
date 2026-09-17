import logging
import sys, os
import tempfile

from pathlib import Path

logger = logging.getLogger(__name__)

from bbmuse.learn.applied import load_applied_table, write_applied_table, applied_table_path

class ApplyRestoreSession:
    def __init__(self, project, module_manager):
        self.project = project
        self.module_manager = module_manager

    def init(self, args):
        self.module_handler = self.module_manager.identify_module(args.module[0])
        if not self.module_handler:
            logger.error("Module handler not found: %s", args.module[0])
            sys.exit(1)

    def list_available_models(self):
        avail_clones_names = [path.stem for path
            in self.module_manager.get_available_clone_run_dirs(self.module_handler)
            if self.module_manager.get_final_model_path(path).exists()]
        avail_sculpts_names = [path.stem for path 
            in self.module_manager.get_available_sculpt_run_dirs(self.module_handler)
            if self.module_manager.get_final_model_path(path).exists()]

        logger.info("Ready-to-apply clones: %s", ", ".join(avail_clones_names))
        logger.info("Ready-to-apply sculpts: %s", ", ".join(avail_sculpts_names))

        applied = load_applied_table(self.module_manager).get(self.module_handler.get_name())
        if applied:
            logger.info("Currently applied: %s", applied["checkpoint"])
        else:
            logger.info("Currently applied: none, the module runs its own _update()")
        logger.info("To apply a specific model, use: bblearn apply <module_name> [--clone|--sculpt] <id>")

    def apply(self, args):
        self.init(args)

        if not args.list:
            models_dir = None

            if args.sculpt:
                models_dir = self.module_manager.get_sculpts_dir(self.module_handler) / args.sculpt
            elif args.clone:
                models_dir = self.module_manager.get_clones_dir(self.module_handler) / args.clone
            else:
                # TODO: default to auto-choose lastest model for apply sessions
                logger.warning("Not yet implement: auto-choose lastest model") # TODO!

            if models_dir:
                model_path = self.module_manager.get_final_model_path(models_dir)
                if model_path.exists():
                    self.write_apply(model_path, device=args.device or "cpu")
                    return
                else:
                    logger.error("Requested model checkpoint not found: %s", model_path)

        self.list_available_models()

    def restore(self, args):
        self.init(args)
        self.write_restore()

    def write_apply(self, checkpoint_path, device="cpu"):
        """
        Record that this module should run a trained model.

        The module's own source file is never touched: the engine reads this
        table when it builds the project and swaps the implementation in
        memory, so the hand-written version stays intact and reviewable, and
        switching back and forth costs nothing.
        """
        table = load_applied_table(self.module_manager)
        name = self.module_handler.get_name()

        work_dir = Path(self.module_manager.get_working_dir()).resolve()
        try:
            relative = Path(checkpoint_path).resolve().relative_to(work_dir)
        except ValueError:
            relative = Path(checkpoint_path).resolve()

        previous = table.get(name)
        table[name] = {"checkpoint": str(relative), "device": device}
        path = write_applied_table(self.module_manager, table)

        if previous:
            logger.info("Module %s now runs %s (was %s).",
                name, relative, previous.get("checkpoint"))
        else:
            logger.info("Module %s now runs %s.", name, relative)
        logger.info("Its source file is unchanged. Recorded in: %s", path)
        logger.info("Undo with: bblearn restore %s", name)

    def write_restore(self):
        name = self.module_handler.get_name()

        # a module that was modified in place by an older bblearn still carries
        # its original source in comments, so put that back first
        module_path = Path(self.module_handler.get_file_location())
        content = self.read_from_module_file(module_path)
        if "#bblearn---backup#" in content:
            self.restore_modified_source(module_path, content)

        table = load_applied_table(self.module_manager)
        if name not in table:
            logger.info("Module %s does not have an applied model.", name)
            return

        removed = table.pop(name)
        write_applied_table(self.module_manager, table)
        logger.info("Module %s no longer runs %s and is back to its own _update().",
            name, removed.get("checkpoint"))

    def restore_modified_source(self, module_path, content):
        logger.info("This module was modified in place by an older bblearn. "
            "Restoring its original source: %s", module_path)
        content = '\n'.join(line.removeprefix("#bblearn---backup#")
            for line in content.splitlines()
            if line.startswith("#bblearn---backup#"))
        self.write_to_module_file(module_path, content)

    def read_from_module_file(self, file_path: str | Path) -> str:
        """
        Read and return the full source code of a module file as a string.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("Path does not exist: %s", file_path)
            raise FileNotFoundError(file_path)

        content = file_path.read_text(encoding="utf-8")
        logger.debug("Read module file from disk: %s", file_path)
        return content


    def write_to_module_file(self, file_path: str | Path, content: str) -> None:
        """
        Write source code to a module file, overwriting any existing content.
        Creates parent directories if they don't exist yet.

        Writes to a temp file in the same directory and atomically renames it
        into place, so a concurrent reader/importer never sees a partially
        written file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, file_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        logger.debug("Wrote module file to disk: %s", file_path)


