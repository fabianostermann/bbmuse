import logging
import sys, os
import tempfile

from pathlib import Path

logger = logging.getLogger(__name__)

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
        logger.info("To apply a specific model, use: bblearn apply <module_name> [--clone|--sculpt] <id>")

    def apply(self, args):
        self.init(args)

        if not args.list:
            ckpt_path = None
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
                    ckpt_path = model_path                
                    self.write_apply(
                        self.module_handler.get_file_location(),
                        model_path,
                    )
                    return
                else:
                    logger.error("Requested model checkpoint not found: %s", model_path)

        self.list_available_models()
    
    def restore(self, args):
        self.init(args)
        self.write_restore(self.module_handler.get_file_location())

    def write_apply(self, module_path, checkpoint_path, device="cpu"):
        content = self.read_from_module_file(module_path)

        if "#bblearn---backup#" in content:
            logger.error("Writing aborted. bblearn-backup tag already in file: %s", module_path)
            return

        # backup original file content
        content = '\n'.join(f"#bblearn---backup#{line}" for line in content.splitlines())
        
        # add warning how to use the modified file
        content = USER_WARNING_STUB.replace(
            "###<bblearn---modle_name>###",
            self.module_handler.get_name()) \
            + '\n' + content
        
        # add code to make module bbmuse-native
        content += BBMUSE_NATIVE_MODULE_STUB
        content = content.replace("###<bblearn---checkpoint_path>###", f"\"{checkpoint_path}\"")
        content = content.replace("###<bblearn---torch.device>###", f"\"{device}\"")
        self.write_to_module_file(module_path, content)

    def write_restore(self, module_path):
        content = self.read_from_module_file(module_path)

        if not "#bblearn---backup#" in content:
            logger.error("Writing aborted. Did not find any bblearn-backup tag in file: %s", module_path)
            return

        content = '\n'.join(f"{line.replace("#bblearn---backup#", "")}"
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

USER_WARNING_STUB = """####
#
#  WARNING:
#    The content of this file was auto-modified by the BbLearn apply/restore utility.
#    Do not modify manually, if you do not exactly know what you are doing.
#
#    The intended way to restore this file is running:
#    $ bblearn restore ###<bblearn---modle_name>###
#
####
"""

# TODO: Hard code checkpoint loading to remove any dependency on bbmuse.learn
BBMUSE_NATIVE_MODULE_STUB = """

from pathlib import Path
import torch
from bbmuse.learn.checkpoint import Checkpoint

# --- this module's blackboard contract ---------------------------------------
USES     = [ "UsedRep" ]
REQUIRES = [ "ReqRep" ]
PROVIDES = [ "ProvRep", "UsedRep" ]
# ------------------------------------------------------------------------------

# --- checkpoint location + inference device ------------------------
CHECKPOINT_PATH = Path(###<bblearn---checkpoint_path>###)
DEVICE = torch.device(###<bblearn---torch.device>###)
# ------------------------------------------------------------------------------

_checkpoint = None
_model = None


def _init():
    global _checkpoint, _model

    _checkpoint = Checkpoint(CHECKPOINT_PATH, DEVICE).load()
    _model = _checkpoint.make_model()  # rebuilds ModuleClone, loads weights, moves to DEVICE
    _model.eval()                      # inference only: disable dropout/BatchNorm updates

    # sanity check: make sure the declared reps actually match this checkpoint
    expected_inputs = set(USES) | set(REQUIRES)
    expected_outputs = set(PROVIDES)
    print(_model.config["input_dims"])
    actual_inputs = set(_model.config["input_dims"].keys())
    actual_outputs = set(_model.config["output_dims"].keys())
    assert expected_inputs == actual_inputs, \
        f"USES/REQUIRES {expected_inputs} do not match checkpoint inputs {actual_inputs}"
    assert expected_outputs == actual_outputs, \
        f"PROVIDES {expected_outputs} do not match checkpoint outputs {actual_outputs}"

    print(
        "Loaded clone checkpoint from '%s' (trained epoch=%s, loss=%.6f)",
        CHECKPOINT_PATH, _checkpoint.get_epoch(), _checkpoint.get_loss(),
    )


def _update(bb):
    with torch.no_grad():
        # pack current blackboard state into tensors, add a batch dim of 1
        # since the model was trained on batched [B, *dims] arrays
        inputs = {
            name: torch.as_tensor(
                getattr(bb, name)._pack(), dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            for name in (USES + REQUIRES)
        }

        outputs = _model(inputs)

        # unpack predictions back onto the blackboard, dropping the batch dim again
        for name in PROVIDES:
            getattr(bb, name)._unpack(outputs[name].squeeze(0))


def close():
    global _checkpoint, _model
    print("Releasing clone model.")
    _model = None
    _checkpoint = None

"""