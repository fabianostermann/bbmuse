import logging

from pathlib import Path
import tomllib

logger = logging.getLogger(__name__)

APPLIED_FILENAME = "applied.toml"

def applied_table_path(module_manager) -> Path:
    return Path(module_manager.get_working_dir()) / APPLIED_FILENAME

def load_applied_table(module_manager) -> dict:
    """
    Which modules currently run a trained model instead of their own _update().

    Kept beside the other bblearn state rather than inside the module file, so
    applying a model never touches the source a person wrote.
    """
    path = applied_table_path(module_manager)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        table = tomllib.load(f)
    logger.debug("Applied models: %s", table)
    return table

def write_applied_table(module_manager, table: dict) -> Path:
    path = applied_table_path(module_manager)
    lines = [
        "# Modules that currently run a trained model in place of their own",
        "# _update(). Written by 'bblearn apply' and removed by 'bblearn restore'.",
        "# The module source files themselves are never modified.",
        "",
    ]
    for module_name, entry in sorted(table.items()):
        lines.append(f"[{module_name}]")
        lines.append(f'checkpoint = "{entry["checkpoint"]}"')
        lines.append(f'device = "{entry.get("device", "cpu")}"')
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

class NeuralImplementation:
    """
    Runs a trained checkpoint in place of a module's own _update().

    The module file is still imported, so PROVIDES/REQUIRES/DELAYED/GROUP/RATE
    and everything else about the contract keep coming from the source the
    author wrote. Only the three lifecycle hooks are replaced.
    """

    def __init__(self, module_handler, checkpoint_path, device="cpu"):
        self.module_handler = module_handler
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self.model = None
        self.checkpoint = None

    def install(self, component):
        component._init = self.init
        component._update = self.update
        component._close = self.close

    def init(self):
        import torch
        from bbmuse.learn.checkpoint import Checkpoint

        self.device = torch.device(self.device_name)
        self.checkpoint = Checkpoint(self.checkpoint_path, self.device).load()
        self.model = self.checkpoint.make_model()
        self.model.eval()

        declared_inputs = set(self.module_handler.get_requires()) \
            | set(self.module_handler.get_uses()) \
            | set(self.module_handler.get_delayed())
        declared_outputs = set(self.module_handler.get_provides())
        actual_inputs = set(self.model.config["input_dims"].keys())
        actual_outputs = set(self.model.config["output_dims"].keys())
        if declared_inputs != actual_inputs:
            raise ValueError(
                f"{self.module_handler} declares inputs {sorted(declared_inputs)}, but "
                f"{self.checkpoint_path} was trained on {sorted(actual_inputs)}. "
                "Re-record and re-train, or restore the module.")
        if declared_outputs != actual_outputs:
            raise ValueError(
                f"{self.module_handler} declares PROVIDES {sorted(declared_outputs)}, but "
                f"{self.checkpoint_path} produces {sorted(actual_outputs)}. "
                "Re-record and re-train, or restore the module.")

        logger.info("%s now runs %s (trained epoch=%s, loss=%.6f).",
            self.module_handler, self.checkpoint_path,
            self.checkpoint.get_epoch(), self.checkpoint.get_loss())

    def update(self, bb):
        import torch

        live_names = list(self.module_handler.get_requires()) + list(self.module_handler.get_uses())
        delayed_names = list(self.module_handler.get_delayed())
        with torch.no_grad():
            # the model was trained on batched [B, *dims] arrays, so add a batch of 1
            inputs = {name: torch.as_tensor(getattr(bb, name)._pack(),
                    dtype=torch.float32, device=self.device).unsqueeze(0)
                for name in live_names}
            inputs |= {name: torch.as_tensor(getattr(bb.prev, name)._pack(),
                    dtype=torch.float32, device=self.device).unsqueeze(0)
                for name in delayed_names}

            outputs = self.model(inputs)

            for name in self.module_handler.get_provides():
                getattr(bb, name)._unpack(outputs[name].squeeze(0))

    def close(self):
        logger.debug("Releasing model for %s.", self.module_handler)
        self.model = None
        self.checkpoint = None
