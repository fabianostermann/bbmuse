import logging

from pathlib import Path
import json

import pickle
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except Exception:
    pd = None


def melt_per_head(record: dict, heads) -> list:
    """Reshape one record into one row PER HEAD, with `head` as a column.

    Keys ending in "__<head>" for a head in `heads` become that row's column;
    every other key is repeated on every row. Matching against the KNOWN head
    names (rather than splitting on "__") is what keeps reward__prefer_c_zero
    a reward instead of a "prefer_c_zero" head.

    Any aggregate that has a per-head counterpart is dropped: it is exactly the
    mean over heads, so it is recoverable with
        df.groupby(["global_update", "agent"])[cols].mean()
    and keeping it would collide with the per-head column of the same name.
    Metrics with no per-head counterpart (weighted_loss, mean_group_size) are
    repeated across an agent's head rows -- aggregate those with .first(),
    never .mean().
    """
    heads = list(heads)
    shared, per_head = {}, {h: {} for h in heads}
    for key, value in record.items():
        for h in heads:
            if key.endswith("__" + h):
                per_head[h][key[:-len(h) - 2]] = value
                break
        else:
            shared[key] = value
    for row in per_head.values():
        for name in row:
            shared.pop(name, None)
    return [{"head": h, **shared, **per_head[h]} for h in heads]


class SessionLogger:

    def __init__(self, run_directory):
        if not pd:
            logger.warning("pandas not available. Falling back to pickling.")

        self.run_directory = Path(run_directory) if run_directory else None
        self.register_error_logfile()

        self.history = []
        self.current_step = {}
    
    def log(self, record_dict):
        """Add a record to the session history."""
        for k, v in record_dict.items():
            if isinstance(v, Tensor):
                record_dict[k] = v.item()
            #if type(v) is float:
            #    record_dict[k] = round(v, 5)

        self.current_step.update(record_dict)
        return self

    def step(self, heads=None):
        """Close the current record.

        With `heads`, the record is emitted in long format: one row per head,
        so the per-head metric columns stay dense and identically named no
        matter which heads an agent owns. Without, columns become "<metric>__<head>".
        """
        logger.debug(self.current_step)
        if heads is None:
            self.history.append(self.current_step)
        else:
            self.history.extend(melt_per_head(self.current_step, heads))
        self.current_step = {}

    def write_to_disk(self):
        if not self.run_directory:
            return

        """Write session history to disk."""
        if pd:
            filepath = Path(self.run_directory) / "metrics.csv"
            df = pd.DataFrame(self.history)
            df.to_csv(filepath, index=False)
        else:
            try:
                filepath = Path(self.run_directory) / "metrics.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(self.history, f)
            except Exception:
                logger.error("Writing anything to the given log file failed: %s", filepath)
                return

        logger.debug("Session log written to: %s", filepath)

    def _sanitize_dict_for_json(self, config_dict):
        result = {}
        for k, v in config_dict.items():
            if isinstance(v, dict):
                result[k] = self._sanitize_dict_for_json(v)  # recurse
            elif callable(v):
                result[k] = v.__name__
            else:
                result[k] = v
        return result

    def write_config_to_disk(self, config_dict, overwrite_dir=None):
        out_dir = self.run_directory
        if overwrite_dir:
            out_dir = overwrite_dir
        if not out_dir:
            return

        filepath = Path(out_dir) / "config.json"
        config_dict = self._sanitize_dict_for_json(config_dict)

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

        
    def register_error_logfile(self):
        if not self.run_directory:
            return

        root_logger = logging.getLogger()

        fh = logging.FileHandler(self.run_directory / "console.log", delay=True)
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))

        root_logger.addHandler(fh)
