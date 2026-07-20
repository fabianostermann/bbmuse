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


class SessionLogger:

    def __init__(self, run_directory):
        if not pd:
            logger.warning("pandas not available. Falling back to pickling.")

        self.run_directory = Path(run_directory)

        self.history = []
        self.current_step = {}
    
    def log(self, record_dict):
        """Add a record to the session history."""
        for k, v in record_dict.items():
            if isinstance(v, Tensor):
                record_dict[k] = v.item()
            if type(v) is float:
                record_dict[k] = round(v, 5)

        self.current_step.update(record_dict)
        return self

    def step(self):
        logger.debug(self.current_step)
        self.history.append(self.current_step)
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

    def write_config_to_disk(self, config_dict):
        if not self.run_directory:
            return

        filepath = Path(self.run_directory) / "config.json"
        config_dict = self._sanitize_dict_for_json(config_dict)

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
