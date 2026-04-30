import logging

from pathlib import Path

import pickle
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except Exception:
    pass


class SessionLogger:

    def __init__(self):
        if not pd:
            logger.warning("pandas not available. Falling back to pickling.")

        self.history = []
        self.current_step = {}
    
    def log(self, record_dict):
        """Add a record to the session history."""
        for k, v in record_dict.items():
            if isinstance(v, Tensor):
                record_dict[k] = v.item()

        self.current_step.update(record_dict)
        return self

    def step(self):
        logger.debug(self.current_step)
        self.history.append(self.current_step)
        self.current_step = {}

    def write_to_disk(self, run_directory):
        """Write session history to disk."""
        if pd:
            filepath = Path(run_directory) / "metrics.csv"
            df = pd.DataFrame(self.history)
            df.to_csv(filepath, index=False)
        else:
            try:
                logger.exception("Got a problem")
                filepath = Path(run_directory) / "metrics.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(self.history, f)
            except Exception:
                logger.error("Writing anything to the given log file failed: %s", filepath)
                return

        logger.debug("Session log written to: %s", filepath)
            


