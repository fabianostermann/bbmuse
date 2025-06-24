import logging

from pathlib import Path
import tomli

logger = logging.getLogger(__name__)

syntax_keywords = {
    "REQUIRES": "requires",
    "PROVIDES": "provides",
    "USES": "uses",
}

class Config(dict):

    def __init__(self, project_dir):
        project_dir = Path(project_dir).absolute()
        config_file = project_dir.joinpath("project.bbmuse")

        if not Path.exists(config_file):
            logger.error("No bbmuse project file found in: %s", project_dir)
            raise FileExistsError(config_file)

        with open(config_file, 'rb') as f:
           self.update(tomli.load(f))
        self["project_dir"] = project_dir
        logger.debug("Config loaded from file: %s", self)
        
        error_logfile = project_dir.joinpath("error.log")
        self.setup_error_logging(error_logfile)
        logger.debug("Error log file: %s", error_logfile)

        
    def setup_error_logging(self, logfile):
        root_logger = logging.getLogger()

        fh = logging.FileHandler(logfile, delay=True)
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))

        root_logger.addHandler(fh)