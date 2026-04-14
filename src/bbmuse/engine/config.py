import logging

from pathlib import Path
import tomllib

logger = logging.getLogger(__name__)

class Config(dict):

    def init_default(self):
        self.update(
            {
                "application": {
                    "name": "Untitled",
                    "description": "No description available.",
                    "icon": "icon.svg",
                },
                "path": {
                    "modules": [ "modules/" ],
                    "representations": [ "representations/" ],
                    "bblearn": ".bblearn/",
                },
            }
        )

    def __init__(self, project_dir):
        self.init_default()

        project_dir = Path(project_dir).absolute()
        config_file = project_dir.joinpath("project.bbmuse")

        if not Path.exists(config_file):
            logger.error("No bbmuse project file found in: %s", project_dir)
            raise FileNotFoundError(config_file)

        with open(config_file, 'rb') as f:
           self.update(tomllib.load(f))
        logger.debug("Config loaded from file: %s", self)

        self._project_dir = project_dir
        logger.debug("Project lives here: %s", self.get_project_dir())
        
        error_logfile = project_dir.joinpath("error.log")
        self.setup_error_logging(error_logfile)
        logger.debug("Error log file: %s", error_logfile)
    
    def get_project_dir(self):
        return Path(self._project_dir).absolute()
        
    def setup_error_logging(self, logfile):
        root_logger = logging.getLogger()

        fh = logging.FileHandler(logfile, delay=True)
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))

        root_logger.addHandler(fh)

    def update(self, new: dict):
        """ Enhance dict update function to make deep updates """
        Config.deep_update(self, new)

    def deep_update(original: dict, new: dict):
        for key, value in new.items():
            if (
                key in original
                and isinstance(original[key], dict)
                and isinstance(value, dict)
            ):
                Config.deep_update(original[key], value)
            else:
                original[key] = value
