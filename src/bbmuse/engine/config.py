from pathlib import Path
import tomli

class Config(dict):

    def __init__(self, project_dir):
        project_dir = Path(project_dir).absolute()
        with open(project_dir.joinpath("project.bbmuse"), 'rb') as f:
           self.update(tomli.load(f))
        self["project_dir"] = project_dir