from src.yaml_utils import read_yml, write_yml


class ConfigManager:
    def __init__(self, yml_file):
        self.yml_file = yml_file

        self.__dict__.update(read_yml(yml_file))

    def save(self):
        write_yml(self.__dict__, self.yml_file)
