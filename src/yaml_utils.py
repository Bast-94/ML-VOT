import os

import yaml


def write_yml(dictionnary, file_path="config.yml"):
    with open(file_path, "w") as fichier:
        yaml.dump(dictionnary, fichier)
    print(f"Config saved to {file_path}")


def read_yml(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = yaml.safe_load(file)
            if content is None:
                content = {}
    else:
        content = {}
    return content
