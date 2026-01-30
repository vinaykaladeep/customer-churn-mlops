import os
import yaml
from typing import List


def read_yaml(path: str) -> dict:
    """
    Reads a YAML file and returns content as dict
    """
    with open(path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def create_directories(paths: List[str]) -> None:
    """
    Create list of directories if they do not exist
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)