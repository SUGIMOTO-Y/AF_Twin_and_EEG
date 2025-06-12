import yaml
import os
def read_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
