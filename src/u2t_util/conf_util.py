import yaml


def init_config(cf_config_path):
    with open(cf_config_path, 'r') as file:
        cf_config = yaml.load(file, Loader=yaml.FullLoader)
    return cf_config
