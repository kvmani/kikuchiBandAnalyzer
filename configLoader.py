import yaml


def load_config(config_path="bandDetectorOptionsMagnetite.yml"):
    """
    Loads configuration from a YAML file.
    :param config_path: Path to the YAML configuration file.
    :return: Dictionary with configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
