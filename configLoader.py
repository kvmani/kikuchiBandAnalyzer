import yaml


def load_config(config_path: str = "bandDetectorOptionsMagnetite.yml") -> dict:
    """Load options from a YAML configuration file.

    The configuration is expected to describe data paths, simulation
    parameters and material information as used by
    ``KikuchiBandWidthAutomator`` and related utilities.  The YAML file
    usually contains keys such as ``h5_file_path``, ``phase_list`` and
    ``hkl_list`` among others.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file.  Defaults to
        ``"bandDetectorOptionsMagnetite.yml"``.

    Returns
    -------
    dict
        Parsed configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
