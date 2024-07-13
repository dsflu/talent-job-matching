import yaml

def load_yaml_config(config_path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        A dictionary containing the config.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config