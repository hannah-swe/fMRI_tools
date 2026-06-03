from pathlib import Path
import yaml


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Please copy config.example.yml to config.yml."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config