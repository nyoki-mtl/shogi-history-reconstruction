"""Load base and experiment YAML configurations with OmegaConf."""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from posdiffusion.utils.project_dirs import ProjectDirs

# The runtime configuration object is an OmegaConf ``DictConfig``. The alias
# keeps type hints readable across the codebase.
Config = DictConfig


def load_config(
    experiment: str | Path,
    base: str | Path | None = None,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load an experiment config merged onto the base config.

    Args:
        experiment: Path to an experiment YAML (e.g.
            ``configs/experiments/constrained.yaml``).
        base: Optional override for the base config path. Defaults to
            ``configs/base.yaml``.
        overrides: Optional dotlist overrides, e.g. ``["training.batch_size=128"]``.

    Returns:
        Merged ``DictConfig``.
    """
    config_dir = ProjectDirs.get_config_dir()
    base_path = Path(base) if base is not None else config_dir / "base.yaml"
    experiment_path = Path(experiment)

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    base_cfg = OmegaConf.load(base_path)
    experiment_cfg = OmegaConf.load(experiment_path)
    cfg = OmegaConf.merge(base_cfg, experiment_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    assert isinstance(cfg, DictConfig)
    return cfg
