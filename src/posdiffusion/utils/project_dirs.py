"""Project directory utilities for path resolution."""

import os
from pathlib import Path


class ProjectDirs:
    """Utility class for resolving project paths regardless of execution context."""

    _root_dir: Path | None = None

    @classmethod
    def get_root_dir(cls) -> Path:
        """Get the project root directory.

        Uses multiple strategies to find the project root:
        1. Environment variable PROJECT_ROOT
        2. Search for pyproject.toml marker file
        3. Fallback to package location

        Returns:
            Path to project root directory

        Raises:
            RuntimeError: If project root cannot be determined
        """
        if cls._root_dir is not None:
            return cls._root_dir

        # Strategy 1: Environment variable
        if root_env := os.environ.get("PROJECT_ROOT"):
            cls._root_dir = Path(root_env).resolve()
            return cls._root_dir

        # Strategy 2: Search for marker files
        current = Path(__file__).parent
        while current != current.parent:
            # Look for common project root markers
            markers = ["pyproject.toml", "setup.py", ".git", "uv.lock"]
            if any((current / marker).exists() for marker in markers):
                cls._root_dir = current
                return cls._root_dir
            current = current.parent

        # Strategy 3: Fallback to package location (src/posdiffusion -> project root)
        # This file is in src/posdiffusion/utils/, so go up 3 levels
        fallback = Path(__file__).parent.parent.parent
        if (fallback / "pyproject.toml").exists():
            cls._root_dir = fallback
            return cls._root_dir

        raise RuntimeError(
            "Could not determine project root directory. "
            "Set PROJECT_ROOT environment variable or ensure pyproject.toml exists."
        )

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory (project_root/data)."""
        return cls.get_root_dir() / "data"

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get the config directory (project_root/configs)."""
        return cls.get_root_dir() / "configs"

    @classmethod
    def get_work_dir(cls) -> Path:
        """Get the work directory (project_root/work_dir)."""
        return cls.get_root_dir() / "work_dir"

    @classmethod
    def get_scripts_dir(cls) -> Path:
        """Get the scripts directory (project_root/scripts)."""
        return cls.get_root_dir() / "scripts"

    @classmethod
    def resolve_path(cls, path: str | Path) -> Path:
        """Resolve a path relative to project root if it's not absolute.

        Args:
            path: Path to resolve (string or Path object)

        Returns:
            Resolved absolute path
        """
        path = Path(path)
        if path.is_absolute():
            return path
        return cls.get_root_dir() / path

    @classmethod
    def resolve_data_path(cls, path: str | Path) -> Path:
        """Resolve a path relative to the data directory.

        Args:
            path: Path relative to data directory

        Returns:
            Resolved absolute path
        """
        return cls.get_data_dir() / path

    @classmethod
    def resolve_config_path(cls, path: str | Path) -> Path:
        """Resolve a path relative to the config directory.

        Args:
            path: Path relative to config directory

        Returns:
            Resolved absolute path
        """
        return cls.get_config_dir() / path
