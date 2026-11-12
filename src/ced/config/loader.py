from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from ced.config.models import RuntimeConfig

DEFAULT_CONFIG = Path("src/ced/config/default.yaml")
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"required environment variable is unset: {name}")

    return _ENV.sub(replace, value)


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def load_runtime(path: Path = DEFAULT_CONFIG) -> RuntimeConfig:
    return RuntimeConfig.model_validate(_expand_env(read_yaml(path)))
