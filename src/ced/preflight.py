from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

from ced.config import RuntimeConfig, is_remote_worker


def require_cpu_worker(config: RuntimeConfig) -> None:
    hostname = socket.gethostname()
    if not is_remote_worker() or not hostname.startswith("wenceslaus"):
        raise RuntimeError(
            "CPU stages must run on wenceslaus through the ced remote worker; local execution is forbidden"
        )
    if config.cpu_workers > 4:
        raise RuntimeError("wenceslaus CPU workers may not exceed four")


def require_program(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required program is unavailable on wenceslaus: {name}")
    return path


def require_disk(path: Path, required_bytes: int, reserve_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(path).free
    needed = required_bytes + int(reserve_gb * 1024**3)
    if available < needed:
        raise RuntimeError(f"insufficient disk at {path}: need {needed} bytes including reserve, have {available}")


def require_fullmap(path: Path) -> Path:
    if not path.is_file() and not path.is_dir():
        raise RuntimeError(f"fullmap does not exist on wenceslaus: {path}")
    from tablassert import rs
    from tablassert.fullmap import fullmap_db_path

    database = fullmap_db_path(path).resolve()
    if not database.is_file() or not os.access(database, os.R_OK):
        raise RuntimeError(f"fullmap database is not readable: {database}")
    rs.hydrate_curies(database, [])
    rs.lookup_fullmap_terms(database, ["__ced_preflight__"], return_format="pairs")
    return database
