import os
import re
from pathlib import Path

import pytest


RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def find_versioned_runs(base_dir="outputs", require_models_dir=False):
    """Return versioned run directories sorted from newest to oldest."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    runs = []
    for path in base_path.iterdir():
        if not path.is_dir() or not RUN_DIR_PATTERN.match(path.name):
            continue
        if require_models_dir and not (path / "models_saved").is_dir():
            continue
        runs.append(path)
    return sorted(runs, reverse=True)


@pytest.fixture(scope="module")
def latest_run_dir():
    runs = find_versioned_runs(require_models_dir=True)
    if not runs:
        pytest.skip("Nenhuma run encontrada em outputs/ - execute run_experiment() primeiro")
    return str(runs[0])


@pytest.fixture(scope="module")
def models_dir(latest_run_dir):
    models_path = os.path.join(latest_run_dir, "models_saved")
    if not os.path.exists(models_path):
        pytest.skip(f"models_saved nao encontrado em {latest_run_dir}")
    return models_path


@pytest.fixture(scope="module")
def metrics_dir(latest_run_dir):
    return os.path.join(latest_run_dir, "metrics")
