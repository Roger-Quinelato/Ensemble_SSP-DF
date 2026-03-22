import json
import os

import pytest

from tests.test_integration_train_infer import latest_run_dir  # noqa: F401


def test_manifest_paths_are_relative(request):
    outputs_base = "outputs"
    if not os.path.exists(outputs_base):
        pytest.skip("Execute run_experiment() primeiro")

    run_dir = request.getfixturevalue("latest_run_dir")
    manifest_path = os.path.join(run_dir, "models_saved", "models_manifest.json")
    if not os.path.exists(manifest_path):
        pytest.skip("Execute run_experiment() primeiro")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    paths = []
    for section in ("iso", "hbos", "temporal"):
        for entry in manifest.get(section, []):
            if isinstance(entry, dict) and "path" in entry:
                paths.append(entry["path"])

    for section in ("scaler", "gru_scaler"):
        entry = manifest.get(section, {})
        if isinstance(entry, dict) and entry.get("path"):
            paths.append(entry["path"])

    for p_entry in manifest.get("thresholds", {}).values():
        if isinstance(p_entry, dict) and p_entry.get("path"):
            paths.append(p_entry["path"])

    for path in paths:
        assert not os.path.isabs(path), (
            f"Manifesto nao portatil: path absoluto encontrado: {path}"
        )
