import json
import os
import re

import pytest


def test_thresholds_are_serialized_after_training(tmp_path):
    """
    Apos run_experiment(), os arquivos thresholds_p{90,95,99}.json
    devem existir em models_saved/.
    FALHA se thresholds nao foram serializados - o que significa que
    inference.py sempre operara em modo degradado.
    """
    _ = tmp_path  # marcador explicito para evitar alerta de argumento nao usado
    models_dir = None

    # Descobrir a pasta de modelos mais recente em outputs/
    outputs_base = "outputs"
    run_pattern = re.compile(r"^\d{8}_\d{6}$")
    if os.path.exists(outputs_base):
        runs = sorted(
            [
                f
                for f in os.listdir(outputs_base)
                if os.path.isdir(os.path.join(outputs_base, f)) and run_pattern.match(f)
            ],
            reverse=True,
        )
        if runs:
            models_dir = os.path.join(outputs_base, runs[0], "models_saved")
    if not models_dir or not os.path.exists(models_dir):
        pytest.skip("Nenhum run encontrado - execute run_experiment() antes")

    for p in [90, 95, 99]:
        thresh_path = os.path.join(models_dir, f"thresholds_p{p}.json")
        assert os.path.exists(thresh_path), (
            f"BLOQUEANTE: thresholds_p{p}.json nao encontrado em {models_dir}. "
            f"inference.py operara em modo degradado recalibrando nos dados de producao."
        )
        with open(thresh_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0, f"thresholds_p{p}.json esta vazio"
        for model_name, thresh in data.items():
            assert isinstance(thresh, (int, float)), (
                f"Threshold de {model_name} nao e numerico: {thresh}"
            )

    # Verificar que inference funciona sem modo degradado
    from src.pipeline.inference import load_thresholds

    thresholds = load_thresholds(models_dir, percentile=95)
    assert thresholds is not None, (
        "load_thresholds() retornou None mesmo com arquivo existente"
    )

    manifest_path = os.path.join(models_dir, "models_manifest.json")
    assert os.path.exists(manifest_path), "models_manifest.json nao encontrado"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for section in ("iso", "hbos", "temporal"):
        for entry in manifest.get(section, []):
            sha = entry.get("sha256")
            assert isinstance(sha, str) and len(sha) == 64, (
                f"Manifesto sem sha256 valido em {section}: {entry}"
            )

    for scaler_key in ("scaler", "gru_scaler"):
        if scaler_key in manifest:
            sha = manifest[scaler_key].get("sha256")
            assert isinstance(sha, str) and len(sha) == 64, (
                f"Manifesto sem sha256 valido para {scaler_key}"
            )
