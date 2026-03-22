"""
Teste de fluxo do pipeline com layout oficial versionado por run_id.
"""
import os

import pytest

from tests.test_integration_train_infer import latest_run_dir  # noqa: F401


def test_pipeline_flow_outputs_exist(request):
    """
    Valida que os artefatos oficiais existem em outputs/<run_id>/...
    Nao executa o pipeline; apenas verifica uma run ja gerada.
    """
    outputs_base = "outputs"
    if not os.path.exists(outputs_base):
        pytest.skip("Execute run_experiment() primeiro")

    has_any_run = any(
        os.path.isdir(os.path.join(outputs_base, d)) and d != "logs"
        for d in os.listdir(outputs_base)
    )
    if not has_any_run:
        pytest.skip("Execute run_experiment() primeiro")

    run_dir = request.getfixturevalue("latest_run_dir")
    expected_paths = [
        os.path.join(run_dir, "master_table", "resultado_final.parquet"),
        os.path.join(run_dir, "metrics", "perfil_dados.json"),
        os.path.join(run_dir, "metrics", "concordancia_modelos.csv"),
        os.path.join(run_dir, "models_saved", "models_manifest.json"),
    ]
    for path in expected_paths:
        assert os.path.exists(path), f"Arquivo esperado nao encontrado: {path}"
