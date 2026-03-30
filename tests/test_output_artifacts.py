import json
from pathlib import Path
import pytest


pytestmark = [pytest.mark.integration]


def test_pipeline_flow_outputs_exist(latest_run_dir):
    """
    Valida que os artefatos oficiais existem em outputs/<run_id>/...
    Nao executa o pipeline; apenas verifica a run mais recente ja gerada.
    """
    run_dir = Path(latest_run_dir)
    expected_paths = [
        run_dir / "master_table" / "resultado_final.parquet",
        run_dir / "metrics" / "perfil_dados.json",
        run_dir / "metrics" / "concordancia_modelos.csv",
        run_dir / "models_saved" / "models_manifest.json",
    ]

    for path in expected_paths:
        assert path.exists(), f"Arquivo esperado nao encontrado: {path}"


def test_models_saved_generated(models_dir):
    models_path = Path(models_dir)
    assert models_path.is_dir(), f"Diretorio de modelos ausente: {models_path}"
    assert any(models_path.iterdir()), f"Nenhum arquivo gerado em {models_path}"


def test_manifest_paths_are_relative(models_dir):
    manifest_path = Path(models_dir) / "models_manifest.json"
    if not manifest_path.exists():
        raise AssertionError(f"models_manifest.json nao encontrado em {models_dir}")

    with manifest_path.open(encoding="utf-8") as f:
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

    for threshold_entry in manifest.get("thresholds", {}).values():
        if isinstance(threshold_entry, dict) and threshold_entry.get("path"):
            paths.append(threshold_entry["path"])

    for path in paths:
        assert not Path(path).is_absolute(), (
            f"Manifesto nao portatil: path absoluto encontrado: {path}"
        )


def test_report_is_generated_and_complete(latest_run_dir):
    report_path = Path(latest_run_dir) / "relatorio_executivo.html"
    assert report_path.exists(), f"Relatorio ausente em {report_path}"

    size = report_path.stat().st_size
    assert size > 10_000, (
        f"Relatorio em {report_path} tem apenas {size} bytes - parece incompleto"
    )

    content = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Relatorio de Anomalias" in content or "RelatÃ³rio de Anomalias" in content
    assert "ensemble_alert" in content or "Alertas" in content
    assert "Aviso Metodologico" in content or "Aviso MetodolÃ³gico" in content
    assert "Limitation" in content or "Limitacao" in content or "LimitaÃ§Ã£o" in content
